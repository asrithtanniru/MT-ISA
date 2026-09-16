"""
MT-ISA training script.

Important experimental choices:
- Flan-T5-base by default (250M).
- batch size 64, LR 1e-5, AdamW, linear decay, max 20 epochs.
- Validation is created from the training set if no separate validation JSON
  is supplied, so the test set is never used for early stopping.
- Best checkpoint is selected by validation macro-F1, as specified by the paper.
- Existing auxiliary JSON is used; no auxiliary regeneration occurs here.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer

from mt_isa_model import MTISAModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

POLARITY_TO_ID = {"positive": 0, "negative": 1, "neutral": 2}
ID_TO_POLARITY = {v: k for k, v in POLARITY_TO_ID.items()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_validation_split(
    instances: List[Dict],
    validation_fraction: float,
    seed: int,
):
    """Stratified split by polarity, without touching the test set."""
    rng = random.Random(seed)
    by_label = {label: [] for label in POLARITY_TO_ID}
    for x in instances:
        by_label[x["gold_polarity"].lower()].append(x)

    train, val = [], []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * validation_fraction)))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        instances: List[Dict],
        tokenizer,
        auxiliary_data: Optional[List[Dict]] = None,
        max_length: int = 256,
        require_auxiliary: bool = False,
    ):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.auxiliary_by_id = {
            x["instance_id"]: x for x in (auxiliary_data or [])
        }

        if require_auxiliary:
            missing = [
                x["id"]
                for x in instances
                if x["id"] not in self.auxiliary_by_id
            ]
            if missing:
                raise ValueError(
                    f"{len(missing)} training instances have no auxiliary "
                    f"record. First missing IDs: {missing[:10]}"
                )

    def __len__(self):
        return len(self.instances)

    def _tokenize(self, text: str, max_length: int):
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        inst = self.instances[idx]
        instance_id = inst["id"]
        sentence = inst["sentence"]
        target = inst["target"]
        gold = inst["gold_polarity"].lower()

        aux = self.auxiliary_by_id.get(instance_id)

        if aux is not None:
            aspect_text = (aux.get("aspect") or "").strip() or target
            opinion_text = (aux.get("opinion") or "").strip() or "none"
            aspect_conf = float(aux.get("aspect_confidence", 0.5))
            opinion_conf = float(aux.get("opinion_confidence", 0.5))
        else:
            # Used only for validation/evaluation datasets where auxiliary
            # labels are intentionally not required.
            aspect_text = target
            opinion_text = (inst.get("opinion_term") or "none").strip()
            aspect_conf = 1.0
            opinion_conf = 1.0

        aspect_input = f"aspect: {sentence} [SEP] {target}"
        opinion_input = (
            f"opinion: {sentence} [SEP] {target} [SEP] {aspect_text}"
        )
        polarity_input = f"sentiment polarity: {sentence} [SEP] {target}"

        aspect_tokens = self._tokenize(aspect_input, self.max_length)
        opinion_tokens = self._tokenize(opinion_input, self.max_length)
        polarity_tokens = self._tokenize(polarity_input, self.max_length)

        aspect_label = self._tokenize(aspect_text, 64)["input_ids"].squeeze(0)
        opinion_label = self._tokenize(opinion_text, 64)["input_ids"].squeeze(0)

        return {
            "instance_id": instance_id,
            "sentence": sentence,
            "target": target,
            "gold_polarity": gold,
            "aspect_input_ids": aspect_tokens["input_ids"].squeeze(0),
            "aspect_attention_mask": aspect_tokens["attention_mask"].squeeze(0),
            "aspect_labels": aspect_label,
            "aspect_confidence": torch.tensor(aspect_conf, dtype=torch.float),
            "opinion_input_ids": opinion_tokens["input_ids"].squeeze(0),
            "opinion_attention_mask": opinion_tokens["attention_mask"].squeeze(0),
            "opinion_labels": opinion_label,
            "opinion_confidence": torch.tensor(opinion_conf, dtype=torch.float),
            "polarity_input_ids": polarity_tokens["input_ids"].squeeze(0),
            "polarity_attention_mask": polarity_tokens["attention_mask"].squeeze(0),
            "polarity_label_id": torch.tensor(
                POLARITY_TO_ID[gold], dtype=torch.long
            ),
        }


def move_batch(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def run_epoch(model, loader, optimizer, scheduler, device, train=True):
    model.train(train)
    total_loss = 0.0
    all_gold, all_pred = [], []

    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="Training" if train else "Validation"):
        batch = move_batch(batch, device)

        with torch.set_grad_enabled(train):
            out = model(
                aspect_input_ids=batch["aspect_input_ids"],
                aspect_attention_mask=batch["aspect_attention_mask"],
                aspect_labels=batch["aspect_labels"],
                aspect_confidence=batch["aspect_confidence"],
                opinion_input_ids=batch["opinion_input_ids"],
                opinion_attention_mask=batch["opinion_attention_mask"],
                opinion_labels=batch["opinion_labels"],
                opinion_confidence=batch["opinion_confidence"],
                polarity_input_ids=batch["polarity_input_ids"],
                polarity_attention_mask=batch["polarity_attention_mask"],
                polarity_label_id=batch["polarity_label_id"],
                return_losses=True,
            )
            loss = out["combined_loss"]

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.detach().item())

        pred = out["polarity_logits"].argmax(dim=-1).detach().cpu().tolist()
        gold = batch["polarity_label_id"].detach().cpu().tolist()
        all_pred.extend(pred)
        all_gold.extend(gold)

    avg_loss = total_loss / max(1, len(loader))
    macro_f1 = f1_score(
        all_gold,
        all_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    accuracy = float(np.mean(np.array(all_gold) == np.array(all_pred)))

    return avg_loss, macro_f1, accuracy


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_f1):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "val_f1": float(val_f1),
            "model_name": model.model_name,
            "d_awl_strategy": model.d_awl_strategy,
            "t_awl_version": model.t_awl_version,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        default="data/processed/train_implicit.json",
    )
    parser.add_argument(
        "--aux-data",
        default="data/auxiliary/train_implicit_aux.json",
    )
    parser.add_argument(
        "--val-data",
        default=None,
        help="Optional separate validation JSON. If omitted, split train data.",
    )
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-base",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--d-awl-strategy", default="input",
                        choices=["input", "output", "input_output"])
    parser.add_argument("--t-awl-version", default="alf2",
                        choices=["alf1", "alf2"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default="models/mt_isa_base_input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_instances = load_json(args.train_data)
    aux_data = load_json(args.aux_data)

    logger.info("Training instances: %d", len(train_instances))
    logger.info("Auxiliary records: %d", len(aux_data))

    # Prevent silent auxiliary-data mismatch.
    aux_ids = {x["instance_id"] for x in aux_data}
    train_ids = {x["id"] for x in train_instances}
    overlap = train_ids & aux_ids
    logger.info(
        "Auxiliary coverage: %d/%d training instances (%.2f%%)",
        len(overlap),
        len(train_ids),
        100.0 * len(overlap) / max(1, len(train_ids)),
    )

    train_instances = [
        x for x in train_instances if x.get("is_implicit", True)
    ]

    if args.val_data:
        val_instances = load_json(args.val_data)
        val_instances = [
            x for x in val_instances if x.get("is_implicit", True)
        ]
    else:
        train_instances, val_instances = make_validation_split(
            train_instances,
            args.validation_fraction,
            args.seed,
        )

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    train_ds = AspectSentimentDataset(
        train_instances,
        tokenizer,
        aux_data,
        args.max_length,
        require_auxiliary=True,
    )
    val_ds = AspectSentimentDataset(
        val_instances,
        tokenizer,
        auxiliary_data=None,
        max_length=args.max_length,
        require_auxiliary=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Train split: %d", len(train_ds))
    logger.info("Validation split: %d", len(val_ds))

    model = MTISAModel(
        model_name=args.model_name,
        d_awl_strategy=args.d_awl_strategy,
        t_awl_version=args.t_awl_version,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            0.0, 1.0 - (step / max(1, total_steps))
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        logger.info("=" * 70)
        logger.info("Epoch %d/%d", epoch, args.num_epochs)

        train_loss, train_f1, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler, device, train=True
        )
        val_loss, val_f1, val_acc = run_epoch(
            model, val_loader, None, None, device, train=False
        )

        weights = model.t_awl.get_task_weights()

        logger.info(
            "Train loss %.4f | Train F1 %.4f | Train Acc %.4f",
            train_loss, train_f1, train_acc
        )
        logger.info(
            "Val loss %.4f | Val F1 %.4f | Val Acc %.4f",
            val_loss, val_f1, val_acc
        )
        logger.info("Task weights: %s", weights)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "task_weights": weights,
            }
        )

        save_checkpoint(
            output_dir / f"checkpoint-epoch-{epoch}.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            val_f1,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                output_dir / "checkpoint-best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_f1,
            )
            logger.info("Saved new best checkpoint: val macro-F1=%.4f", val_f1)
        else:
            patience_counter += 1
            logger.info(
                "No validation-F1 improvement: %d/%d",
                patience_counter,
                args.patience,
            )
            if patience_counter >= args.patience:
                logger.info("Early stopping.")
                break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best validation macro-F1: %.4f", best_f1)


if __name__ == "__main__":
    main()
