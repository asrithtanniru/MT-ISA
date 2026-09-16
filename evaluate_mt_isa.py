"""
MT-ISA evaluation.

Default test file is Laptop14 `laptop14_test_all.json`.
For the ISA result, this script filters to `is_implicit == True`.

The checkpoint metadata is used to instantiate the same backbone and
D-AWL/T-AWL configuration used during training.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import T5Tokenizer

from mt_isa_model import MTISAModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ID_TO_POLARITY = {
    0: "positive",
    1: "negative",
    2: "neutral",
}


class Evaluator:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device

        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )

        model_name = checkpoint.get(
            "model_name", "google/flan-t5-base"
        )
        d_awl_strategy = checkpoint.get("d_awl_strategy", "input")
        t_awl_version = checkpoint.get("t_awl_version", "alf2")

        logger.info("Loading backbone: %s", model_name)
        logger.info("D-AWL: %s | T-AWL: %s",
                    d_awl_strategy, t_awl_version)

        self.model = MTISAModel(
            model_name=model_name,
            d_awl_strategy=d_awl_strategy,
            t_awl_version=t_awl_version,
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def predict(self, sentence: str, target: str):
        text = f"sentiment polarity: {sentence} [SEP] {target}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        logits = self.model.predict_polarity(
            inputs["input_ids"],
            inputs["attention_mask"],
        )
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, pred_id].item())

        return ID_TO_POLARITY[pred_id], confidence

    def evaluate(
        self,
        instances,
        output_path=None,
        show_errors=10,
    ):
        gold = []
        pred = []
        predictions = []

        for inst in instances:
            gold_label = inst["gold_polarity"].lower()
            predicted, confidence = self.predict(
                inst["sentence"], inst["target"]
            )

            gold.append(gold_label)
            pred.append(predicted)

            predictions.append(
                {
                    "instance_id": inst["id"],
                    "sentence": inst["sentence"],
                    "target": inst["target"],
                    "gold_polarity": gold_label,
                    "predicted_polarity": predicted,
                    "confidence": confidence,
                    "correct": predicted == gold_label,
                }
            )

        metrics = {
            "accuracy": float(accuracy_score(gold, pred)),
            "macro_f1": float(
                f1_score(
                    gold, pred,
                    labels=["positive", "negative", "neutral"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_precision": float(
                precision_score(
                    gold, pred,
                    labels=["positive", "negative", "neutral"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    gold, pred,
                    labels=["positive", "negative", "neutral"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "correct": int(sum(x == y for x, y in zip(gold, pred))),
            "total": len(gold),
        }

        print("\n" + "=" * 65)
        print("MT-ISA LAPTOP14 ISA EVALUATION")
        print("=" * 65)
        print(f"Instances: {metrics['total']}")
        print(
            f"Accuracy: {metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total']})"
        )
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
        print("=" * 65)

        print("\nClassification report:")
        print(
            classification_report(
                gold,
                pred,
                labels=["positive", "negative", "neutral"],
                zero_division=0,
            )
        )

        errors = [x for x in predictions if not x["correct"]]
        print(
            f"Example errors ({min(show_errors, len(errors))} "
            f"of {len(errors)}):"
        )
        for i, e in enumerate(errors[:show_errors], 1):
            print(f"\n{i}. {e['instance_id']}")
            print(f"   Sentence: {e['sentence']}")
            print(f"   Target: {e['target']}")
            print(f"   Gold: {e['gold_polarity']}")
            print(
                f"   Predicted: {e['predicted_polarity']} "
                f"(confidence={e['confidence']:.3f})"
            )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"metrics": metrics, "predictions": predictions},
                    f,
                    indent=2,
                )
            logger.info("Saved predictions to %s", output_path)

        return metrics, predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to checkpoint-best.pt",
    )
    parser.add_argument(
        "--test-data",
        default="data/processed/laptop14_test_all.json",
    )
    parser.add_argument(
        "--output",
        default="models/mt_isa_base_input/laptop14_isa_predictions.json",
    )
    parser.add_argument("--show-errors", type=int, default=10)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all test records instead of only implicit records.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(args.test_data, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not args.all:
        before = len(instances)
        instances = [
            x for x in instances
            if bool(x.get("is_implicit", False))
        ]
        logger.info(
            "ISA filter: %d/%d test records are implicit",
            len(instances),
            before,
        )
    else:
        logger.info("Evaluating ALL test records: %d", len(instances))

    evaluator = Evaluator(args.model_path, device)
    evaluator.evaluate(
        instances,
        output_path=args.output,
        show_errors=args.show_errors,
    )


if __name__ == "__main__":
    main()
