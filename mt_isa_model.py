"""
MT-ISA model with D-AWL input/output weighting and numerically stable ALF2 T-AWL.

Fixes:
- D-AWL input: confidence-scaled T5 embeddings are actually used.
- D-AWL output: confidence is applied per-example before reduction.
- D-AWL input_output: both are applied.
- Auxiliary padding tokens are converted to -100.
- ALF2 follows the paper objective with bounded log(sigma^2) for stability.
- Non-finite losses are detected immediately with diagnostics.

The bounded log(sigma^2) is a numerical-stability safeguard; it does not
change the intended ALF2 objective in the normal operating range.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


class DataLevelAWL(nn.Module):
    VALID = {"input", "output", "input_output"}

    def __init__(self, strategy: str = "input"):
        super().__init__()
        if strategy not in self.VALID:
            raise ValueError(
                f"Unknown D-AWL strategy: {strategy}. "
                f"Choose one of {sorted(self.VALID)}."
            )
        self.strategy = strategy

    @staticmethod
    def scale_embeddings(
        embeddings: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Input D-AWL: e_i = c_i * Emb(x_i)."""
        c = confidence_scores.reshape(-1).to(
            device=embeddings.device, dtype=embeddings.dtype
        ).view(-1, 1, 1)
        return embeddings * c

    @staticmethod
    def weighted_token_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output D-AWL: calculate per-example mean token NLL, then weight each
        example by its confidence before batch averaging.
        """
        confidence_scores = confidence_scores.reshape(-1)

        if logits.size(0) != labels.size(0):
            raise ValueError("Logits and labels have different batch sizes.")
        if confidence_scores.size(0) != labels.size(0):
            raise ValueError("Confidence and labels have different batch sizes.")

        vocab = logits.size(-1)
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.size(0), labels.size(1))

        valid = labels.ne(-100)
        token_counts = valid.sum(dim=1).clamp_min(1).to(token_loss.dtype)
        per_example_loss = (
            token_loss * valid.to(token_loss.dtype)
        ).sum(dim=1) / token_counts

        c = confidence_scores.to(
            device=logits.device, dtype=per_example_loss.dtype
        )
        return (per_example_loss * c).mean()


class TaskLevelAWL(nn.Module):
    """
    Homoscedastic task uncertainty with ALF1/ALF2.

    ALF2:
        L = La/sigma_a^2 + Lo/sigma_o^2 + Lp/sigma_p^2
            + sum_k log(sigma_k^2 + 1)

    log_sigma_sq represents log(sigma^2), initialized at zero.
    """

    MIN_LOG_SIGMA_SQ = -10.0
    MAX_LOG_SIGMA_SQ = 10.0

    def __init__(self, num_tasks: int = 3, alf_version: str = "alf2"):
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if alf_version not in {"alf1", "alf2"}:
            raise ValueError("alf_version must be 'alf1' or 'alf2'.")

        self.num_tasks = num_tasks
        self.alf_version = alf_version
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))

    def _bounded_log_sigma_sq(self) -> torch.Tensor:
        return torch.clamp(
            self.log_sigma_sq,
            min=self.MIN_LOG_SIGMA_SQ,
            max=self.MAX_LOG_SIGMA_SQ,
        )

    def _sigma_sq(self) -> torch.Tensor:
        return torch.exp(self._bounded_log_sigma_sq())

    def forward(self, losses: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task losses, got {len(losses)}"
            )

        for i, loss in enumerate(losses):
            if loss is None:
                raise ValueError(f"Task loss {i} is None.")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite task loss before ALF2: "
                    f"task={i}, loss={loss.detach().item()}"
                )

        sigma_sq = self._sigma_sq()

        weighted = [
            losses[k] / sigma_sq[k] for k in range(self.num_tasks)
        ]

        if self.alf_version == "alf1":
            reg = torch.sum(self._bounded_log_sigma_sq())
        else:
            reg = torch.sum(torch.log(sigma_sq + 1.0))

        combined = sum(weighted) + reg

        if not torch.isfinite(combined):
            raise FloatingPointError(
                "ALF produced a non-finite loss. "
                f"log_sigma_sq={self.log_sigma_sq.detach().cpu().tolist()}, "
                f"sigma_sq={sigma_sq.detach().cpu().tolist()}, "
                f"task_losses={[float(x.detach().cpu()) for x in losses]}"
            )

        return combined

    @torch.no_grad()
    def get_task_weights(self) -> Dict[str, float]:
        sigma_sq = self._sigma_sq()
        weights = 1.0 / sigma_sq
        names = ["aspect", "opinion", "polarity"]
        return {
            names[k] if k < len(names) else f"task_{k}":
            float(weights[k].item())
            for k in range(self.num_tasks)
        }


class MTISAModel(nn.Module):
    """MT-ISA with Flan-T5 backbone and the repository's polarity head."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        d_awl_strategy: str = "input",
        t_awl_version: str = "alf2",
        num_polarity_classes: int = 3,
    ):
        super().__init__()

        self.model_name = model_name
        self.d_awl_strategy = d_awl_strategy
        self.t_awl_version = t_awl_version
        self.num_polarity_classes = num_polarity_classes

        self.backbone = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        self.hidden_dim = self.backbone.config.d_model
        self.d_awl = DataLevelAWL(d_awl_strategy)
        self.t_awl = TaskLevelAWL(3, t_awl_version)

        self.polarity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_polarity_classes),
        )

    def _prepare_labels(
        self, labels: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Convert tokenizer padding IDs to -100 for ignored NLL tokens."""
        if labels is None:
            return None
        labels = labels.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def _auxiliary_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        confidence=None,
        strategy="input",
    ):
        """
        Auxiliary T5 forward with numerically safe D-AWL.

        D-AWL:
          input:       confidence scales encoder input embeddings
          output:      confidence weights per-example auxiliary loss
          input_output: both

        Padding target tokens are ignored. Examples with no valid target
        tokens are excluded from the auxiliary-loss average.
        """
        device = input_ids.device

        # Ensure labels are safe for T5 CrossEntropy/NLL.
        labels = labels.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        valid_mask = labels.ne(-100)
        valid_counts = valid_mask.sum(dim=1)
        valid_examples = valid_counts.gt(0)

        # A completely empty target batch cannot produce a meaningful
        # supervised auxiliary loss. Return a finite zero loss.
        if not valid_examples.any():
            return None, torch.zeros((), device=device, dtype=torch.float32)

        # Confidence is per-instance.
        if confidence is None:
            confidence = torch.ones(
                input_ids.size(0), device=device, dtype=torch.float32
            )
        else:
            confidence = confidence.to(device=device, dtype=torch.float32).reshape(-1)

        if confidence.numel() != input_ids.size(0):
            raise ValueError(
                f"Confidence size {confidence.numel()} does not match batch size "
                f"{input_ids.size(0)}"
            )

        if not torch.isfinite(confidence).all():
            raise FloatingPointError(
                f"Non-finite D-AWL confidence: "
                f"min={torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).min().item():.6g}, "
                f"max={torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).max().item():.6g}"
            )

        confidence = confidence.clamp(0.0, 1.0)

        # Input D-AWL: scale encoder embeddings by confidence.
        inputs_embeds = None
        if strategy in ("input", "input_output"):
            inputs_embeds = self.t5.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds * confidence.view(-1, 1, 1).to(
                dtype=inputs_embeds.dtype
            )

            if not torch.isfinite(inputs_embeds).all():
                raise FloatingPointError(
                    "Non-finite inputs_embeds after D-AWL confidence scaling"
                )

        # Forward T5. Do not ask HF to compute the reduced loss; we compute
        # a masked per-example token NLL below so D-AWL can be instance-level.
        outputs = self.t5(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite T5 auxiliary logits"
            )

        # Compute NLL in float32 for numerical stability.
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        safe_labels = labels.clamp_min(0)
        selected = log_probs.gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        token_nll = -selected
        token_nll = token_nll * valid_mask.float()

        per_example_loss = token_nll.sum(dim=1) / valid_counts.clamp_min(1).float()

        # Only examples with at least one valid target token participate.
        per_example_loss = per_example_loss[valid_examples]
        example_conf = confidence[valid_examples]

        if strategy in ("output", "input_output"):
            # Output D-AWL: confidence weights the per-instance loss.
            denom = example_conf.sum().clamp_min(1e-8)
            loss = (per_example_loss * example_conf).sum() / denom
        else:
            loss = per_example_loss.mean()

        loss = loss.float()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite auxiliary loss with D-AWL '{strategy}': {loss.item()}"
            )

        return outputs, loss

    def forward(
        self,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        opinion_input_ids: torch.Tensor,
        opinion_attention_mask: torch.Tensor,
        polarity_input_ids: torch.Tensor,
        polarity_attention_mask: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        aspect_confidence: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        opinion_confidence: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        polarity_label_id: Optional[torch.Tensor] = None,
        return_losses: bool = True,
    ) -> Dict:

        outputs = {}

        if aspect_input_ids is not None:
            aspect_output, aspect_loss = self._auxiliary_forward(
                aspect_input_ids,
                aspect_attention_mask,
                aspect_labels,
                aspect_confidence,
            )
            outputs["aspect_loss"] = aspect_loss
            outputs["aspect_logits"] = aspect_output.logits

        if opinion_input_ids is not None:
            opinion_output, opinion_loss = self._auxiliary_forward(
                opinion_input_ids,
                opinion_attention_mask,
                opinion_labels,
                opinion_confidence,
            )
            outputs["opinion_loss"] = opinion_loss
            outputs["opinion_logits"] = opinion_output.logits

        if polarity_input_ids is not None:
            encoder_output = self.backbone.encoder(
                input_ids=polarity_input_ids,
                attention_mask=polarity_attention_mask,
            )
            hidden = encoder_output.last_hidden_state[:, 0, :]
            polarity_logits = self.polarity_head(hidden)
            outputs["polarity_logits"] = polarity_logits

            if polarity_label_id is not None:
                outputs["polarity_loss"] = F.cross_entropy(
                    polarity_logits, polarity_label_id.long()
                )
            elif polarity_labels is not None:
                labels = self._prepare_labels(polarity_labels)
                outputs["polarity_loss"] = self.backbone(
                    input_ids=polarity_input_ids,
                    attention_mask=polarity_attention_mask,
                    labels=labels,
                ).loss

        if (
            return_losses
            and outputs.get("aspect_loss") is not None
            and outputs.get("opinion_loss") is not None
            and outputs.get("polarity_loss") is not None
        ):
            task_losses = (
                outputs["aspect_loss"],
                outputs["opinion_loss"],
                outputs["polarity_loss"],
            )
            outputs["combined_loss"] = self.t_awl(task_losses)
            outputs["task_weights"] = self.t_awl.get_task_weights()

        return outputs

    @torch.no_grad()
    def predict_polarity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = encoder_output.last_hidden_state[:, 0, :]
        return self.polarity_head(hidden)

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "d_awl_strategy": self.d_awl_strategy,
            "t_awl_version": self.t_awl_version,
            "task_weights": self.t_awl.get_task_weights(),
        }
