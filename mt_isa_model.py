"""
MT-ISA model with faithful D-AWL input/output weighting and ALF2 T-AWL.

This version keeps the repository's existing polarity classification head and
task inputs, but fixes:
1) D-AWL input strategy: confidence-scaled input embeddings are actually fed
   into Flan-T5.
2) D-AWL output strategy: confidence is applied per training instance before
   reduction, not as batch-mean scaling.
3) Padding tokens in auxiliary labels are ignored in NLL.
4) ALF2 follows Eq. (9) of the MT-ISA paper.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer


logger = logging.getLogger(__name__)


class DataLevelAWL(nn.Module):
    """Data-Level Automatic Weight Learning."""

    VALID = {"input", "output", "input_output"}

    def __init__(self, strategy: str = "input"):
        super().__init__()
        if strategy not in self.VALID:
            raise ValueError(f"Unknown D-AWL strategy: {strategy}")
        self.strategy = strategy

    @staticmethod
    def scale_embeddings(
        embeddings: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (4): e_i = c_i * Emb(x_i)."""
        c = confidence_scores.to(
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
        Eq. (6)/(7), implemented as a per-example mean token NLL followed by
        confidence weighting and batch averaging.

        labels == -100 are ignored.
        """
        vocab = logits.size(-1)
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.size(0), labels.size(1))

        valid = labels.ne(-100)
        token_counts = valid.sum(dim=1).clamp_min(1)
        per_example_loss = (token_loss * valid).sum(dim=1) / token_counts

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
    """

    def __init__(self, num_tasks: int = 3, alf_version: str = "alf2"):
        super().__init__()
        if alf_version not in {"alf1", "alf2"}:
            raise ValueError(f"Unknown ALF version: {alf_version}")
        self.num_tasks = num_tasks
        self.alf_version = alf_version
        # log(sigma^2); sigma^2 starts at 1 => equal initial task weights.
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task losses, got {len(losses)}"
            )

        sigma_sq = torch.exp(self.log_sigma_sq)
        weighted = [
            loss / (sigma_sq[k] + 1e-8)
            for k, loss in enumerate(losses)
        ]

        if self.alf_version == "alf1":
            reg = torch.sum(self.log_sigma_sq)
        else:
            reg = torch.sum(torch.log(sigma_sq + 1.0))

        # Do not introduce an additional /3 normalization: Eq. (9) is a sum.
        return sum(weighted) + reg

    @torch.no_grad()
    def get_task_weights(self) -> Dict[str, float]:
        sigma_sq = torch.exp(self.log_sigma_sq)
        w = 1.0 / (sigma_sq + 1e-8)
        return {
            "aspect": float(w[0].item()),
            "opinion": float(w[1].item()),
            "polarity": float(w[2].item()),
        }


class MTISAModel(nn.Module):
    """
    MT-ISA with Flan-T5 encoder-decoder backbone and a 3-class polarity head.

    The auxiliary tasks use the shared T5 backbone:
      aspect  -> generated aspect
      opinion -> generated opinion
    The primary task uses the encoder representation for polarity.
    """

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

    def _prepare_labels(self, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Convert tokenizer padding IDs to -100 so padding is ignored."""
        if labels is None:
            return None
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def _auxiliary_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
    ):
        labels = self._prepare_labels(labels)

        # D-AWL input: confidence-scaled embeddings.
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        if confidence is not None and self.d_awl_strategy in {
            "input", "input_output"
        }:
            inputs_embeds = self.d_awl.scale_embeddings(
                inputs_embeds, confidence
            )

        output = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        if labels is None:
            return output, None

        if confidence is not None and self.d_awl_strategy in {
            "output", "input_output"
        }:
            loss = self.d_awl.weighted_token_loss(
                output.logits, labels, confidence
            )
        else:
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return output, loss

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

        # Aspect auxiliary task.
        if aspect_input_ids is not None:
            aspect_output, aspect_loss = self._auxiliary_forward(
                aspect_input_ids,
                aspect_attention_mask,
                aspect_labels,
                aspect_confidence,
            )
            outputs["aspect_loss"] = aspect_loss
            outputs["aspect_logits"] = aspect_output.logits

        # Opinion auxiliary task.
        if opinion_input_ids is not None:
            opinion_output, opinion_loss = self._auxiliary_forward(
                opinion_input_ids,
                opinion_attention_mask,
                opinion_labels,
                opinion_confidence,
            )
            outputs["opinion_loss"] = opinion_loss
            outputs["opinion_logits"] = opinion_output.logits

        # Primary polarity task.
        if polarity_input_ids is not None:
            encoder_output = self.backbone.encoder(
                input_ids=polarity_input_ids,
                attention_mask=polarity_attention_mask,
            )
            # Keep the repository's established first-token representation.
            cls_hidden = encoder_output.last_hidden_state[:, 0, :]
            polarity_logits = self.polarity_head(cls_hidden)
            outputs["polarity_logits"] = polarity_logits

            if polarity_label_id is not None:
                outputs["polarity_loss"] = F.cross_entropy(
                    polarity_logits, polarity_label_id.long()
                )
            elif polarity_labels is not None:
                # Optional text-generation polarity path.
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
            outputs["combined_loss"] = self.t_awl(
                (
                    outputs["aspect_loss"],
                    outputs["opinion_loss"],
                    outputs["polarity_loss"],
                )
            )
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
