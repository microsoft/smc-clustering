"""
PyTorch Lightning module that trains a next-token LM with grammar-constrained NLL.

This module wraps an arbitrary model that maps input token IDs → logits [B, T, V] and enforces the project grammar
(<K>/<V>, quotes, JSON punctuation) via per-step allowed-token masks. Targets are teacher-forced (shifted by one),
masks are computed from the tokenizer's grammar automaton, and a constrained NLL is minimized. An auxiliary metric
(`invalid_mass`) reports how much raw probability the model assigns to illegal tokens before masking.
"""

from __future__ import annotations

import logging
import math

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from jsonlm.grammar.runtime import get_runtime
from jsonlm.models.criterion import constrained_nll, invalid_mass
from jsonlm.tokenization.tokenizer import JsonLMTokenizer


class LitConstrainedLM(LightningModule):
    """Lightning wrapper for grammar-constrained language modeling.

    Args:
        model: A torch.nn.Module that accepts input_ids [B, T] and returns logits [B, T, V].
        tokenizer: The tokenizer describing joint vocabulary (specials + BPE).
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Number of warmup steps for learning rate schedule.
        max_steps: Total training steps for cosine decay; if None, uses constant LR.
        struct_weight: Weight for structure/EOS tokens in loss (1.0 = no down-weight).
        downweight_eos: Whether to down-weight EOS tokens in loss.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: JsonLMTokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps_override: int | None = None,
        struct_weight: float = 0.5,
        downweight_eos: bool = True,
    ) -> None:
        """Initialize the LitConstrainedLM."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps_override = max_steps_override
        self.struct_weight = struct_weight
        self.downweight_eos = downweight_eos

        # Save hyperparameters for Lightning checkpoints (omit large objects).
        self.save_hyperparameters(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps_override,
                "struct_weight": self.struct_weight,
                "downweight_eos": self.downweight_eos,
                "specials_size": self.tokenizer.specials_size,
                "bpe_size": self.tokenizer.bpe_size,
            },
        )

    def _build_masks_for_batch(self, ids_with_eos: torch.Tensor) -> torch.BoolTensor:
        """Construct [B, T, V] Boolean masks of allowed next tokens for teacher forcing."""
        rt = get_runtime(self.tokenizer, device=ids_with_eos.device)
        return rt.build_masks(ids_with_eos)

    def _build_weights(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Build per-position weights that down-weight structure tokens and optionally EOS.

        Args:
            target_ids: Token IDs of shape [B, T].

        Returns:
            weights: Tensor of shape [B, T] with per-position weights.
        """
        voc = self.tokenizer.vocabulary
        specials = [
            voc.token_id("{"),
            voc.token_id("}"),
            voc.token_id("["),
            voc.token_id("]"),
            voc.token_id(":"),
            voc.token_id(","),
            voc.k_id,
            voc.v_id,
            voc.quote_id,
        ]
        w = torch.ones_like(target_ids, dtype=torch.float32, device=target_ids.device)
        if self.struct_weight < 1.0:
            for tid in specials:
                w = w * (1.0 - (1.0 - self.struct_weight) * (target_ids == tid).float())
        if self.downweight_eos:
            w = w * (1.0 - (1.0 - self.struct_weight) * (target_ids == voc.eos_id).float())
        return w

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the underlying model to produce logits [B, T, V]."""
        logits = self.model(input_ids)  # expected shape [B, T, V]
        assert logits.dim() == 3 and logits.shape[0] == input_ids.shape[0] and logits.shape[1] == input_ids.shape[1], (
            f"Model must return [B, T, V] logits for input [B, T]; got {tuple(logits.shape)} "
            f"for input {tuple(input_ids.shape)}"
        )
        return logits

    def _shared_step(self, batch: torch.Tensor, stage: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute constrained loss + metrics for a batch and log scalar summaries.

        Args:
            batch: Tensor [B, L] long containing BOS…EOS sequences (no padding assumed in MVP).
            stage: A label like 'train' or 'val' for logging keys.

        Returns:
            loss: Scalar tensor used by Lightning for optimization.
            logs: Dict of logged metrics for potential external consumption.
        """
        assert batch.dim() == 2 and batch.dtype == torch.long, f"Batch must be [B, T] long, got {tuple(batch.shape)}"
        # Inputs predict "next" tokens; targets are the shifted stream.
        input_ids = batch[:, :-1]  # [B, T]
        target_ids = batch[:, 1:]  # [B, T]
        B, T = input_ids.shape
        V = len(self.tokenizer)

        # Model forward.
        logits = self.forward(input_ids)  # [B, T, V]
        assert logits.shape == (B, T, V), f"Logits shape mismatch: {tuple(logits.shape)} vs {(B, T, V)}"

        # Grammar masks aligned with targets.
        masks = self._build_masks_for_batch(batch)  # [B, T, V]

        # Build weights for down-weighting structure tokens.
        weights = self._build_weights(target_ids)

        # Constrained NLL.
        loss, nll_per_token = constrained_nll(
            logits,
            target_ids,
            masks,
            reduction="mean",
            weights=weights,
        )  # scal, [B, T]

        # Diagnostics: raw invalid mass before masking.
        inv_mass = invalid_mass(logits.detach(), masks)  # [B, T]
        inv_mass_mean = inv_mass.mean()

        # Simple accuracy (allowed argmax equals target), for quick smoke; masked positions are regular positions.
        with torch.no_grad():
            # Allowed-only logprobs: set disallowed to -inf and argmax.
            masked_logits = logits.masked_fill(~masks, float("-inf"))
            pred = masked_logits.argmax(dim=-1)  # [B, T]
            acc = (pred == target_ids).float().mean()  # scalar

        # Log scalars (Lightning handles aggregation).
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, batch_size=B)
        self.log(f"{stage}/invalid_mass", inv_mass_mean, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)

        logs = {"loss": loss, "invalid_mass": inv_mass_mean, "acc": acc}
        return loss, logs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Lightning training_step: compute loss and return it for optimization."""
        loss, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Lightning validation_step: log metrics; Lightning aggregates automatically."""
        _loss, _ = self._shared_step(batch, stage="val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW with cosine LR schedule with warmup."""
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Compute total steps from Trainer if no explicit override
        if self.max_steps_override is not None and self.max_steps_override > 0:
            total_steps = self.max_steps_override
        else:
            try:
                total_steps = int(self.trainer.estimated_stepping_batches)
            except RuntimeError:
                # No trainer attached (e.g., in unit tests), use default
                logging.warning("No trainer attached; using default total_steps=1000 for LR schedule")
                total_steps = 1000

        # Choose warmup if not provided (e.g., 2k steps or 5%)
        warmup = (
            self.warmup_steps
            if (self.warmup_steps is not None and self.warmup_steps >= 0)
            else max(1, min(2000, int(0.05 * total_steps)))
        )

        def _lr_lambda(step: int) -> float:
            """Learning rate schedule."""
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / max(1, total_steps - warmup)
            floor = 0.10

            return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * progress))

        sch = LambdaLR(opt, lr_lambda=_lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
