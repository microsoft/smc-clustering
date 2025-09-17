"""
PyTorch Lightning module that trains a next-token LM with grammar-constrained NLL.

This module wraps an arbitrary model that maps input token IDs → logits [B, T, V] and enforces the project grammar
(<K>/<V>, quotes, JSON punctuation) via per-step allowed-token masks. Targets are teacher-forced (shifted by one),
masks are computed from the tokenizer's grammar automaton, and a constrained NLL is minimized. An auxiliary metric
(`invalid_mass`) reports how much raw probability the model assigns to illegal tokens before masking.
"""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW

from jsonlm.grammar.automaton import GrammarAutomaton, GrammarState
from jsonlm.grammar.mask import allowed_token_mask
from jsonlm.models.criterion import constrained_nll, invalid_mass
from jsonlm.tokenization.tokenizer import JsonLMTokenizer


class LitConstrainedLM(LightningModule):
    """Lightning wrapper for grammar-constrained language modeling.

    Args:
        model: A torch.nn.Module that accepts input_ids [B, T] and returns logits [B, T, V].
        tokenizer: The tokenizer describing joint vocabulary (specials + BPE).
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: JsonLMTokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay

        # Build a reusable automaton; stateless step() calls will drive per-example states.
        self.automaton = GrammarAutomaton(tokenizer)

        # Save hyperparameters for Lightning checkpoints (omit large objects).
        self.save_hyperparameters(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "specials_size": self.tokenizer.specials_size,
                "bpe_size": self.tokenizer.bpe_size,
            },
        )

    def _build_masks_for_batch(self, ids_with_eos: torch.Tensor) -> torch.BoolTensor:
        """Construct [B, T, V] Boolean masks of allowed next tokens for teacher forcing.

        Given sequences with BOS…EOS, this builds a mask per timestep aligned with targets (i.e., for the prediction
        of token y_t given the prefix y_<t). The final timestep corresponds to EOS; we do not advance the automaton
        after consuming EOS.
        """
        assert ids_with_eos.dim() == 2 and ids_with_eos.dtype == torch.long, "ids_with_eos must be [B, L] long"
        B, L = ids_with_eos.shape
        assert L >= 2, "Need at least BOS and EOS"
        V = len(self.tokenizer)

        T = L - 1
        masks = torch.zeros((B, T, V), dtype=torch.bool, device=ids_with_eos.device)
        eos = self.tokenizer.vocabulary.eos_id

        for b in range(B):
            seq = ids_with_eos[b]  # [L]
            assert seq[0].item() == self.tokenizer.vocabulary.bos_id, "Sequence must start with BOS"

            gs: GrammarState = self.automaton.start()
            for t in range(T):
                y_t = int(seq[t + 1].item())

                # Allowed next tokens before consuming y_t.
                m = allowed_token_mask(gs, self.automaton, self.tokenizer)  # [V]
                masks[b, t] = m.to(device=ids_with_eos.device)

                # If y_t is EOS, allow EOS only for the rest of this row and stop stepping.
                if y_t == eos:
                    if t + 1 < T:
                        masks[b, t + 1 :, :] = False
                        masks[b, t + 1 :, eos] = True
                    break

                # Consume non-EOS gold token in the grammar.
                try:
                    gs = self.automaton.step(gs, y_t)
                except ValueError as e:
                    raise ValueError(f"Automaton reject at b={b}, t={t}, token_id={y_t}: {e}") from e

        return masks

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
        assert batch.dim() == 2 and batch.dtype == torch.long, f"Batch must be [B, L] long, got {tuple(batch.shape)}"
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

        # Constrained NLL.
        loss, nll_per_token = constrained_nll(logits, target_ids, masks, reduction="mean")  # scal, [B, T]

        # Diagnostics: raw invalid mass before masking.
        inv_mass = invalid_mass(logits.detach(), masks)  # [B, T]
        inv_mass_mean = inv_mass.mean()

        # Simple accuracy (allowed argmax equals target), for quick smoke; masked positions are regular positions.
        with torch.no_grad():
            # Allowed-only logprobs: set disallowed to -inf and argmax.
            masked_logits = logits.masked_fill(~masks, float("-inf"))
            pred = masked_logits.argmax(dim=-1)  # [B, T]
            acc = (pred == target_ids).float().mean()

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

    def configure_optimizers(self) -> Any:
        """Configure AdamW with the provided LR/weight decay."""
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
