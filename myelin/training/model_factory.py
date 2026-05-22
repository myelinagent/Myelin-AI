"""Joint-head model factory.

``DeBERTaV3SmallJointHead`` is a tiny wrapper that puts two output
heads on top of a (frozen or trainable) DeBERTa-v3-small backbone:

* utility head: linear → 1 scalar (sigmoid'd by the caller for [0, 1])
* class head: linear → 3 logits over {transient, short_term, persistent}

``build_joint_model(use_pretrained=True)`` loads the HF backbone.
``use_pretrained=False`` returns a tiny dummy backbone so the smoke
test can run on CPU in seconds without downloading 150MB.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeBERTaV3SmallJointHead(nn.Module):
    """Backbone + two heads. ``forward`` returns ``(util_logit, class_logits)``."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.utility_head = nn.Linear(hidden_size, 1)
        self.class_head = nn.Linear(hidden_size, 3)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The wrapped backbone is expected to return either a HuggingFace
        # output object with .last_hidden_state, or a (B, T, H) tensor
        # directly (smoke-test path).
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out
        # Mean-pool over sequence dimension (mask-aware).
        if attention_mask is not None and hidden.dim() == 3:
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        elif hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden
        util_logit = self.utility_head(pooled)
        class_logits = self.class_head(pooled)
        return util_logit, class_logits


class _TinyBackbone(nn.Module):
    """Minimal backbone for the smoke test — embedding + mean."""

    def __init__(self, vocab_size: int = 30000, hidden: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.hidden = hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embedding(input_ids)


def build_joint_model(
    *,
    use_pretrained: bool = True,
    base_model_id: str = "microsoft/deberta-v3-small",
) -> DeBERTaV3SmallJointHead:
    """Build the joint-head model.

    use_pretrained=True downloads the DeBERTa-v3-small backbone from HF
    (~150MB). use_pretrained=False uses a 64-dim toy backbone so the
    smoke test runs in <1s on CPU.
    """
    if not use_pretrained:
        backbone = _TinyBackbone()
        return DeBERTaV3SmallJointHead(backbone, hidden_size=backbone.hidden)
    try:
        from transformers import AutoModel  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"transformers required for use_pretrained=True: {e}"
        ) from e
    backbone = AutoModel.from_pretrained(base_model_id)
    hidden_size = backbone.config.hidden_size
    return DeBERTaV3SmallJointHead(backbone, hidden_size=hidden_size)


__all__ = ["DeBERTaV3SmallJointHead", "build_joint_model"]
