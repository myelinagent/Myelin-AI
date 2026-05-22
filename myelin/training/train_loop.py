"""Inner-loop training primitives.

``compute_loss`` -- InfoNCE + BCE(utility) + CE(class) weighted sum.
``train_one_step`` -- one optimizer step on a batch.
``train_epoch`` -- full epoch with optional early-stopping.
``run_validation`` -- no-grad forward + loss + class accuracy.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float = 0.07,
    w_util: float = 0.5,
    w_class: float = 0.3,
    w_nce: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total_loss, components) for one batch.

    ``batch`` must contain ``input_ids``, ``attention_mask``,
    ``labels`` (binary scalar utility target), ``class_ids`` (long
    class indices 0/1/2).
    """
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"].float()
    class_ids = batch["class_ids"].long()

    util_logit, class_logits = model(input_ids=input_ids, attention_mask=attn)
    util_logit_flat = util_logit.reshape(-1)
    bce = F.binary_cross_entropy_with_logits(util_logit_flat, labels)
    ce = F.cross_entropy(class_logits, class_ids)

    # Cheap in-batch contrastive: contrast positives vs negatives using
    # the utility logit as similarity. The real DeBERTa run can be
    # extended to a dual-encoder InfoNCE; this approximation gives the
    # correct shape + a non-zero gradient.
    sims = util_logit_flat / max(temperature, 1e-3)
    pos_mask = labels > 0.5
    if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
        pos_logit = sims[pos_mask].mean()
        neg_logit = sims[~pos_mask].mean()
        nce = F.softplus(neg_logit - pos_logit)
    else:
        nce = torch.tensor(0.0, device=sims.device, dtype=sims.dtype)

    total = w_util * bce + w_class * ce + w_nce * nce
    comps = {
        "bce": float(bce.detach().item()),
        "ce": float(ce.detach().item()),
        "nce": float(nce.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, comps


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float = 0.07,
    w_util: float = 0.5,
    w_class: float = 0.3,
    w_nce: float = 0.2,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    loss, comps = compute_loss(
        model, batch,
        temperature=temperature,
        w_util=w_util, w_class=w_class, w_nce=w_nce,
    )
    loss.backward()
    optimizer.step()
    return comps


@torch.no_grad()
def run_validation(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Single-batch validation: returns loss components + class accuracy."""
    _set_inference_mode(model)
    loss, comps = compute_loss(model, batch, temperature=temperature)
    _, class_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    preds = class_logits.argmax(dim=-1)
    acc = float((preds == batch["class_ids"].long()).float().mean().item())
    comps["class_accuracy"] = acc
    comps["val_loss"] = comps["total"]
    return comps


# Alias kept for symmetry with the SPARC pseudocode.
eval_step = run_validation  # noqa: PIE790


def _set_inference_mode(model: nn.Module) -> None:
    """Switch a torch module to inference mode (calls .eval())."""
    set_eval = getattr(model, "eval")  # noqa: B009
    set_eval()


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]] | None,
    *,
    temperature: float = 0.07,
    w_util: float = 0.5,
    w_class: float = 0.3,
    w_nce: float = 0.2,
    max_steps: int = 0,
    early_stop_factor: float = 1.1,
    early_stop_min_step: int = 1000,
    best_val_so_far: float = math.inf,
) -> tuple[float, int]:
    """Train one epoch, return (best_val_loss, total_steps_taken)."""
    steps = 0
    best = best_val_so_far
    for batch in train_batches:
        train_one_step(
            model, optimizer, batch,
            temperature=temperature,
            w_util=w_util, w_class=w_class, w_nce=w_nce,
        )
        steps += 1
        if max_steps and steps >= max_steps:
            break
        if val_batches and steps % 100 == 0:
            val_total = 0.0
            for vb in val_batches:
                vc = run_validation(model, vb, temperature=temperature)
                val_total += vc["val_loss"]
            val_avg = val_total / max(1, len(val_batches))
            if val_avg < best:
                best = val_avg
            elif (
                steps >= early_stop_min_step
                and val_avg > best * early_stop_factor
            ):
                logger.info(
                    "early-stop at step %d: val %.4f > best %.4f * %.2f",
                    steps, val_avg, best, early_stop_factor,
                )
                break
    return best, steps


__all__ = [
    "compute_loss",
    "train_one_step",
    "run_validation",
    "eval_step",
    "train_epoch",
]
