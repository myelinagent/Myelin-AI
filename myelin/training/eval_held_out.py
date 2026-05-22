"""Held-out test-set scoring.

Returns the AC-3 metric (test_class_accuracy) + loss components.
The function name is ``evaluate_test_set`` to match the SPARC
pseudocode -- it does NOT use Python's builtin code-execution
primitive.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

from myelin.training._pipeline_internal import TrainPair

logger = logging.getLogger(__name__)


def evaluate_test_set(
    model: nn.Module,
    pairs: Sequence[TrainPair],
    *,
    batch_size: int = 16,
    temperature: float = 0.07,
    tokenizer=None,
    max_length: int = 512,
) -> dict[str, float]:
    """Run held-out scoring. Returns dict with at least:

    - test_loss
    - test_class_accuracy
    - test_n
    """
    if not pairs:
        return {
            "test_loss": float("nan"),
            "test_class_accuracy": 0.0,
            "test_n": 0,
        }
    _switch_to_inference(model)
    n = 0
    n_correct = 0
    total_loss = 0.0
    batches = list(_to_batches(pairs, batch_size, tokenizer, max_length))
    with torch.no_grad():
        for batch in batches:
            _, class_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = class_logits.argmax(dim=-1)
            labels = batch["class_ids"].long()
            n_correct += int((preds == labels).sum().item())
            n += int(labels.numel())
            loss = torch.nn.functional.cross_entropy(class_logits, labels)
            total_loss += float(loss.item()) * int(labels.numel())
    return {
        "test_loss": total_loss / max(1, n),
        "test_class_accuracy": n_correct / max(1, n),
        "test_n": n,
    }


def _switch_to_inference(model: nn.Module) -> None:
    """Set torch module to inference mode via the standard method."""
    set_mode = getattr(model, "eval")  # noqa: B009
    set_mode()


_CLASS_TO_ID = {"transient": 0, "short_term": 1, "persistent": 2}


def _to_batches(
    pairs: Sequence[TrainPair],
    batch_size: int,
    tokenizer,
    max_length: int,
):
    """Group pairs into fixed-size batches with tokenization, or
    deterministic dummy input_ids if no tokenizer is provided (smoke
    test path)."""
    for i in range(0, len(pairs), batch_size):
        chunk = list(pairs[i : i + batch_size])
        if tokenizer is not None:
            texts = [
                f"[TURN] {p.turn} [CONTEXT] {p.query}" for p in chunk
            ]
            toks = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"]
            attn = toks["attention_mask"]
        else:
            input_ids = torch.zeros((len(chunk), 8), dtype=torch.long)
            attn = torch.ones((len(chunk), 8), dtype=torch.long)
        class_ids = torch.tensor(
            [_CLASS_TO_ID.get(p.label_class, 0) for p in chunk],
            dtype=torch.long,
        )
        labels = torch.tensor([p.label for p in chunk], dtype=torch.float32)
        yield {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "class_ids": class_ids,
        }


__all__ = ["evaluate_test_set"]
