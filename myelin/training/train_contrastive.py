"""Main training entrypoint for the Memory-R1 contrastive utility
encoder.

Usage::

    python -m myelin.training.train_contrastive \\
        --data gs://bucket/personamem-dmem-g2-*/results.jsonl \\
        --out gs://bucket/myelin-mr1/v1-2026-05-21/

The training image (cortex-train-mr1:v1-<commit>) calls
``main()`` with the materialized local paths after a ``gsutil cp``
prefetch.

Outputs (written to ``out_dir``):
- ``model.safetensors`` — fp16 trained weights
- ``model.onnx`` — exported ONNX session for CPU inference (optional)
- ``tokenizer.json`` + ``tokenizer_config.json`` — fast Rust BPE
- ``metadata.json`` — {weights_sha256, train/val/test losses,
  test_class_accuracy, env hyperparameters, base_model_id}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *,
    data_paths: list[str],
    out_dir: str,
    seed: int = 42,
) -> int:
    """Run the training pipeline and write artifacts.

    Returns POSIX exit code: 0 = success + AC-3 pass; 1 = failure or
    AC-3 fail (caller must read ``metadata.json::test_class_accuracy``
    to distinguish).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(seed)
    try:
        import torch
        import numpy as np  # noqa: F401
    except ImportError as e:
        logger.error("training requires torch + numpy: %s", e)
        return 1
    torch.manual_seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from myelin.training._config import TrainConfig
    from myelin.training.data_pipeline import DataPipeline, InsufficientDataError
    from myelin.training._pipeline_internal import (
        HardNegativeMiner,
        persona_stratified_split,
    )
    from myelin.training.eval_held_out import evaluate_test_set
    from myelin.training.model_factory import build_joint_model
    from myelin.training.train_loop import train_epoch

    cfg = TrainConfig.from_env()
    logger.info("TrainConfig: %s", cfg)

    # 1. Load + hard-neg + split.
    try:
        pairs = DataPipeline().build_pairs(data_paths)
    except InsufficientDataError as e:
        logger.error("InsufficientDataError: %s", e)
        return 1

    # Cheap embed_fn for hard-neg mining (smoke uses bag-of-bytes hash).
    miner = HardNegativeMiner(
        embed_fn=lambda texts: [_cheap_embed(t) for t in texts],
        k=5,
        seed=seed,
    )
    positives = [p for p in pairs if p.label > 0.5]
    pairs = miner.augment(positives, all_corpus_turns=[p.turn for p in pairs])

    train, val, test = persona_stratified_split(
        pairs,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        seed=seed,
    )
    logger.info(
        "splits: train=%d val=%d test=%d", len(train), len(val), len(test)
    )

    # 2. Build model.
    model = build_joint_model(
        use_pretrained=cfg.use_pretrained, base_model_id=cfg.base_model_id
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate_head,
        weight_decay=cfg.weight_decay,
    )

    # 3. Train.
    tokenizer = _load_tokenizer_or_none(cfg)
    train_batches = list(_pairs_to_batches(train, cfg, tokenizer))
    val_batches = list(_pairs_to_batches(val, cfg, tokenizer))

    best_val = float("inf")
    total_steps = 0
    for epoch in range(cfg.num_epochs):
        if epoch >= cfg.unfreeze_after_epoch and cfg.use_pretrained:
            model.unfreeze_backbone()
            for g in optimizer.param_groups:
                g["lr"] = cfg.learning_rate_full
        elif epoch == 0 and cfg.use_pretrained:
            model.freeze_backbone()
        logger.info("epoch %d/%d", epoch + 1, cfg.num_epochs)
        best_val, steps = train_epoch(
            model, optimizer,
            train_batches, val_batches,
            temperature=cfg.infonce_temperature,
            w_util=cfg.loss_weight_utility,
            w_class=cfg.loss_weight_class,
            w_nce=cfg.loss_weight_nce,
            max_steps=cfg.max_steps,
            early_stop_factor=cfg.early_stop_factor,
            early_stop_min_step=cfg.early_stop_min_step,
            best_val_so_far=best_val,
        )
        total_steps += steps
        if cfg.max_steps and total_steps >= cfg.max_steps:
            break

    # 4. Eval.
    test_metrics = evaluate_test_set(
        model, test,
        batch_size=cfg.batch_size,
        tokenizer=tokenizer,
        max_length=cfg.tokenizer_max_length,
    )
    logger.info("test_metrics: %s", test_metrics)

    # 5. Save artifacts.
    weights_path = out_path / "model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(_state_dict_fp16(model), str(weights_path))
    except Exception as e:
        logger.warning(
            "safetensors save failed (%s); writing torch fallback", e
        )
        torch.save(model.state_dict(), str(weights_path))
    sha = _sha256_file(weights_path)

    metadata = {
        "version": "contrastive_v1",
        "weights_sha256": sha,
        "best_val_loss": best_val,
        "test_class_accuracy": test_metrics["test_class_accuracy"],
        "test_loss": test_metrics["test_loss"],
        "test_n": test_metrics["test_n"],
        "config": {
            "base_model_id": cfg.base_model_id,
            "infonce_temperature": cfg.infonce_temperature,
            "loss_weights": [
                cfg.loss_weight_utility,
                cfg.loss_weight_class,
                cfg.loss_weight_nce,
            ],
            "learning_rate_head": cfg.learning_rate_head,
            "learning_rate_full": cfg.learning_rate_full,
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "seed": seed,
        },
        "train_n": len(train),
        "val_n": len(val),
        "test_n_actual": len(test),
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote %s + metadata.json (SHA256=%s)", weights_path, sha)

    # AC-3 gate (only when not in smoke mode).
    if not cfg.max_steps:
        target = cfg.target_test_class_accuracy
        if test_metrics["test_class_accuracy"] < target:
            logger.error(
                "AC-3 FAIL: test_class_accuracy=%.3f < target %.3f",
                test_metrics["test_class_accuracy"], target,
            )
            return 1
    return 0


def _cheap_embed(text: str, *, dim: int = 16) -> list[float]:
    """Deterministic toy embedding for hard-neg mining when no real
    embedder is available. Bag-of-bytes hash mapped to dim coords."""
    if not text:
        return [0.0] * dim
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 - 0.5 for b in h[:dim]]


def _load_tokenizer_or_none(cfg):
    if not cfg.use_pretrained:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        return AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)
    except Exception as e:
        logger.warning(
            "tokenizer load failed for %s (%s); using dummy ids",
            cfg.base_model_id, e,
        )
        return None


def _pairs_to_batches(pairs, cfg, tokenizer):
    from myelin.training.eval_held_out import _to_batches
    yield from _to_batches(
        pairs, cfg.batch_size, tokenizer, cfg.tokenizer_max_length
    )


def _state_dict_fp16(model):
    import torch
    return {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Memory-R1 contrastive training")
    parser.add_argument(
        "--data", action="append", required=True,
        help="path to results.jsonl (repeatable)",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return main(data_paths=args.data, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    sys.exit(_cli())


__all__ = ["main"]
