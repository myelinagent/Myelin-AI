"""Smoke test for the training entrypoint.

Per SPARC Phase 5 A.5: 1-step train + eval on a tiny n=100 fixture.
Smoke does NOT require GPU. The real training is the Vertex job.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is part of [training] extra")

try:
    from myelin.training import _config as _cfg_mod  # type: ignore[import-not-found]
    from myelin.training._config import TrainConfig  # type: ignore[import-not-found]
    from myelin.training.model_factory import (  # type: ignore[import-not-found]
        DeBERTaV3SmallJointHead,
        build_joint_model,
    )
    from myelin.training.train_loop import (  # type: ignore[import-not-found]
        train_one_step,
        compute_loss,
    )
    from myelin.training.eval_held_out import evaluate_test_set  # type: ignore[import-not-found]
    from myelin.training._pipeline_internal import TrainPair  # type: ignore[import-not-found]
    _HAS = True
except ImportError:
    _HAS = False


pytestmark = pytest.mark.skipif(
    not _HAS, reason="training module not yet implemented"
)


# ----- fixtures ---------------------------------------------------


def _make_smoke_pairs(n: int = 100) -> list:
    pairs = []
    for i in range(n):
        cls = ["transient", "short_term", "persistent"][i % 3]
        pairs.append(
            TrainPair(
                turn=f"content number {i} about topic-{i % 5}",
                query=f"q{i}",
                label=1.0 if i % 3 == 2 else 0.0,
                label_class=cls,
                persona_id=f"p{i % 4}",
                source="positive" if i % 2 == 0 else "hard-neg",
            )
        )
    return pairs


# ----- tests ------------------------------------------------------


def test_train_config_reads_env_defaults(monkeypatch):
    """All hyperparameters externalized to env (Phase 4 refinement)."""
    monkeypatch.setenv("MYELIN_MR1_INFONCE_TEMP", "0.123")
    monkeypatch.setenv("MYELIN_MR1_MAX_LEN", "256")
    cfg = TrainConfig.from_env()
    assert cfg.infonce_temperature == 0.123
    assert cfg.tokenizer_max_length == 256


def test_train_config_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("MYELIN_MR1_INFONCE_TEMP", raising=False)
    monkeypatch.delenv("MYELIN_MR1_MAX_LEN", raising=False)
    cfg = TrainConfig.from_env()
    assert cfg.infonce_temperature == 0.07
    assert cfg.tokenizer_max_length == 512


def test_model_factory_returns_joint_head_with_two_outputs():
    """The model returns (util_logit, class_logits) per Algorithm D."""
    model = build_joint_model(use_pretrained=False)
    # use_pretrained=False uses a tiny init for fast smoke.
    assert isinstance(model, torch.nn.Module)
    # Build a dummy forward.
    input_ids = torch.zeros((2, 8), dtype=torch.long)
    attn = torch.ones((2, 8), dtype=torch.long)
    util_logit, class_logits = model(input_ids=input_ids, attention_mask=attn)
    assert util_logit.shape[-1] == 1
    assert class_logits.shape[-1] == 3


def test_train_one_step_reduces_loss():
    """Smoke: one optimizer step lowers loss on a deterministic batch."""
    model = build_joint_model(use_pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.zeros((4, 8), dtype=torch.long),
        "attention_mask": torch.ones((4, 8), dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "class_ids": torch.tensor([2, 0, 2, 1]),
    }
    loss_before, _ = compute_loss(model, batch, temperature=0.07)
    for _ in range(5):
        train_one_step(model, optimizer, batch, temperature=0.07)
    loss_after, _ = compute_loss(model, batch, temperature=0.07)
    assert loss_after.item() <= loss_before.item() + 1e-3, \
        f"loss {loss_after.item():.4f} did not decrease from {loss_before.item():.4f}"


def test_eval_returns_test_class_accuracy():
    """Eval emits the AC-3 metric (class_accuracy)."""
    model = build_joint_model(use_pretrained=False)
    pairs = _make_smoke_pairs(20)
    metrics = evaluate_test_set(model, pairs, batch_size=4)
    assert "test_class_accuracy" in metrics
    assert 0.0 <= metrics["test_class_accuracy"] <= 1.0
    assert "test_loss" in metrics


def test_smoke_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    """Phase 5 A.5: n=100 smoke run completes without exception and
    writes a metadata.json next to a (possibly empty) safetensors."""
    pytest.importorskip("safetensors")
    from myelin.training import train_contrastive as tc  # type: ignore[import-not-found]

    # Stage 100-row fixture data on disk.
    data_path = tmp_path / "results.jsonl"
    with data_path.open("w") as f:
        for i in range(150):
            row = {
                "qid": f"q{i}",
                "persona_id": f"p{i % 5}",
                "system": "myelin",
                "axis": ["biographical", "preference"][i % 2],
                "response": f"r{i}",
                "gold_answer": f"r{i}" if i % 2 == 0 else "wrong",
                "retrieved_memories": [
                    {"id": f"m{i}", "content": f"turn-{i}", "query": f"q{i}"},
                ],
            }
            f.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("MYELIN_MR1_MAX_STEPS", "5")  # smoke caps to 5 steps
    monkeypatch.setenv("MYELIN_MR1_BATCH_SIZE", "4")
    monkeypatch.setenv("MYELIN_MR1_NUM_EPOCHS", "1")
    monkeypatch.setenv("MYELIN_MR1_USE_PRETRAINED", "0")  # tiny init

    rc = tc.main(
        data_paths=[str(data_path)],
        out_dir=str(out_dir),
        seed=42,
    )
    assert rc == 0
    assert (out_dir / "metadata.json").exists()
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert "test_class_accuracy" in meta
    assert "weights_sha256" in meta
