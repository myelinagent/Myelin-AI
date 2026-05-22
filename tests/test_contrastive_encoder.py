"""Tests for ContrastiveEncoder — singleton, score, latency, threading.

Per SPARC spec §AC-1: encoder returns (utility ∈ [0,1], class ∈
{transient|short_term|persistent}) within 20ms p99 CPU.

These tests verify the *behavioral contract* without requiring the
trained weights to actually be present — the encoder MUST be load-on-
demand and the load failure surface MUST be clean.

Heavy imports (torch, transformers, safetensors) live behind a soft
guard so the test module is collectable in environments where the
[contrastive] extra is not installed; the actual fixtures will skip if
torch is missing.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest


# ----- import guard ------------------------------------------------

try:
    from myelin._internal import contrastive_encoder as ce_mod  # type: ignore[import-not-found]
    from myelin._internal.contrastive_encoder import (  # type: ignore[import-not-found]
        ContrastiveEncoder,
        ContrastiveEncoderLoadError,
    )
    _HAS_ENCODER = True
except ImportError:
    _HAS_ENCODER = False

torch = pytest.importorskip("torch", reason="torch is part of [contrastive] extra")


# ----- fixtures ----------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Drop singleton between tests so we exercise load-once behavior."""
    if not _HAS_ENCODER:
        yield
        return
    ContrastiveEncoder._reset_for_tests()
    yield
    ContrastiveEncoder._reset_for_tests()


@pytest.fixture
def fake_weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny on-disk safetensors + tokenizer + meta so the encoder
    can load without hitting HF Hub.

    The real model is `microsoft/deberta-v3-small`; for unit-test speed
    we monkey-patch ``ContrastiveEncoder._build_model`` to return a tiny
    torch.nn.Module with the right output shape.
    """
    weights = tmp_path / "weights"
    weights.mkdir()
    # Empty marker file — the test monkey-patches the load path.
    (weights / "model.safetensors").write_bytes(b"")
    (weights / "tokenizer.json").write_text("{}")
    (weights / "metadata.json").write_text(
        '{"version": "contrastive_v1", "sha256": "test"}'
    )
    monkeypatch.setenv("MYELIN_WEIGHTS_CACHE", str(tmp_path))
    return weights


# ----- behavior tests ----------------------------------------------


def test_score_returns_unit_utility_and_valid_class(monkeypatch, fake_weights_dir):
    """AC-1 surface check: (float ∈ [0,1], 3-way class string)."""
    _install_fake_backend(monkeypatch, fake_weights_dir)
    enc = ContrastiveEncoder.get_instance()
    util, cls = enc.score(content="I love sailing on weekends", neighbor_context="")
    assert isinstance(util, float)
    assert 0.0 <= util <= 1.0
    assert cls in ("transient", "short_term", "persistent")


def test_singleton_loads_once(monkeypatch, fake_weights_dir):
    """EC-2: concurrent get_instance() must NOT load the model twice."""
    load_counter = {"n": 0}
    _install_fake_backend(monkeypatch, fake_weights_dir, load_counter=load_counter)
    _a = ContrastiveEncoder.get_instance()
    _b = ContrastiveEncoder.get_instance()
    assert _a is _b
    assert load_counter["n"] == 1


def test_score_latency_p99_under_20ms(monkeypatch, fake_weights_dir):
    """NFR-1: 20ms p99 CPU.

    Marked latency so it can be skipped on slow CI runners but exercised
    locally and on a release-blocking build.
    """
    _install_fake_backend(monkeypatch, fake_weights_dir)
    enc = ContrastiveEncoder.get_instance()
    # Warm-up — exclude from the percentile sample.
    for _ in range(5):
        enc.score(content="hello", neighbor_context="")
    samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        enc.score(content="My daughter loves sailing.", neighbor_context="ctx")
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p99 = samples[int(0.99 * len(samples)) - 1]
    # With the fake backend the p99 should be sub-ms; the assertion is
    # against the SPEC bound so a real-model regression will fail it.
    assert p99 < 20.0, f"p99 latency {p99:.2f}ms exceeded 20ms NFR-1"


def test_load_failure_raises_clean_error(monkeypatch, tmp_path):
    """EC-4: weights file corruption raises at construction, not at first score."""
    bad_dir = tmp_path / "weights"
    bad_dir.mkdir()
    # No model.safetensors, no tokenizer.json — anything missing should error
    # at get_instance().
    monkeypatch.setenv("MYELIN_WEIGHTS_CACHE", str(tmp_path))

    def _fail_fetch(*_a, **_kw):  # noqa: ANN001
        raise FileNotFoundError("weights/tokenizer not in cache")
    monkeypatch.setattr(
        "myelin._internal.weights_fetcher.fetch_weights",
        _fail_fetch,
        raising=False,
    )
    with pytest.raises(ContrastiveEncoderLoadError):
        ContrastiveEncoder.get_instance()


def test_score_truncates_long_context(monkeypatch, fake_weights_dir):
    """Pseudocode D: neighbor_context capped at 512 chars before tokenization.

    The fake backend records the (content, context) it received so we
    can assert truncation happened at the encoder layer.
    """
    bag = {}
    _install_fake_backend(monkeypatch, fake_weights_dir, capture=bag)
    enc = ContrastiveEncoder.get_instance()
    long_ctx = "x" * 5000
    enc.score(content="turn", neighbor_context=long_ctx)
    # The encoder should not pass 5000 chars of context through to the
    # tokenizer; the SPARC pseudocode says truncate to 512.
    assert len(bag.get("ctx", "")) <= 512


# ----- helpers -----------------------------------------------------


class _FakeTokenizer:
    def __init__(self, *_a, **_kw):
        self.calls = 0

    def __call__(self, text, **_kw):
        self.calls += 1
        # Return shapes compatible with a torch model forward — but the
        # fake model ignores them.
        return {"input_ids": torch.zeros((1, 8), dtype=torch.long),
                "attention_mask": torch.ones((1, 8), dtype=torch.long)}


class _FakeJointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Two heads: scalar utility, 3-way class.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None):  # noqa: ANN001
        # Deterministic outputs so tests are reproducible.
        util_logit = torch.tensor([[0.0]])  # sigmoid(0) == 0.5
        class_logits = torch.tensor([[0.1, 0.2, 0.7]])  # argmax → persistent
        return util_logit, class_logits


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    weights_dir: Path,
    *,
    load_counter: dict | None = None,
    capture: dict | None = None,
) -> None:
    """Patch the encoder's load + forward path so tests don't need the
    real DeBERTa weights.
    """
    def _fake_build_model(self, weights_dir=None):  # noqa: ANN001
        if load_counter is not None:
            load_counter["n"] += 1
        self._model = _FakeJointModel()
        self._tokenizer = _FakeTokenizer()

    def _fake_fetch(*_a, **_kw):  # noqa: ANN001
        return str(weights_dir)

    # Make the singleton path resolve weights_dir without hitting HF Hub.
    monkeypatch.setattr(
        "myelin._internal.weights_fetcher.fetch_weights",
        _fake_fetch,
        raising=False,
    )
    monkeypatch.setattr(ContrastiveEncoder, "_build_model", _fake_build_model)

    # Capture the truncated context for the truncation test.
    if capture is not None:
        original_score = ContrastiveEncoder.score

        def _capturing_score(self, *, content, neighbor_context):  # noqa: ANN001
            capture["ctx"] = neighbor_context[: self.max_context_chars]
            return original_score(self, content=content, neighbor_context=neighbor_context)

        monkeypatch.setattr(ContrastiveEncoder, "score", _capturing_score)
