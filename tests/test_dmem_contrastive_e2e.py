"""End-to-end tests for DMemGate + contrastive_v1 backend.

AC-4: Zero OpenAI API calls when MYELIN_DMEM_UTILITY_BACKEND=contrastive_v1.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from myelin._internal.dmem_gate import DMemGate


# ----- fixtures ---------------------------------------------------


class _FailingOpenAI:
    """If the contrastive backend leaks any OpenAI call, this client
    raises immediately so the test fails LOUD instead of silently
    spending API budget."""

    def __init__(self) -> None:
        self.chat = self
        self.completions = self

    def create(self, **_kw):
        raise AssertionError(
            "OpenAI client called while utility_backend=contrastive_v1"
        )


class _StubEmbedder:
    def encode(self, text: str) -> list[float]:
        return [float(len(text)) / 100.0, 0.5, 0.0, 0.0]


@pytest.fixture
def fake_contrastive_score(monkeypatch):
    """Replace ContrastiveEncoder.score with a deterministic stub so the
    e2e test doesn't require trained weights to exist on disk.
    """
    def _score(self, *, content, neighbor_context):  # noqa: ANN001
        # Deterministic: long content → persistent + high util.
        if len(content) > 30:
            return 0.9, "persistent"
        return 0.2, "transient"
    try:
        from myelin._internal.contrastive_encoder import ContrastiveEncoder
    except ImportError:
        pytest.skip("contrastive_encoder not yet implemented")
    monkeypatch.setattr(ContrastiveEncoder, "score", _score)
    monkeypatch.setattr(ContrastiveEncoder, "get_instance",
                        classmethod(lambda cls: cls.__new__(cls)))


# ----- tests ------------------------------------------------------


def test_contrastive_backend_makes_zero_openai_calls(
    fake_contrastive_score, monkeypatch
):
    """AC-4: backend=contrastive_v1 → no OpenAI traffic."""
    failing_client = _FailingOpenAI()
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_BACKEND", "contrastive_v1")
    gate = DMemGate(
        utility_backend="contrastive_v1",
        llm_client=failing_client,  # would raise if touched
        embedder=_StubEmbedder(),
    )
    # 5 turns of varied lengths — none should hit the OpenAI client.
    for content in (
        "hi",
        "My favorite hobby is sailing on weekend mornings near Marina Bay",
        "yes",
        "ok",
        "I prefer pour-over coffee from light roasts especially Ethiopian beans",
    ):
        d = gate.evaluate(content=content, user_id="u1")
        assert d is not None


def test_env_override_picks_contrastive_backend(
    fake_contrastive_score, monkeypatch
):
    """FR-5: MYELIN_DMEM_UTILITY_BACKEND=contrastive_v1 selects the new
    backend without explicit constructor arg."""
    failing_client = _FailingOpenAI()
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_BACKEND", "contrastive_v1")
    gate = DMemGate(llm_client=failing_client, embedder=_StubEmbedder())
    assert gate.utility_backend == "contrastive_v1"
    d = gate.evaluate(
        content="My daughter Melanie was born in August", user_id="u1"
    )
    assert d.utility_class in ("transient", "short_term", "persistent")


def test_explicit_constructor_arg_wins_over_env(
    fake_contrastive_score, monkeypatch
):
    """Constructor kwarg always wins; env is only consulted on sentinel."""
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_BACKEND", "contrastive_v1")
    # Caller explicitly asks for llm_judge — env must be ignored.
    gate = DMemGate(utility_backend="llm_judge", embedder=_StubEmbedder(),
                    llm_client=_FailingOpenAI())
    assert gate.utility_backend == "llm_judge"


def test_unknown_backend_value_raises(monkeypatch):
    """Defensive: garbage env value should not silently fall through."""
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_BACKEND", "definitely-not-a-backend")
    with pytest.raises((ValueError, RuntimeError)):
        DMemGate(embedder=_StubEmbedder(), llm_client=_FailingOpenAI())


def test_fallback_to_llm_judge_on_encoder_load_failure(
    monkeypatch
):
    """EC-4 (with graceful degradation per FR-4 lazy-load path):
    if ContrastiveEncoder fails to load, the gate should fall back to
    the LLM-judge path with a warning — NOT crash the bench.
    """
    try:
        from myelin._internal.contrastive_encoder import ContrastiveEncoder
        from myelin._internal.contrastive_encoder import (
            ContrastiveEncoderLoadError,
        )
    except ImportError:
        pytest.skip("contrastive_encoder not yet implemented")

    def _raise_load(cls):  # noqa: ANN001
        raise ContrastiveEncoderLoadError("simulated weight load failure")
    monkeypatch.setattr(ContrastiveEncoder, "get_instance",
                        classmethod(_raise_load))

    # The fallback path means the LLM judge IS called, so wire a stub
    # that returns a valid response instead of _FailingOpenAI.
    class _OkLLM:
        def __init__(self):
            self.calls = 0
            self.chat = self
            self.completions = self
        def create(self, **_kw):
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"class": "persistent", "score": 0.8, "reason": "fb"}'
                    )
                )]
            )
    ok = _OkLLM()
    gate = DMemGate(
        utility_backend="contrastive_v1",
        llm_client=ok,
        embedder=_StubEmbedder(),
    )
    d = gate.evaluate(content="some content", user_id="u1")
    assert d is not None
    # The fallback path WILL call the LLM — count > 0 proves graceful
    # degradation worked.
    assert ok.calls > 0


def test_distribution_shift_monitor_logs_warning(
    fake_contrastive_score, monkeypatch, caplog
):
    """EC-3: 100 consecutive transient turns → warning logged."""
    import logging
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("MYELIN_DMEM_AUTO_FALLBACK_THRESHOLD", "0.9")
    gate = DMemGate(
        utility_backend="contrastive_v1",
        llm_client=_FailingOpenAI(),
        embedder=_StubEmbedder(),
    )
    for _ in range(100):
        gate.evaluate(content="hi", user_id="u1")  # all → transient by stub
    # The gate should have logged a distribution-shift warning by now.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("transient" in m.lower() or "shift" in m.lower() for m in msgs)
