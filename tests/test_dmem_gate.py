"""Tests for the D-MEM-style two-axis salience gate.

All tests use a stub LLM client + stub embedder so no network calls
are made. The LLM-judge prompt schema and cache LRU semantics are
verified against the stubs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from myelin import MemoryClient
from myelin._internal.dmem_gate import (
    DMemDecision,
    DMemGate,
    _parse_judge_response,
)


# -----------------------------------------------------------------
# Stub embedder + LLM client
# -----------------------------------------------------------------


class StubEmbedder:
    """Returns deterministic embeddings keyed off text length + first char."""

    def encode(self, text: str) -> list[float]:
        # Cheap 4-d embedding that varies by content so cosine isn't trivial.
        if not text:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            float(len(text)) / 100.0,
            float(ord(text[0])) / 256.0,
            float(text.count(" ")) / 10.0,
            1.0 if "love" in text.lower() else 0.0,
        ]


@dataclass
class _StubChoice:
    message: Any


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubCompletion:
    choices: list


class StubLLM:
    """Mimics the OpenAI client's ``.chat.completions.create`` shape."""

    def __init__(self, scripted: list[dict] | None = None) -> None:
        # scripted: list of dicts that will be JSON-encoded as the
        # judge response, in order. Cycles if exhausted.
        self.scripted = scripted or [
            {"class": "persistent", "score": 0.9, "reason": "stub-persist"},
        ]
        self.call_count = 0
        self.chat = self  # so .chat.completions.create resolves
        self.completions = self

    def create(self, **kwargs) -> Any:
        idx = self.call_count % len(self.scripted)
        self.call_count += 1
        import json
        payload = json.dumps(self.scripted[idx])
        return _StubCompletion(
            choices=[_StubChoice(message=_StubMessage(content=payload))]
        )


# -----------------------------------------------------------------
# Parser tests
# -----------------------------------------------------------------


def test_parse_well_formed_json():
    cls, score, reason = _parse_judge_response(
        '{"class": "persistent", "score": 0.8, "reason": "user fact"}'
    )
    assert cls == "persistent"
    assert score == 0.8
    assert reason == "user fact"


def test_parse_embedded_json():
    raw = 'Sure: {"class": "transient", "score": 0.1, "reason": "filler"}'
    cls, score, reason = _parse_judge_response(raw)
    assert cls == "transient"
    assert score == 0.1


def test_parse_garbage_falls_back_to_transient():
    cls, score, _ = _parse_judge_response("absolute garbage no json here")
    assert cls == "transient"
    assert score == 0.0


def test_parse_unknown_class_falls_back_to_transient():
    cls, score, _ = _parse_judge_response('{"class": "bogus", "score": 0.5}')
    assert cls == "transient"


def test_parse_score_clamped_to_unit_interval():
    cls, score, _ = _parse_judge_response('{"class": "persistent", "score": 5.0}')
    assert 0.0 <= score <= 1.0


# -----------------------------------------------------------------
# DMemGate.evaluate behavior
# -----------------------------------------------------------------


def test_evaluate_transient_drops_regardless_of_surprise():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "transient", "score": 0.9, "reason": "chitchat"}
            ]
        ),
        embedder=StubEmbedder(),
    )
    d = gate.evaluate(content="hello there friend", user_id="u1")
    assert d.tag is False  # transient always drops
    assert d.utility_class == "transient"


def test_evaluate_persistent_high_utility_keeps():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "persistent", "score": 0.95, "reason": "biographical"}
            ]
        ),
        embedder=StubEmbedder(),
        rpe_threshold=0.20,  # lower so surprise≈0.5 × 0.95 = 0.475 passes
    )
    d = gate.evaluate(
        content="My daughter Melanie's birthday is August thirteenth.",
        user_id="u1",
    )
    assert d.tag is True
    assert d.utility_class == "persistent"
    assert d.utility_score >= 0.9


def test_evaluate_caches_by_content_hash():
    stub = StubLLM(
        scripted=[
            {"class": "persistent", "score": 0.7, "reason": "fact"},
        ]
    )
    gate = DMemGate(llm_client=stub, embedder=StubEmbedder())
    same_text = "I prefer green tea over coffee."
    gate.evaluate(content=same_text, user_id="u1")
    gate.evaluate(content=same_text, user_id="u1")
    # 1 LLM call, 1 cache hit.
    assert stub.call_count == 1
    assert gate.stats["llm_calls_made"] == 1
    assert gate.stats["llm_cache_hits"] == 1
    assert gate.stats["cache_hit_rate"] == 0.5


def test_evaluate_cache_is_per_content_not_per_user():
    """Same string from different users still hits the same cache row.

    Rationale: the utility judge is about the content, not the user.
    """
    stub = StubLLM(
        scripted=[
            {"class": "short_term", "score": 0.3, "reason": "ok"},
        ]
    )
    gate = DMemGate(llm_client=stub, embedder=StubEmbedder())
    gate.evaluate(content="bored today", user_id="u1")
    gate.evaluate(content="bored today", user_id="u2")
    assert stub.call_count == 1
    assert gate.stats["cache_hit_rate"] == 0.5


def test_evaluate_short_term_with_low_utility_drops():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "short_term", "score": 0.20, "reason": "topical"}
            ]
        ),
        embedder=StubEmbedder(),
        utility_threshold=0.30,  # default
        rpe_threshold=0.30,
    )
    d = gate.evaluate(content="I'm tired tonight", user_id="u1")
    assert d.tag is False  # utility below threshold


def test_evaluate_llm_failure_uses_soft_fallback_not_transient():
    """Phase G2 fix (2026-05-20): on LLM judge exception we now fall
    back to a *soft* default (short_term / 0.5) instead of transient/0.0.
    Rationale: the original (transient, 0.0) fallback caused a single
    rate-limited turn to silently filter the entire bench. Soft-fallback
    keeps the bench progressing with a sane class so threshold logic
    still applies.

    The fallback class is configurable via DMemGate(llm_error_fallback=).
    """
    class FailingLLM:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **k):
            raise RuntimeError("simulated network failure")

    gate = DMemGate(llm_client=FailingLLM(), embedder=StubEmbedder())
    d = gate.evaluate(content="some content", user_id="u1")
    # Soft fallback: short_term / 0.5 by default. Tag still drops here
    # because surprise * 0.5 < default rpe_threshold=0.30 with the
    # cold-start surprise≈0.5 → rpe ≈ 0.25 < 0.30.
    assert d.utility_class == "short_term"
    assert d.utility_score == pytest.approx(0.5)
    assert "judge-error-fallback" in d.reason
    assert "RuntimeError" in d.reason


def test_evaluate_llm_timeout_fallback_keeps_bench_alive():
    """Mocks httpx.ReadTimeout (the exact exception class the local
    Phase G Run 3 stall would have produced if the OpenAI client had
    actually surfaced the per-request timeout). Asserts the gate
    returns the configurable fallback decision instead of crashing or
    blocking, and that the LRU cache does NOT cache the failure.
    """
    try:
        import httpx
        timeout_exc = httpx.ReadTimeout("simulated read timeout")
    except ImportError:  # pragma: no cover - httpx ships with openai
        timeout_exc = TimeoutError("simulated read timeout")

    class TimingOutLLM:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.call_count = 0

        def create(self, **k):
            self.call_count += 1
            raise timeout_exc

    stub = TimingOutLLM()
    gate = DMemGate(
        llm_client=stub,
        embedder=StubEmbedder(),
        llm_error_fallback=("transient", 0.1, "test-fallback"),
    )
    d = gate.evaluate(content="some content", user_id="u1")
    # The configured fallback is used verbatim, not a hardcoded class.
    assert d.utility_class == "transient"
    assert d.utility_score == pytest.approx(0.1)
    assert "test-fallback" in d.reason
    # Failed calls don't populate the cache, so a retry on the same
    # content would call the LLM again (intentional — retry-when-back).
    assert len(gate._utility_cache) == 0


def test_lazy_openai_client_respects_timeout_and_retry_env(monkeypatch):
    """Verify the lazy OpenAI client is constructed with bounded timeout
    + max_retries (the actual fix that prevents the local Phase G Run 3
    8h-stall failure mode at the SDK level). We don't actually need the
    SDK to be importable in CI — we patch the import shim and assert the
    kwargs the gate would have passed.
    """
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = self
            self.completions = self

        def create(self, **k):  # pragma: no cover - never called
            raise RuntimeError("should not be called by this test")

    import sys
    import types as _types
    fake_mod = _types.ModuleType("openai")
    fake_mod.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MYELIN_DMEM_LLM_TIMEOUT", "12.5")
    monkeypatch.setenv("MYELIN_DMEM_LLM_MAX_RETRIES", "1")

    client = DMemGate._lazy_openai_client()
    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "sk-test"
    assert captured["timeout"] == pytest.approx(12.5)
    assert captured["max_retries"] == 1


def test_hyperparameter_env_pickup_used_when_constructor_omits(monkeypatch):
    """Phase G3 ablation sweep — three hyperparameters now env-var-driven
    so a single Docker image can run the whole sweep without rebuild.

    When the constructor omits these args (sentinel default), the gate
    must read MYELIN_DMEM_RPE_THRESHOLD / MYELIN_DMEM_UTILITY_THRESHOLD
    / MYELIN_DMEM_NEIGHBOR_WINDOW at __post_init__ time.
    """
    monkeypatch.setenv("MYELIN_DMEM_RPE_THRESHOLD", "0.20")
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_THRESHOLD", "0.40")
    monkeypatch.setenv("MYELIN_DMEM_NEIGHBOR_WINDOW", "128")
    gate = DMemGate(
        llm_client=StubLLM(),
        embedder=StubEmbedder(),
    )
    assert gate.rpe_threshold == pytest.approx(0.20)
    assert gate.utility_threshold == pytest.approx(0.40)
    assert gate.neighbor_window_size == 128


def test_explicit_constructor_args_win_over_env(monkeypatch):
    """Explicit kwargs override env. Existing unit tests rely on this
    (they pass e.g. ``rpe_threshold=0.20`` to force a kept-turn) and
    must keep working even if Phase G3 envs are set globally in CI.
    """
    monkeypatch.setenv("MYELIN_DMEM_RPE_THRESHOLD", "0.95")
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_THRESHOLD", "0.99")
    monkeypatch.setenv("MYELIN_DMEM_NEIGHBOR_WINDOW", "256")
    gate = DMemGate(
        llm_client=StubLLM(),
        embedder=StubEmbedder(),
        rpe_threshold=0.10,
        utility_threshold=0.15,
        neighbor_window_size=8,
    )
    assert gate.rpe_threshold == pytest.approx(0.10)
    assert gate.utility_threshold == pytest.approx(0.15)
    assert gate.neighbor_window_size == 8


def test_hyperparameter_env_unparseable_falls_back_to_default(monkeypatch):
    """Garbage env values must not crash construction — fall back to the
    baked-in 0.30 / 0.30 / 64 defaults with a logger.warning."""
    monkeypatch.setenv("MYELIN_DMEM_RPE_THRESHOLD", "not-a-float")
    monkeypatch.setenv("MYELIN_DMEM_UTILITY_THRESHOLD", "")  # empty
    monkeypatch.setenv("MYELIN_DMEM_NEIGHBOR_WINDOW", "9999")  # out of range
    gate = DMemGate(
        llm_client=StubLLM(),
        embedder=StubEmbedder(),
    )
    assert gate.rpe_threshold == pytest.approx(0.30)
    assert gate.utility_threshold == pytest.approx(0.30)
    assert gate.neighbor_window_size == 64


def test_hyperparameter_env_unset_uses_baked_defaults(monkeypatch):
    """No env set → reproduce the Phase G2 baseline defaults exactly."""
    monkeypatch.delenv("MYELIN_DMEM_RPE_THRESHOLD", raising=False)
    monkeypatch.delenv("MYELIN_DMEM_UTILITY_THRESHOLD", raising=False)
    monkeypatch.delenv("MYELIN_DMEM_NEIGHBOR_WINDOW", raising=False)
    gate = DMemGate(
        llm_client=StubLLM(),
        embedder=StubEmbedder(),
    )
    assert gate.rpe_threshold == pytest.approx(0.30)
    assert gate.utility_threshold == pytest.approx(0.30)
    assert gate.neighbor_window_size == 64


def test_neighbor_window_resets_per_user():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "persistent", "score": 0.7, "reason": "fact"}
            ]
        ),
        embedder=StubEmbedder(),
    )
    for i in range(5):
        gate.evaluate(content=f"turn-{i}", user_id="u1")
    assert len(gate._neighbor_windows["u1"]) == 5
    gate.reset_user("u1")
    assert "u1" not in gate._neighbor_windows


def test_neighbor_window_capped_at_size():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "persistent", "score": 0.7, "reason": "fact"}
            ]
        ),
        embedder=StubEmbedder(),
        neighbor_window_size=3,
    )
    for i in range(10):
        gate.evaluate(content=f"unique-{i}", user_id="u1")
    assert len(gate._neighbor_windows["u1"]) == 3  # capped


# -----------------------------------------------------------------
# MemoryClient end-to-end with dmem_gate injection
# -----------------------------------------------------------------


def test_client_dmem_mode_with_injected_gate_persists_kept_turns():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "persistent", "score": 0.9, "reason": "biographical"}
            ]
        ),
        embedder=StubEmbedder(),
        rpe_threshold=0.20,
    )
    c = MemoryClient(salience_mode="dmem", dmem_gate=gate)
    try:
        mem = c.add("My daughter Melanie's birthday is August thirteenth.")
        assert mem.id  # was kept
        hits = c.search("daughter")
        assert hits
    finally:
        c.close()


def test_client_dmem_mode_drops_transient_chitchat():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "transient", "score": 0.05, "reason": "filler"}
            ]
        ),
        embedder=StubEmbedder(),
    )
    c = MemoryClient(salience_mode="dmem", dmem_gate=gate)
    try:
        for _ in range(5):
            c.add("hello hi yes ok")
        tel = c.salience_telemetry
        assert tel["n_add_dropped"] == 5
        assert tel["tag_rate"] == 0.0
        assert tel["dmem"]["cache_hit_rate"] == pytest.approx(0.8)
    finally:
        c.close()


def test_client_dmem_telemetry_shape():
    gate = DMemGate(
        llm_client=StubLLM(
            scripted=[
                {"class": "persistent", "score": 0.9, "reason": "fact"},
                {"class": "transient", "score": 0.1, "reason": "fluff"},
            ]
        ),
        embedder=StubEmbedder(),
        rpe_threshold=0.20,
    )
    c = MemoryClient(salience_mode="dmem", dmem_gate=gate)
    try:
        c.add("Important biographical fact one")
        c.add("hello fluff")
        tel = c.salience_telemetry
        assert tel["salience_mode"] == "dmem"
        assert tel["n_add_calls"] == 2
        assert tel["n_add_kept"] == 1
        assert tel["n_add_dropped"] == 1
        assert "dmem" in tel
        assert tel["dmem"]["llm_calls_made"] == 2
    finally:
        c.close()
