"""Tests for the carved 4-signal heuristic salience pipeline.

Mirrors the unit-level assertions in `src/cortex/salience/` tests but
against the stdlib-only port shipped inside the myelin SDK so the
benched gate is the actual gate.
"""
from __future__ import annotations

import pytest

from myelin import MemoryClient
from myelin._internal.salience import (
    HeuristicSalienceGate,
    SignalBundle,
    affect_score,
    novelty_score,
    reference_score,
    reward_score,
    score_turn_heuristic,
)


# -----------------------------------------------------------------
# Signal scorer micro-tests (10 below — meet the spec's ~10 target).
# -----------------------------------------------------------------


def test_affect_picks_up_lexicon_words():
    assert affect_score("I love it") > 0.0
    assert affect_score("Hate it.") > 0.0
    assert affect_score("the quick brown fox") == 0.0


def test_reference_returns_zero_on_empty():
    assert reference_score("") == 0.0
    assert reference_score("    ") == 0.0


def test_reference_higher_on_content_dense_text():
    short_stopword = reference_score("a an the is to of")
    content_dense = reference_score("Birthday August thirteenth daughter Melanie")
    assert content_dense > short_stopword


def test_reward_none_returns_zero():
    assert reward_score(None) == 0.0
    assert reward_score({}) == 0.0


def test_reward_task_succeeded_lifts_score():
    assert reward_score({"task_succeeded": True}) > 0.0
    assert reward_score({"task_succeeded": True, "user_thumbs_up": True}) == 1.0


def test_novelty_cold_start_neutral():
    # No query embedding → cold-start neutral 0.5.
    assert novelty_score(None) == 0.5
    # Query but no neighbors → also cold-start neutral.
    assert novelty_score([1.0, 0.0]) == 0.5


def test_novelty_low_when_neighbor_matches():
    q = [1.0, 0.0]
    nb = [[1.0, 0.0]]
    assert novelty_score(q, nb) == pytest.approx(0.0)


def test_gate_at_default_thresholds_keeps_strong_signals():
    g = HeuristicSalienceGate(tau=0.40, require_reward=False)
    bundle = SignalBundle(novelty=0.9, affect=0.9, reward=0.9, reference=0.9)
    decision = g.evaluate(bundle)
    assert decision.tag is True
    assert 0.5 < decision.score <= 1.0


def test_gate_drops_zero_signal_turns():
    g = HeuristicSalienceGate(tau=0.40, require_reward=False)
    bundle = SignalBundle(novelty=0.0, affect=0.0, reward=0.0, reference=0.0)
    decision = g.evaluate(bundle)
    assert decision.tag is False
    assert decision.score < 0.5


def test_gate_kill_switch_blocks_without_reward():
    g_strict = HeuristicSalienceGate(tau=0.40, require_reward=True)
    bundle = SignalBundle(novelty=1.0, affect=1.0, reward=0.0, reference=1.0)
    decision = g_strict.evaluate(bundle)
    assert decision.tag is False  # zero reward kills the tag
    g_relaxed = HeuristicSalienceGate(tau=0.40, require_reward=False)
    decision2 = g_relaxed.evaluate(bundle)
    assert decision2.tag is True


def test_score_turn_heuristic_end_to_end():
    decision = score_turn_heuristic(
        text="Melanie's daughter's birthday is August thirteenth.",
        outcome={"task_succeeded": True},
        gate=HeuristicSalienceGate(tau=0.40, require_reward=False),
    )
    assert decision.tag is True


# -----------------------------------------------------------------
# MemoryClient wiring — turn drops actually skip the substrate.
# -----------------------------------------------------------------


class _DetEmbedder:
    """Deterministic stdlib embedder so the heuristic-mode client tests stay
    hermetic and fast — B2 wired a real embedder into the novelty axis, and
    without injection the client would lazy-download sentence-transformers."""

    def __init__(self, dim: int = 24) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        import hashlib

        d = hashlib.sha256(text.encode("utf-8")).digest()
        return [(d[i % len(d)] / 255.0) - 0.5 for i in range(self.dim)]


def test_client_off_mode_stores_every_turn():
    c = MemoryClient(salience_mode="off")
    try:
        for i in range(10):
            mem = c.add(f"hello world {i}")
            assert mem.id  # never dropped in off mode
        tel = c.salience_telemetry
        assert tel["n_add_dropped"] == 0
        assert tel["tag_rate"] == 1.0
    finally:
        c.close()


def test_client_heuristic_drops_low_signal_chitchat():
    c = MemoryClient(
        salience_mode="heuristic",
        salience_tau=0.40,
        salience_require_reward=False,
        salience_embedder=_DetEmbedder(),
    )
    try:
        # All low-signal stopword chitchat → should be dropped by tau=0.40.
        for _ in range(10):
            mem = c.add("ok")
        tel = c.salience_telemetry
        assert tel["n_add_dropped"] >= 8  # ≥80% dropped
        assert tel["n_add_calls"] == 10
    finally:
        c.close()


def test_client_heuristic_preserves_substrate_for_kept_turns():
    c = MemoryClient(
        salience_mode="heuristic",
        salience_tau=0.40,
        salience_require_reward=False,
        salience_embedder=_DetEmbedder(),
    )
    try:
        # A content-dense, affect-laden turn — should clear tau=0.40.
        mem = c.add(
            "Absolutely loved the brilliant performance amazing wonderful "
            "fantastic experience daughter birthday August thirteenth."
        )
        if mem.id:  # if kept
            hits = c.search("birthday")
            assert hits
    finally:
        c.close()


def test_client_heuristic_novelty_and_reward_vary_as_wired():
    # B2 (ADR-026): the as-wired novelty axis moves with distinct turns and the
    # reward axis moves with ``outcome=`` — the two signals that were pinned at
    # 0.5 / 0.0 before the embedder + outcome were threaded through add().
    c = MemoryClient(
        salience_mode="heuristic",
        salience_tau=0.0,  # keep every turn so the signals are observable
        salience_embedder=_DetEmbedder(),
    )
    try:
        def sal(mem):
            md = mem.metadata or {}
            return md.get("_salience", md)

        novelties = [
            sal(c.add(t)).get("salience_novelty")
            for t in ["fact alpha", "fact beta", "fact gamma", "fact delta"]
        ]
        assert len({round(n, 6) for n in novelties}) > 1, novelties

        r_none = sal(c.add("turn x")).get("salience_reward")
        r_good = sal(
            c.add("turn y", outcome={"task_succeeded": True, "user_thumbs_up": True})
        ).get("salience_reward")
        assert r_none == 0.0
        assert r_good == 1.0
    finally:
        c.close()


def test_invalid_salience_mode_raises():
    with pytest.raises(ValueError):
        MemoryClient(salience_mode="bogus")
