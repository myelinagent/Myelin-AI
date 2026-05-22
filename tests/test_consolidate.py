"""F-ML-2 dream-synthesis tests — with mock LLM client."""
from __future__ import annotations

import pytest

from myelin import ConsolidationReport, MemoryClient


class _MockLLM:
    """Trivial LLM stand-in.

    `responses` is a dict from substring-of-prompt -> canned answer.
    If no key matches, returns ''. Exposes a `.complete(prompt)`
    method matching the SDK's recommended shape.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        for needle, answer in self.responses.items():
            if needle.lower() in prompt.lower():
                return answer
        return ""


def test_consolidate_no_llm_returns_well_formed_report():
    client = MemoryClient()
    client.add("Scott prefers tea.")
    client.add("Alice loves chess.")
    report = client.consolidate()
    assert isinstance(report, ConsolidationReport)
    assert report.facts_sampled == 2
    assert report.dream_queries_generated == 6  # 3 templates per fact
    assert report.llm_calls == 0
    assert report.hit_rate == 0.0
    assert report.gaps_surfaced == 0
    client.close()


def test_consolidate_with_perfect_llm_high_hit_rate():
    """An oracle LLM that always echoes the fact -> hit_rate near 1.0."""

    class OracleLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            # Extract the fact from our template "...about: <fact>"
            # and just echo it back. We rely on the fact appearing in
            # the prompt verbatim.
            return prompt

    llm = OracleLLM()
    client = MemoryClient(llm_client=llm)
    client.add("Scott prefers tea.")
    client.add("Alice loves chess.")
    report = client.consolidate()
    assert report.facts_sampled == 2
    assert report.llm_calls == 6
    # All three templates contain the fact verbatim, so every grade hits.
    assert report.hit_rate == 1.0
    assert report.gaps_surfaced == 0
    client.close()


def test_consolidate_with_selective_llm():
    """LLM answers some facts but not others -> partial hit_rate."""
    llm = _MockLLM(
        responses={
            "scott prefers tea": "Scott prefers tea.",
            # No answer for "Alice loves chess." -> miss.
        }
    )
    client = MemoryClient(llm_client=llm)
    client.add("Scott prefers tea.")
    client.add("Alice loves chess.")
    report = client.consolidate()
    # 6 queries; 3 hit (the tea queries), 3 miss (the chess queries).
    assert report.llm_calls == 6
    assert report.hit_rate == pytest.approx(0.5, abs=0.01)
    assert report.gaps_surfaced == 3


def test_consolidate_llm_call_count():
    """One call per dream-query; 3 templates per fact."""
    llm = _MockLLM()
    client = MemoryClient(llm_client=llm)
    for i in range(4):
        client.add(f"fact {i}")
    client.consolidate()
    assert llm.calls == 12  # 4 facts * 3 templates


def test_consolidate_respects_max_samples():
    llm = _MockLLM()
    client = MemoryClient(llm_client=llm)
    for i in range(50):
        client.add(f"fact {i}")
    report = client.consolidate(max_samples=10)
    assert report.facts_sampled == 10
    assert report.dream_queries_generated == 30


def test_consolidate_empty_substrate():
    llm = _MockLLM()
    client = MemoryClient(llm_client=llm)
    report = client.consolidate()
    assert report.facts_sampled == 0
    assert report.dream_queries_generated == 0
    assert report.llm_calls == 0


def test_consolidate_with_callable_llm():
    """A plain callable should work as llm_client."""

    calls = {"n": 0}

    def my_llm(prompt: str) -> str:
        calls["n"] += 1
        return prompt

    client = MemoryClient(llm_client=my_llm)
    client.add("fact 1")
    report = client.consolidate()
    assert report.llm_calls == 3
    assert calls["n"] == 3


def test_consolidate_with_generate_llm():
    """LLM exposing `.generate(prompt)` should work."""

    class HFStyleLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            return prompt  # echo

    llm = HFStyleLLM()
    client = MemoryClient(llm_client=llm)
    client.add("fact 1")
    report = client.consolidate()
    assert report.llm_calls == 3
    assert llm.calls == 3


def test_consolidate_invalid_llm_raises():
    class NotALLM:
        pass  # no .complete, not callable, no .generate

    client = MemoryClient(llm_client=NotALLM())
    client.add("fact 1")
    with pytest.raises(TypeError):
        client.consolidate()


def test_consolidate_skips_tombstoned_facts():
    """Tombstoned facts should not be self-tested."""
    from datetime import UTC, datetime, timedelta

    llm = _MockLLM()
    client = MemoryClient(llm_client=llm)
    # Add one healthy + one that will be tombstoned by prune.
    client.add("healthy fact", confidence=0.9, importance=0.9)
    poison = client.add(
        "poison fact",
        confidence=0.01,
        importance=0.01,
        metadata={"subject": "x", "predicate": "decided", "object": "poison"},
    )
    backend = client._backend
    backdated = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    with backend._lock:
        backend._conn.execute(
            "UPDATE memories SET t_valid = ? WHERE id = ?",
            (backdated, poison.id),
        )
        backend._conn.commit()
    client.prune()
    client.prune(now=datetime.now(UTC) + timedelta(days=14))
    # Now consolidate — only the healthy fact should be sampled.
    report = client.consolidate()
    assert report.facts_sampled == 1
    client.close()


def test_consolidate_llm_exception_marked_as_miss():
    """If the LLM raises, we treat as a missed query — not a crash."""

    class FailingLLM:
        calls = 0

        def complete(self, prompt: str) -> str:
            type(self).calls += 1
            raise RuntimeError("LLM timeout")

    llm = FailingLLM()
    client = MemoryClient(llm_client=llm)
    client.add("fact 1")
    report = client.consolidate()
    assert report.llm_calls == 3
    assert FailingLLM.calls == 3
    assert report.hit_rate == 0.0
    assert report.gaps_surfaced == 3
