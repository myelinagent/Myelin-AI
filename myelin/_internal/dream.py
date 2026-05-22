"""Dream-synthesis consolidation — distilled F-ML-2.

The cortex F-ML-2 mechanism (`scripts/vertex/_harness.py`,
`src/cortex/scheduler/jobs/F_ML_2_dream_synthesis.py`) does:

1. Sample post-sleep T2/T3 facts that are eligible for self-test.
2. Generate "dream queries" — paraphrased natural-language questions
   whose ground-truth answer is the seeded fact's content.
3. Call the answerer (in cortex: gpt-4o-mini at seed=42) on each query.
4. Grade hit/miss by substring match between the answerer's response
   and the seed fact's content.
5. Return per-cycle `hit_rate`, gap count, and any newly-extracted
   facts to write back.

This module ports that loop into a dependency-free shape: it takes a
list of `Memory` and a caller-supplied `llm_client` and returns a
`ConsolidationReport`. The query generator + grader are deterministic
heuristics — the caller's LLM IS the answerer, not the consolidator.

Why heuristic generation: the cortex F-ML-2 path uses gpt-4o-mini to
generate dream queries (cost ~$0.001/qid). Bundling that would force
an OpenAI dep on every SDK install. v1 ships heuristic templates that
fire the same self-test loop; v1.1 may expose a `query_generator=`
hook for callers who want LLM-quality dream queries.
"""
from __future__ import annotations

from typing import Any, Callable

from myelin.types import ConsolidationReport, Memory

# Heuristic templates. Each template takes the fact content and emits a
# question whose ground-truth answer should contain the content (modulo
# casing and minor surface variation). Designed for substring-match
# grading; the caller's LLM answer just needs to contain the fact text.
_QUERY_TEMPLATES: tuple[str, ...] = (
    "What do you remember about: {fact}",
    "Tell me what you know related to: {fact}",
    "Recall the fact: {fact}",
)


def generate_dream_queries(memory: Memory) -> list[str]:
    """Generate dream queries for a single memory. Deterministic."""
    return [tpl.format(fact=memory.content) for tpl in _QUERY_TEMPLATES]


def grade_hit(answer: str, memory: Memory) -> bool:
    """Substring-match grading: does the answer mention the fact?

    Mirrors the F-ML-2 substring-match grader. Lowercase + whitespace-
    normalize both sides before checking. Empty answer always misses.
    """
    if not answer:
        return False
    needle = " ".join(memory.content.lower().split())
    haystack = " ".join(answer.lower().split())
    return needle in haystack


def run_dream_cycle(
    *,
    memories: list[Memory],
    llm_client: Any | None,
    max_samples: int = 32,
) -> ConsolidationReport:
    """Run one dream-synthesis cycle over `memories`.

    Caller-supplied `llm_client` must expose a `.complete(prompt: str)
    -> str` method (most popular LLM SDKs do; thin adapters are trivial
    to write — see README).

    Returns a ConsolidationReport. When `llm_client` is None, the report
    is well-formed with `hit_rate=0.0`, `llm_calls=0`, but the count of
    sampled facts and generated queries is still surfaced so the caller
    can verify the sampling step ran.

    `max_samples` caps the number of memories self-tested per call to
    keep cost bounded; 32 mirrors the F-ML-2 per-qid cap.
    """
    eligible = [
        m
        for m in memories
        if m.health_state == "healthy" and m.t_invalid is None
    ]
    sample = eligible[:max_samples]

    queries: list[tuple[Memory, str]] = []
    for m in sample:
        for q in generate_dream_queries(m):
            queries.append((m, q))

    report = ConsolidationReport(
        dream_queries_generated=len(queries),
        facts_sampled=len(sample),
    )

    if llm_client is None or not queries:
        return report

    hits = 0
    answer_func = _resolve_llm_callable(llm_client)
    for memory, query in queries:
        try:
            answer = answer_func(query)
        except Exception:  # noqa: BLE001 — caller's LLM, anything goes
            answer = ""
        report.llm_calls += 1
        if grade_hit(answer, memory):
            hits += 1

    report.hit_rate = hits / report.llm_calls if report.llm_calls else 0.0
    report.gaps_surfaced = report.llm_calls - hits
    # `new_facts_extracted` would require a second-pass extractor over
    # the answerer's responses. v1 leaves this at 0 — the dream cycle
    # surfaces gaps but does not auto-write back. v1.1 may add an
    # extractor hook so the caller's LLM can produce structured facts
    # from each answer.
    return report


def _resolve_llm_callable(llm_client: Any) -> Callable[[str], str]:
    """Best-effort adapter to a `complete(prompt)` interface.

    Supports any of:
    - `client.complete(prompt) -> str`     (our recommended shape)
    - `client(prompt) -> str`              (plain callable)
    - `client.generate(prompt) -> str`     (HF-style)

    Raises TypeError if none of these match — caller must adapt their
    LLM client to one of these shapes.
    """
    if hasattr(llm_client, "complete"):
        return llm_client.complete  # type: ignore[no-any-return]
    if callable(llm_client):
        return llm_client  # type: ignore[no-any-return]
    if hasattr(llm_client, "generate"):
        return llm_client.generate  # type: ignore[no-any-return]
    raise TypeError(
        "llm_client must expose .complete(prompt), be callable(prompt), "
        "or expose .generate(prompt). See myelin README for examples."
    )
