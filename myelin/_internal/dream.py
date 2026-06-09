"""Dream-synthesis consolidation — distilled F-ML-2.

The cortex F-ML-2 mechanism (`scripts/vertex/_harness.py`,
`src/cortex/scheduler/jobs/F_ML_2_dream_synthesis.py`) does:

1. Sample post-sleep T2/T3 facts that are eligible for self-test.
2. Generate "dream queries" — natural-language questions with the fact's
   answer key HELD OUT (elided), so the answer is not in the prompt.
3. Call the answerer (in cortex: gpt-4o-mini at seed=42) on each query.
4. Grade hit/miss by matching the answerer's response against the
   held-out answer key — not the verbatim fact — so an echo of the
   query cannot score a hit (it scores recall, not parroting).
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

# Held-out query templates. Each wraps the memory content with its ANSWER
# KEY elided (replaced by a blank), so the question never contains the
# answer — an answerer must recall the key from the memory store, not echo
# the prompt. This mirrors the cortex F-ML-2 grader, which scores the
# response against a separate `expected` key rather than the whole fact.
_QUERY_TEMPLATES: tuple[str, ...] = (
    "Fill in the blank from memory: {masked}",
    "What is the missing detail? {masked}",
    "Recall the elided word: {masked}",
)

# Common words that are never the salient "answer" of a fact. Kept small and
# dependency-free; the real F-ML-2 path uses an LLM to pick the held-out span.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "for", "and", "or", "but", "in", "on", "at", "by", "with",
    "as", "from", "that", "this", "these", "those", "it", "its",
    "i", "you", "he", "she", "they", "we", "me", "him", "her", "them", "us",
    "my", "your", "his", "their", "our",
    "do", "does", "did", "has", "have", "had", "will", "would", "can", "could",
    "prefers", "prefer", "likes", "like", "enjoys", "enjoy", "uses", "use",
    "used", "wants", "want", "knows", "know", "remember", "recall", "tell",
    "about", "related", "what", "which", "who", "when", "where", "how",
})


def _content_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens, in order. Punctuation is a delimiter."""
    tok: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tok.append("".join(cur))
            cur = []
    if cur:
        tok.append("".join(cur))
    return tok


def _extract_answer_key(content: str) -> str:
    """The held-out answer token for a fact. Deterministic heuristic.

    "Ask about the subject, hold out the most distinctive object word":
    skip the leading subject token, drop stopwords, then take the longest
    remaining token (ties broken by later position — objects trail
    subjects). E.g. "Scott prefers oolong tea" -> "oolong".
    """
    toks = _content_tokens(content)
    if not toks:
        return ""
    candidates = [t for t in toks[1:] if t not in _STOPWORDS]
    if not candidates:
        candidates = [t for t in toks if t not in _STOPWORDS] or toks
    return max(candidates, key=lambda t: (len(t), toks.index(t)))


def _mask_key(content: str, key: str) -> str:
    """Replace whole-word occurrences of ``key`` in ``content`` with ``___``."""
    out: list[str] = []
    word: list[str] = []
    for ch in content:
        if ch.isalnum():
            word.append(ch)
            continue
        if word:
            seg = "".join(word)
            out.append("___" if seg.lower() == key else seg)
            word = []
        out.append(ch)
    if word:
        seg = "".join(word)
        out.append("___" if seg.lower() == key else seg)
    return "".join(out)


def generate_dream_queries(memory: Memory) -> list[str]:
    """Generate held-out dream queries for a memory. Deterministic.

    Each query elides the answer key, so the fact is withheld from the
    answerer's prompt — an echo of the query cannot score a hit.
    """
    key = _extract_answer_key(memory.content)
    masked = _mask_key(memory.content, key) if key else memory.content
    return [tpl.format(masked=masked) for tpl in _QUERY_TEMPLATES]


def grade_hit(answer: str, memory: Memory) -> bool:
    """Key-recall grading: does the answer contain the memory's answer key?

    Grades the answer against the held-out key (`_extract_answer_key`),
    NOT the whole fact — so echoing the (key-elided) question misses, while
    a correct paraphrase that surfaces the key hits. Empty answer always
    misses. Whole-word match after lowercasing.
    """
    if not answer:
        return False
    key = _extract_answer_key(memory.content)
    if not key:
        return False
    return key in _content_tokens(answer)


def _ground_prompt(context: list[str], query: str) -> str:
    """Prepend retrieved memories to a dream query so the answerer must recall
    from the store, not guess. ADR-027 Track C: turns the self-test from an
    ungrounded cloze into a real recall test."""
    facts = "\n".join(f"- {c}" for c in context)
    return f"Remembered facts:\n{facts}\n\n{query}"


def run_dream_cycle(
    *,
    memories: list[Memory],
    llm_client: Any | None,
    max_samples: int = 32,
    retriever: Callable[[str], Any] | None = None,
) -> ConsolidationReport:
    """Run one dream-synthesis cycle over `memories`.

    Caller-supplied `llm_client` must expose a `.complete(prompt: str)
    -> str` method (most popular LLM SDKs do; thin adapters are trivial
    to write — see README).

    `retriever` (ADR-027 Track C) — optional `Callable[[query], Iterable[str]]`
    that returns context snippets for a dream query (e.g.
    ``lambda q: [r.memory.content for r in client.search(q, top_k=k)]``). When
    supplied, the answerer is prompted WITH the retrieved memories, so the
    self-test measures genuine store-grounded recall rather than cloze-from-
    query. Default ``None`` preserves the v1 (ungrounded) behaviour exactly: a
    retriever miss (empty result or raise) falls back to the bare query, never
    a crash.

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
        prompt = query
        if retriever is not None:
            try:
                context = [str(c) for c in retriever(query)]
            except Exception:  # noqa: BLE001 — caller's retriever, anything goes
                context = []
            if context:
                prompt = _ground_prompt(context, query)
        try:
            answer = answer_func(prompt)
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
