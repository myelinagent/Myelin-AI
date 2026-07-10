"""Deterministic 1,000-episode promoter/consolidation benchmark.

Measures the end-to-end cost of:
  - add()            1,000 facts (episodes)
  - consolidate()    dream self-test across all healthy facts
  - prune()          quarantine + caspase pass

The test is deterministic (no random, no LLM, no wall-clock sleeps) so
per-run variance comes from system factors (filesystem cache, CPU
frequency scaling, background load). On a modern server-class core you
should see consolidate() << 5 s and prune() << 3 s.

Caveats:
  - SQLite :memory: backend — I/O cost is minimal; a persistent-backed
    run would add fsync latency.
  - The consolidation loop in this test uses a trivial echo-LLM that
    returns the prompt, so every query is a gradable (zero-hit) answer.
    A real LLM would dominate wall time (10–100 ms/call × 3000 calls).
  - All times are real (wall) clock on a shared machine; no warmup or
    statistical burn-in is performed.
  - Numbers quoted in the report are from the specific hardware this
    ran on and are NOT portable to different CPU generations, Python
    runtimes, or storage backends.
"""

from __future__ import annotations

import datetime
import time

import pytest

from myelin import MemoryClient


def _fake_llm(prompt: str) -> str:
    """Trivial echo — no LLM cost; lets us measure the framework overhead."""
    return prompt


def _episode(client: MemoryClient, i: int, *, event_anchored: bool = False) -> None:
    """Simulate one episode: one fact + optional event anchoring."""
    kwargs = {}
    if event_anchored:
        from myelin import Event

        kwargs["event"] = Event(
            event_id=f"ev-{i}",
            type="conversation",
            anchor_datetime=datetime.datetime(2026, 1, 1)
            + datetime.timedelta(hours=i),
        )
    client.add(
        f"Episode {i}: agent prefers {'tea' if i % 2 == 0 else 'coffee'} "
        f"with task score {i % 5}.",
        metadata={
            "subject": f"episode_{i}",
            "predicate": "prefers",
            "object": "tea" if i % 2 == 0 else "coffee",
        },
        **kwargs,
    )


@pytest.mark.benchmark
class TestConsolidateBenchmark:
    """1,000-episode benchmark for the promoter/consolidation path."""

    @staticmethod
    def _run_benchmark(n_episodes: int = 1000) -> dict:
        client = MemoryClient(llm_client=_fake_llm, salience_mode="off")

        # --- Phase 1: ingest ------------------------------------------------
        t0 = time.perf_counter()
        ev_anchored = n_episodes // 10  # 10% event-anchored
        for i in range(n_episodes):
            _episode(client, i, event_anchored=i < ev_anchored)
        t_ingest = time.perf_counter() - t0

        # --- Phase 2: consolidate -------------------------------------------
        t0 = time.perf_counter()
        consolidate_report = client.consolidate(max_samples=n_episodes)
        t_consolidate = time.perf_counter() - t0

        # --- Phase 3: prune -------------------------------------------------
        fast_forward = (
            datetime.datetime.now(datetime.UTC)
            + datetime.timedelta(days=14)
        )
        t0 = time.perf_counter()
        prune_report = client.prune(now=fast_forward)
        t_prune = time.perf_counter() - t0

        # --- Phase 4: working-set search (tests batch get path) -------------
        client2 = MemoryClient(
            llm_client=_fake_llm,
            salience_mode="off",
            working_memory=True,
        )
        import uuid

        session_id = str(uuid.uuid4())
        # Observe + commit a handful of turns to build a working set
        observed = []
        for i in range(100):
            turn = client2.observe(
                f"working fact {i}: something about tea",
                session_id=session_id,
            )
            observed.append(turn.turn_id)
        client2.commit(observed, session_id=session_id)
        # Warm up the working set by searching a few times
        for _ in range(3):
            client2.search("tea", session_id=session_id, modes=("working",))
        t0 = time.perf_counter()
        for _ in range(10):
            ws = client2.search(
                "tea", session_id=session_id, modes=("working", "long_term"),
            )
        t_working_search = time.perf_counter() - t0
        client2.close()

        # --- Phase 5: batch get benchmark ------------------------------------
        all_ids = [m.id for m in client._backend.list_memories(
            user_id="default"
        )]
        t0 = time.perf_counter()
        # 10 rounds of batch-get with random subsets of 50 ids each
        import random

        rng = random.Random(42)
        for _ in range(10):
            subset = rng.sample(all_ids, min(50, len(all_ids)))
            _ = client._backend.get_batch(subset)
        t_batch_get = time.perf_counter() - t0

        client.close()

        return {
            "n_episodes": n_episodes,
            "t_ingest": t_ingest,
            "t_consolidate": t_consolidate,
            "t_prune": t_prune,
            "t_working_search_10x": t_working_search,
            "t_batch_get_10runs": t_batch_get,
            "consolidate": {
                "facts_sampled": consolidate_report.facts_sampled,
                "dream_queries": consolidate_report.dream_queries_generated,
                "llm_calls": consolidate_report.llm_calls,
                "hit_rate": consolidate_report.hit_rate,
                "gaps": consolidate_report.gaps_surfaced,
            },
            "prune": {
                "scanned": prune_report.scanned,
                "quarantined": prune_report.quarantined_count,
                "tombstoned": prune_report.tombstoned_count,
                "retained": prune_report.retained_count,
            },
        }

    def test_1000_episode_benchmark(self) -> None:
        """Run the 1,000-episode benchmark and assert deterministic shape."""
        result = self._run_benchmark(1000)

        # --- Correctness assertions (semantics preserved) ----
        assert result["consolidate"]["facts_sampled"] == 1000
        assert result["consolidate"]["dream_queries"] == 3000  # 3 templates
        assert result["consolidate"]["llm_calls"] == 3000
        # Echo answerer always misses (held-out key is not in echoed prompt)
        assert result["consolidate"]["hit_rate"] == 0.0
        assert result["consolidate"]["gaps"] == 3000

        # Prune scanned all
        assert result["prune"]["scanned"] == 1000
        # Retained + quarantined should cover all healthy facts
        # (no tombstoned at this point since nothing was quarantined before)
        assert (
            result["prune"]["quarantined"]
            + result["prune"]["retained"]
            + result["prune"]["tombstoned"]
            == 1000
        )

        # --- Performance output (not assertions — for human review) ----
        print(f"\n{'='*60}")
        print(f"  Benchmark: {result['n_episodes']:,} episodes")
        print(f"{'='*60}")
        print(f"  ingest           : {result['t_ingest']:.3f} s")
        print(f"  consolidate      : {result['t_consolidate']:.3f} s")
        print(f"  prune            : {result['t_prune']:.3f} s")
        print(f"  working search×10: {result['t_working_search_10x']:.3f} s")
        print(f"  batch get×10     : {result['t_batch_get_10runs']:.3f} s")
        print(
            f"  consolidate: "
            f"{result['consolidate']['facts_sampled']} facts → "
            f"{result['consolidate']['dream_queries']} queries, "
            f"hit_rate={result['consolidate']['hit_rate']}"
        )
        print(
            f"  prune: "
            f"{result['prune']['scanned']} scanned, "
            f"{result['prune']['quarantined']} quarantined, "
            f"{result['prune']['tombstoned']} tombstoned, "
            f"{result['prune']['retained']} retained"
        )
        print(f"{'='*60}")
        print(
            "  NOTE: Times are wall-clock on this hardware. "
            "LLM-backed runs would add ~10-100ms/call."
        )
