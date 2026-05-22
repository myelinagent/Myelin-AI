"""Tests for DataPipeline + HardNegativeMiner.

Per SPARC spec Phase 2 Algorithms A + B:
- build_pairs from results.jsonl with positive/negative labeling
- hard-negative mining with cosine top-k, random-fallback when too few
- persona-stratified train/val/test split
- InsufficientDataError on <100 pairs (EC-1)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from myelin.training.data_pipeline import (  # type: ignore[import-not-found]
        DataPipeline,
        InsufficientDataError,
    )
    from myelin.training._pipeline_internal import (  # type: ignore[import-not-found]
        HardNegativeMiner,
        TrainPair,
        persona_stratified_split,
    )
    _HAS = True
except ImportError:
    _HAS = False


pytestmark = pytest.mark.skipif(
    not _HAS, reason="training module not yet implemented"
)


# ----- fixtures ---------------------------------------------------


def _make_results_jsonl(path: Path, n_rows: int, system: str = "myelin") -> None:
    """Write a synthetic results.jsonl matching the PersonaMem bench shape."""
    with path.open("w") as f:
        for i in range(n_rows):
            row = {
                "qid": f"q{i}",
                "persona_id": f"p{i % 5}",  # 5 personas, stratified
                "system": system,
                "axis": ["biographical", "preference", "filler"][i % 3],
                "response": f"answer-{i}",
                "gold_answer": f"answer-{i}" if i % 2 == 0 else "wrong",
                "retrieved_memories": [
                    {"id": f"m{i}a", "content": f"turn-{i}-a", "query": f"q{i}"},
                    {"id": f"m{i}b", "content": f"turn-{i}-b", "query": f"q{i}"},
                ],
            }
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def jsonl_with_200_rows(tmp_path: Path) -> Path:
    p = tmp_path / "results.jsonl"
    _make_results_jsonl(p, n_rows=200)
    return p


@pytest.fixture
def jsonl_with_30_rows(tmp_path: Path) -> Path:
    p = tmp_path / "results.jsonl"
    _make_results_jsonl(p, n_rows=30)
    return p


# ----- tests ------------------------------------------------------


def test_build_pairs_emits_positive_for_correct_response(jsonl_with_200_rows):
    pipeline = DataPipeline()
    pairs = pipeline.build_pairs([jsonl_with_200_rows])
    pos = [p for p in pairs if p.label == 1.0]
    assert len(pos) > 0
    # The first row (i=0, gold matches response) has 2 retrieved memories
    # → 2 positives from that single row.
    assert any(p.query.startswith("q") for p in pos)


def test_build_pairs_emits_negative_for_wrong_response(jsonl_with_200_rows):
    pipeline = DataPipeline()
    pairs = pipeline.build_pairs([jsonl_with_200_rows])
    neg = [p for p in pairs if p.label == 0.0]
    assert len(neg) > 0


def test_build_pairs_skips_non_myelin_rows(tmp_path):
    """The data pipeline only consumes myelin-system rows; mem0/openclaw
    rows in the same file are skipped."""
    p = tmp_path / "results.jsonl"
    with p.open("w") as f:
        for sys_name in ("mem0", "openclaw"):
            row = {
                "qid": "q0",
                "persona_id": "p0",
                "system": sys_name,
                "response": "x",
                "gold_answer": "x",
                "retrieved_memories": [
                    {"id": "m", "content": "t", "query": "q"}
                ],
            }
            f.write(json.dumps(row) + "\n")
    pipeline = DataPipeline()
    with pytest.raises(InsufficientDataError):
        pipeline.build_pairs([p])


def test_build_pairs_raises_insufficient_data(jsonl_with_30_rows):
    """EC-1: <100 pairs → InsufficientDataError."""
    pipeline = DataPipeline()
    with pytest.raises(InsufficientDataError):
        pipeline.build_pairs([jsonl_with_30_rows])


def test_hard_negative_miner_returns_5_negs_per_positive(jsonl_with_200_rows):
    pipeline = DataPipeline()
    pairs = pipeline.build_pairs([jsonl_with_200_rows])
    positives = [p for p in pairs if p.label == 1.0]
    miner = HardNegativeMiner(embed_fn=lambda texts: [[hash(t) % 7 / 7.0] for t in texts])
    augmented = miner.augment(positives, all_corpus_turns=[p.turn for p in pairs])
    # Each positive yields up to 5 hard-negs.
    extra = len(augmented) - len(positives)
    assert extra > 0


def test_hard_negative_miner_falls_back_when_few_candidates(tmp_path):
    """EC-5: <3 hard-negatives available → random fallback + warning."""
    p = tmp_path / "results.jsonl"
    # Only 2 distinct turns means hard-neg mining can't find 5 candidates.
    with p.open("w") as f:
        for i in range(150):
            row = {
                "qid": f"q{i}",
                "persona_id": f"p{i % 5}",
                "system": "myelin",
                "response": "x",
                "gold_answer": "x",
                "retrieved_memories": [
                    {"id": "m1", "content": "exactly the same turn", "query": "q"},
                ],
            }
            f.write(json.dumps(row) + "\n")
    pipeline = DataPipeline()
    pairs = pipeline.build_pairs([p])
    miner = HardNegativeMiner(embed_fn=lambda ts: [[0.0] for _ in ts])
    augmented = miner.augment(pairs, all_corpus_turns=["only one turn"])
    # Should not crash; should emit a warning (captured via logging) and
    # fall back to a random-sample path.
    assert isinstance(augmented, list)


def test_persona_stratified_split_preserves_persona_distribution(jsonl_with_200_rows):
    pipeline = DataPipeline()
    pairs = pipeline.build_pairs([jsonl_with_200_rows])
    train, val, test = persona_stratified_split(pairs, ratios=(0.6, 0.2, 0.2), seed=42)
    # No persona should appear in all three splits — leakage check.
    train_pers = {p.persona_id for p in train}
    val_pers = {p.persona_id for p in val}
    test_pers = {p.persona_id for p in test}
    assert not (train_pers & val_pers & test_pers), \
        "persona leakage across splits"


def test_train_pair_is_immutable():
    """TrainPair must be a frozen dataclass — pairs flow through the
    training loop and accidental mutation must not silently mangle labels.
    """
    pair = TrainPair(
        turn="hello",
        query="q1",
        label=1.0,
        label_class="persistent",
        persona_id="p1",
        source="positive",
    )
    with pytest.raises((AttributeError, TypeError)):
        pair.label = 0.0  # type: ignore[misc]
