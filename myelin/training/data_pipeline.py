"""Public DataPipeline.build_pairs — Algorithm A from the SPARC spec.

Consumes PersonaMem results.jsonl files emitted by the bench harness,
returns ``list[TrainPair]`` ready for hard-negative mining + training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from myelin.training._pipeline_internal import (
    TrainPair,
    classify_gold_answer,
)

logger = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    """Raised when the consolidated jsonl yields < 100 training pairs
    (EC-1). Tiny datasets can't support a contrastive objective; the
    caller must add more bench runs to the staging GCS prefix."""


_MIN_PAIRS = 100


class DataPipeline:
    """Build training pairs from PersonaMem results.jsonl files."""

    def __init__(self, *, system: str = "myelin", min_pairs: int = _MIN_PAIRS) -> None:
        self.system = system
        self.min_pairs = min_pairs

    def build_pairs(self, files: Iterable[Path | str]) -> list[TrainPair]:
        pairs: list[TrainPair] = []
        rows_seen = 0
        rows_kept = 0
        for path in files:
            p = Path(path)
            if not p.exists():
                logger.warning("DataPipeline: missing file %s — skipping", p)
                continue
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("skipping malformed jsonl row in %s", p)
                        continue
                    rows_seen += 1
                    if row.get("system") != self.system:
                        continue
                    rows_kept += 1
                    pairs.extend(self._row_to_pairs(row))
        logger.info(
            "DataPipeline: %d rows seen, %d kept (system=%s), %d pairs emitted",
            rows_seen, rows_kept, self.system, len(pairs),
        )
        if len(pairs) < self.min_pairs:
            raise InsufficientDataError(
                f"only {len(pairs)} training pairs (need >= {self.min_pairs}) — "
                "stage more PersonaMem bench artifacts and retry"
            )
        return pairs

    def _row_to_pairs(self, row: dict) -> list[TrainPair]:
        """Convert a single bench row into 0+ TrainPair instances."""
        gold = str(row.get("gold_answer", ""))
        response = str(row.get("response", ""))
        is_correct = _substring_match(response, gold)
        label_class = classify_gold_answer(gold)
        persona_id = str(row.get("persona_id", "unknown"))
        query = str(row.get("qid", ""))
        out: list[TrainPair] = []
        retrieved = row.get("retrieved_memories", []) or []
        for mem in retrieved:
            if isinstance(mem, dict):
                turn = str(mem.get("content", ""))
            else:
                turn = str(mem)
            if not turn:
                continue
            out.append(TrainPair(
                turn=turn,
                query=query,
                label=1.0 if is_correct else 0.0,
                label_class=label_class if is_correct else "transient",
                persona_id=persona_id,
                source="positive" if is_correct else "negative",
            ))
        return out


def _substring_match(response: str, gold: str) -> bool:
    if not response or not gold:
        return False
    return gold.strip().lower() in response.strip().lower()


__all__ = ["DataPipeline", "InsufficientDataError"]
