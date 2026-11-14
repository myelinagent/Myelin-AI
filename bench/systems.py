"""Memory systems under test, behind one small interface.

A system ingests timestamped conversation turns for one benchmark example
and returns the turns it ranks highest for a question. ``MyelinSystem`` is
the system being measured; ``BM25System`` is a control that ranks the same
turns with textbook BM25, so the gap between the two is retrieval quality
alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from myelin import MemoryClient

_USER = "bench"


@dataclass(frozen=True)
class Turn:
    session_id: str
    session_date: datetime
    index: int
    role: str
    content: str
    # Marked by the benchmark as containing the answer.
    evidence: bool = False

    @property
    def text(self) -> str:
        return f"{self.role}: {self.content}"


class MemorySystem(Protocol):
    name: str

    def reset(self) -> None: ...

    def ingest(self, turns: list[Turn]) -> None: ...

    def retrieve(self, question: str, k: int) -> list[Turn]: ...


class MyelinSystem:
    """Myelin as shipped: one memory per turn, default search."""

    name = "myelin"
    # Shared by every instance so a model is loaded once per process.
    embedder: Any = None

    def __init__(self) -> None:
        self._client: MemoryClient | None = None
        self._turns: dict[str, Turn] = {}

    def reset(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = MemoryClient(user_id=_USER, embedder=type(self).embedder)
        self._turns = {}

    def ingest(self, turns: list[Turn]) -> None:
        assert self._client is not None, "call reset() first"
        for turn in turns:
            memory = self._client.add(
                turn.text,
                t_valid=turn.session_date,
                metadata={"session_id": turn.session_id, "turn": turn.index},
            )
            self._turns[memory.id] = turn

    def retrieve(self, question: str, k: int) -> list[Turn]:
        assert self._client is not None, "call reset() first"
        hits = self._client.search(question, top_k=k)
        return [self._turns[hit.memory.id] for hit in hits]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with and or not "
    "what who where when why how do does did this that these those i you he she "
    "it we they me him her us them my your his its our their s".split()
)


def _terms(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


class BM25System:
    """Control: Okapi BM25 over the same turns (k1=1.5, b=0.75)."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self.reset()

    def reset(self) -> None:
        self._turns: list[Turn] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()

    def ingest(self, turns: list[Turn]) -> None:
        for turn in turns:
            terms = _terms(turn.text)
            tf = Counter(terms)
            self._turns.append(turn)
            self._tfs.append(tf)
            self._lengths.append(len(terms))
            self._df.update(tf.keys())

    def retrieve(self, question: str, k: int) -> list[Turn]:
        n = len(self._turns)
        if n == 0:
            return []
        avg_len = sum(self._lengths) / n or 1.0
        query = set(_terms(question))
        idf = {
            t: math.log(1.0 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            for t in query
            if t in self._df
        }
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self._k1 * (1.0 - self._b + self._b * self._lengths[i] / avg_len)
            for term, weight in idf.items():
                freq = tf.get(term, 0)
                if freq:
                    score += weight * freq * (self._k1 + 1.0) / (freq + norm)
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._turns[i] for _, i in scored[:k]]

    def close(self) -> None:
        self.reset()


SYSTEMS = {"myelin": MyelinSystem, "bm25": BM25System}
