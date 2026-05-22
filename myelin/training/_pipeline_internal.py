"""Internal data-pipeline primitives.

* ``TrainPair`` — single training instance (turn + query + label).
* ``HardNegativeMiner.augment`` — top-k cosine hard-negs per positive,
  random-fallback when too few candidates (EC-5).
* ``persona_stratified_split`` — split TrainPairs into (train, val,
  test) without persona-id leakage.
* ``classify_gold_answer`` — heuristic label_class assignment from a
  gold answer string (biographical → persistent, preference → short_term,
  filler → transient).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainPair:
    """A single training instance.

    ``turn`` is the retrieved memory text; ``query`` is the user
    question against which utility is judged. ``label`` is a binary
    scalar (1.0 = retrieval-useful, 0.0 = not). ``label_class`` is the
    3-way classification target (transient/short_term/persistent).
    ``source`` is provenance — "positive", "hard-neg", or "random-neg".
    """

    turn: str
    query: str
    label: float
    label_class: str
    persona_id: str
    source: str


_PERSISTENT_KEYWORDS = (
    "name is", "birthday", "born in", "daughter", "son", "spouse",
    "lives in", "works as", "i am a", "i'm a", "married", "graduated",
)
_SHORT_TERM_KEYWORDS = (
    "favorite", "like", "love", "prefer", "enjoy", "hobby", "interest",
    "today", "tonight", "this week",
)


def classify_gold_answer(gold: str) -> str:
    """Heuristic class label for the contrastive target.

    PersonaMem doesn't label retrieved memories directly; we derive a
    weak class signal from substring patterns in the gold answer. The
    contrastive loss + the InfoNCE term still dominate; class CE is
    auxiliary.
    """
    g = gold.lower()
    for k in _PERSISTENT_KEYWORDS:
        if k in g:
            return "persistent"
    for k in _SHORT_TERM_KEYWORDS:
        if k in g:
            return "short_term"
    return "transient"


class HardNegativeMiner:
    """Cosine top-k hard-negative miner with random-fallback.

    Per Algorithm B: embed every query + every corpus turn, then for
    each positive emit up to k hard-negatives that are similar to the
    query but not in the positive's retrieved set. Falls back to a
    random sample when the corpus is too small (EC-5).
    """

    def __init__(
        self,
        embed_fn: Callable[[Sequence[str]], list[list[float]]],
        *,
        k: int = 5,
        sim_threshold: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.embed_fn = embed_fn
        self.k = k
        self.sim_threshold = sim_threshold
        self.rng = random.Random(seed)

    def augment(
        self,
        positives: Sequence[TrainPair],
        all_corpus_turns: Sequence[str],
    ) -> list[TrainPair]:
        result: list[TrainPair] = list(positives)
        if not positives or not all_corpus_turns:
            return result
        if len(set(all_corpus_turns)) < 3:
            logger.warning(
                "HardNegativeMiner: only %d distinct corpus turns; "
                "falling back to random-negative sampling for all positives",
                len(set(all_corpus_turns)),
            )
            for p in positives:
                for t in self.rng.sample(
                    list(all_corpus_turns),
                    k=min(self.k, len(all_corpus_turns)),
                ):
                    result.append(TrainPair(
                        turn=t, query=p.query, label=0.0,
                        label_class="transient",
                        persona_id=p.persona_id,
                        source="random-neg",
                    ))
            return result

        try:
            query_embs = self.embed_fn([p.query for p in positives])
            turn_embs = self.embed_fn(list(all_corpus_turns))
        except Exception as e:
            logger.warning(
                "HardNegativeMiner.embed_fn raised %s; random fallback",
                type(e).__name__,
            )
            return result

        for i, p in enumerate(positives):
            sims = [_cosine(query_embs[i], te) for te in turn_embs]
            ranked = sorted(
                range(len(all_corpus_turns)),
                key=lambda j: sims[j],
                reverse=True,
            )
            picked = 0
            for j in ranked:
                if picked >= self.k:
                    break
                if sims[j] < self.sim_threshold:
                    continue
                turn = all_corpus_turns[j]
                if turn == p.turn:
                    continue
                result.append(TrainPair(
                    turn=turn, query=p.query, label=0.0,
                    label_class="transient",
                    persona_id=p.persona_id,
                    source="hard-neg",
                ))
                picked += 1
            if picked < 3:
                logger.warning(
                    "HardNegativeMiner: only %d hard-negs for query %r; "
                    "topping up with random samples",
                    picked, p.query,
                )
                for t in self.rng.sample(
                    list(all_corpus_turns),
                    k=min(self.k - picked, len(all_corpus_turns)),
                ):
                    result.append(TrainPair(
                        turn=t, query=p.query, label=0.0,
                        label_class="transient",
                        persona_id=p.persona_id,
                        source="random-neg",
                    ))
        return result


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def persona_stratified_split(
    pairs: Sequence[TrainPair],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[list[TrainPair], list[TrainPair], list[TrainPair]]:
    """Split by persona_id so no persona appears in more than one fold.

    This protects against persona-style memorization — the train/val/
    test guarantee is that the model has never seen ANY example from a
    held-out persona, not just a held-out (persona, turn) pair.
    """
    if abs(sum(ratios) - 1.0) > 1e-3:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    by_persona: dict[str, list[TrainPair]] = {}
    for p in pairs:
        by_persona.setdefault(p.persona_id, []).append(p)
    personas = sorted(by_persona.keys())
    rng = random.Random(seed)
    rng.shuffle(personas)

    n = len(personas)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    train_p = set(personas[:n_train])
    val_p = set(personas[n_train : n_train + n_val])
    test_p = set(personas[n_train + n_val :])

    train: list[TrainPair] = []
    val: list[TrainPair] = []
    test: list[TrainPair] = []
    for persona_id, ps in by_persona.items():
        if persona_id in train_p:
            train.extend(ps)
        elif persona_id in val_p:
            val.extend(ps)
        else:
            test.extend(ps)
    return train, val, test


__all__ = [
    "TrainPair",
    "HardNegativeMiner",
    "persona_stratified_split",
    "classify_gold_answer",
]
