"""Run LongHAUL with a sentence-transformers model as the embedder.

    python -m bench.embedder.evaluate MODEL --data locomo10.json

``MODEL`` is a directory written by ``bench.embedder.train`` or a model
id such as ``BAAI/bge-small-en-v1.5``. Every other argument is passed
to ``bench.lifetime``, whose table this prints.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from typing import Any


class SentenceEmbedder:
    """What ``MemoryClient`` needs from an embedder, with each text
    embedded once however many arms and schedules ask for it."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name)
        self._known: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            new = [t for t in dict.fromkeys(texts) if t not in self._known]
            if new:
                vectors = self._model.encode(new, normalize_embeddings=True)
                self._known.update(zip(new, vectors.tolist(), strict=True))
            return [self._known[t] for t in texts]

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    import myelin.embedders
    from bench import lifetime

    # bench.lifetime builds its embedder from this name.
    myelin.embedders.StaticEmbedder = SentenceEmbedder  # type: ignore[misc,assignment]
    return lifetime.main([*argv[1:], "--embedder", argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
