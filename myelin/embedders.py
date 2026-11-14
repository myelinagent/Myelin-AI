"""Embedders for hybrid search.

``MemoryClient(embedder=...)`` accepts anything with an
``encode(text) -> sequence of floats`` method (or a plain callable). This
module provides one that needs no GPU and no deep-learning framework:

    from myelin import MemoryClient
    from myelin.embedders import StaticEmbedder

    client = MemoryClient(embedder=StaticEmbedder())

``StaticEmbedder`` wraps a model2vec static model: a lookup table of
token vectors that are averaged, so encoding takes microseconds on a
CPU. Install it with ``pip install "myelin-ai[static]"``. The model is
downloaded from the Hugging Face Hub on first use and cached.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_STATIC_MODEL = "minishlab/potion-retrieval-32M"


class StaticEmbedder:
    """A model2vec static embedding model, loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_STATIC_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:
                raise ImportError(
                    "StaticEmbedder needs the model2vec package: "
                    'pip install "myelin-ai[static]"'
                ) from exc
            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, text: str) -> Sequence[float]:
        return self._load().encode([text])[0].tolist()

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts in one call; much faster than one at a time."""
        return self._load().encode(list(texts)).tolist()


__all__ = ["DEFAULT_STATIC_MODEL", "StaticEmbedder"]
