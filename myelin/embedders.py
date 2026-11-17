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

``NeuralEmbedder`` wraps a small transformer fine-tuned for finding the
conversation turn that answers a question. It finds more than the
static model and is slower: it needs PyTorch, and embeds hundreds of
texts a second on a CPU where the static model embeds tens of
thousands. Install it with ``pip install "myelin-ai[neural]"``:

    from myelin.embedders import NeuralEmbedder

    client = MemoryClient(embedder=NeuralEmbedder())

Vectors from different embedders are not compared with each other.
After changing embedder, run ``client.backfill_embeddings()``: it
re-embeds the memories the previous one embedded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_STATIC_MODEL = "minishlab/potion-retrieval-32M"
DEFAULT_NEURAL_MODEL = "myelinagent/myelin-embed-small-en-v1"


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


class NeuralEmbedder:
    """A sentence-transformers model, loaded on first use.

    The default is ``myelinagent/myelin-embed-small-en-v1``: a 33M
    parameter English model (``BAAI/bge-small-en-v1.5`` fine-tuned on
    synthetic conversations). Any sentence-transformers model id or
    local directory works. ``device`` is passed through; ``None`` lets
    sentence-transformers pick a GPU when there is one.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NEURAL_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "NeuralEmbedder needs the sentence-transformers package: "
                    'pip install "myelin-ai[neural]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, text: str) -> Sequence[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts in one call; much faster than one at a time."""
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


__all__ = [
    "DEFAULT_NEURAL_MODEL",
    "DEFAULT_STATIC_MODEL",
    "NeuralEmbedder",
    "StaticEmbedder",
]
