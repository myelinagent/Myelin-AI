"""Singleton inference path for the Memory-R1 contrastive encoder.

Per SPARC spec Phase 2 Algorithm D + Phase 4 Refinement:

* Load-once singleton (multiple ``DMemGate`` constructions share one
  model instance — saves ~140MB RAM + ~2s init each).
* Lazy load — first ``get_instance()`` triggers weights fetch.
* ONNX-first inference — if the cache dir contains ``model.onnx`` we
  use ``onnxruntime`` (2-3x faster on CPU than naive PyTorch forward).
  Falls back to safetensors + torch if ONNX absent.
* Tokenizer pre-compiled — the fast Rust BPE is built once and reused
  for every ``score()`` call.

The encoder returns ``(utility ∈ [0,1], class ∈
{transient|short_term|persistent})``. The caller (``DMemGate``)
multiplies utility by the Z-normalized surprise to derive the RPE.

Memory shape:
* Input string max 512 chars context + the per-turn content.
* Tokenizer max_length controlled by ``MYELIN_MR1_MAX_LEN`` env
  (default 512).
* Returns scalars — no tensor lifecycle leaks across calls.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

UtilityClass = Literal["transient", "short_term", "persistent"]
_CLASS_NAMES: tuple[UtilityClass, ...] = ("transient", "short_term", "persistent")


class ContrastiveEncoderLoadError(RuntimeError):
    """Raised when the encoder cannot be initialized (weights missing,
    SHA mismatch, torch import failed, tokenizer broken, etc.)."""


class ContrastiveEncoder:
    """Process-singleton wrapper around the trained DeBERTaV3-small
    joint-head model.

    Usage::

        encoder = ContrastiveEncoder.get_instance()
        utility, cls = encoder.score(content="...", neighbor_context="...")
    """

    _instance: Optional["ContrastiveEncoder"] = None
    _lock = threading.Lock()

    # Public knob — Phase 4 refinement says context is capped at 512
    # chars before tokenization. Truncation is the caller-visible
    # contract that ``test_score_truncates_long_context`` asserts.
    max_context_chars: int = 512

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._onnx_session: Any = None
        self._max_length: int = int(os.environ.get("MYELIN_MR1_MAX_LEN", "512"))
        self._loaded: bool = False

    # ----- singleton API -------------------------------------------

    @classmethod
    def get_instance(cls) -> "ContrastiveEncoder":
        """Return the process-wide singleton; load on first call."""
        if cls._instance is not None and cls._instance._loaded:
            return cls._instance
        with cls._lock:
            if cls._instance is not None and cls._instance._loaded:
                return cls._instance
            inst = cls()
            try:
                inst._load()
            except ContrastiveEncoderLoadError:
                raise
            except Exception as e:
                raise ContrastiveEncoderLoadError(
                    f"{type(e).__name__}: {e}"
                ) from e
            cls._instance = inst
            return inst

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Drop the singleton — test-only helper."""
        with cls._lock:
            cls._instance = None

    # ----- public score path ---------------------------------------

    def score(
        self,
        *,
        content: str,
        neighbor_context: str,
    ) -> tuple[float, UtilityClass]:
        """Return (utility ∈ [0,1], class). See module docstring."""
        if not self._loaded:
            raise ContrastiveEncoderLoadError("encoder not loaded")
        # Truncate neighbor_context for the SPARC contract (512-char cap).
        ctx = neighbor_context[: self.max_context_chars]
        text = f"[TURN] {content} [CONTEXT] {ctx}"
        if self._onnx_session is not None:
            util, cls_idx = self._forward_onnx(text)
        else:
            util, cls_idx = self._forward_torch(text)
        util = _clamp_unit(util)
        return util, _CLASS_NAMES[cls_idx]

    # ----- internal: load + forward --------------------------------

    def _load(self) -> None:
        """Fetch weights (via weights_fetcher) and build model + tokenizer."""
        from myelin._internal import weights_fetcher

        weights_dir_str = weights_fetcher.fetch_weights("contrastive_v1")
        weights_dir = Path(weights_dir_str)
        # Try ONNX first for CPU speed; fall back to torch + safetensors.
        onnx_path = weights_dir / "model.onnx"
        if onnx_path.exists():
            self._build_onnx(weights_dir)
        else:
            self._build_model(weights_dir)
        self._loaded = True

    def _build_onnx(self, weights_dir: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ContrastiveEncoderLoadError(
                "onnxruntime not available — install myelin[contrastive]"
            ) from e
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ContrastiveEncoderLoadError(
                "transformers required — install myelin[contrastive]"
            ) from e
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(weights_dir), use_fast=True
            )
        except Exception as e:
            raise ContrastiveEncoderLoadError(
                f"tokenizer load failed: {type(e).__name__}: {e}"
            ) from e
        try:
            self._onnx_session = ort.InferenceSession(
                str(weights_dir / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise ContrastiveEncoderLoadError(
                f"ONNX session init failed: {type(e).__name__}: {e}"
            ) from e

    def _build_model(self, weights_dir: Path | None = None) -> None:
        """Build a torch model + tokenizer from a safetensors snapshot.

        Test code monkey-patches this method to skip the heavy DeBERTa
        load — see tests/test_contrastive_encoder.py.
        """
        try:
            import torch  # type: ignore[import-not-found]
            from safetensors.torch import load_file  # type: ignore[import-not-found]
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ContrastiveEncoderLoadError(
                f"required package missing: {type(e).__name__}: {e}; "
                "install myelin[contrastive]"
            ) from e
        if weights_dir is None:
            raise ContrastiveEncoderLoadError("weights_dir missing")

        # The training module owns the model factory; we import lazily to
        # keep this module dependency-free in inference-only deployments.
        try:
            from myelin.training.model_factory import build_joint_model  # type: ignore[import-not-found]
        except ImportError as e:
            raise ContrastiveEncoderLoadError(
                f"model factory missing: {e}; install myelin[contrastive]"
            ) from e

        model = build_joint_model(use_pretrained=True)
        state = load_file(str(weights_dir / "model.safetensors"))
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            raise ContrastiveEncoderLoadError(
                f"state_dict load failed: {type(e).__name__}: {e}"
            ) from e
        # Switch to inference mode (calls torch's no-grad/eval method).
        model.eval()  # noqa: PIE790
        self._model = model
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(weights_dir), use_fast=True
            )
        except Exception as e:
            raise ContrastiveEncoderLoadError(
                f"tokenizer load failed: {type(e).__name__}: {e}"
            ) from e

    def _forward_onnx(self, text: str) -> tuple[float, int]:
        import numpy as np  # type: ignore[import-not-found]
        toks = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        feed = {
            "input_ids": toks["input_ids"].astype("int64"),
            "attention_mask": toks["attention_mask"].astype("int64"),
        }
        outputs = self._onnx_session.run(None, feed)
        util_logit = float(outputs[0].reshape(-1)[0])
        class_logits = outputs[1].reshape(-1)
        util = 1.0 / (1.0 + math.exp(-util_logit))
        cls_idx = int(np.argmax(class_logits))
        return util, cls_idx

    def _forward_torch(self, text: str) -> tuple[float, int]:
        import torch  # type: ignore[import-not-found]
        toks = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            util_logit, class_logits = self._model(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
            )
        util_logit_v = float(util_logit.reshape(-1)[0])
        util = 1.0 / (1.0 + math.exp(-util_logit_v))
        cls_idx = int(torch.argmax(class_logits.reshape(-1)).item())
        return util, cls_idx


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


__all__ = [
    "ContrastiveEncoder",
    "ContrastiveEncoderLoadError",
    "UtilityClass",
]
