"""Lazy HF-Hub fetch + SHA256 verification for myelin trained weights.

Phase 3 architectural decision: myelin-py owns the published weight
repo (``apex-memory/myelin-contrastive-v1``) but does NOT bundle the
~140MB safetensors file inside the wheel. On first
``DMemGate(utility_backend="contrastive_v1")`` construction the encoder
calls into this module, which:

1. Checks the local cache (``MYELIN_WEIGHTS_CACHE`` env or
   ``~/.cache/myelin/``) for a previously-downloaded snapshot.
2. If absent, calls ``huggingface_hub.snapshot_download`` to pull the
   tokenizer + weights + metadata.json into the cache.
3. Verifies ``model.safetensors`` SHA256 against the module-level
   ``PINNED_SHA256`` constant. A mismatch raises
   ``WeightsShaMismatchError`` — never load tampered weights.
4. Returns the absolute cache path so the encoder can map-load the
   tensors.

The PINNED_SHA256 constant is intentionally empty until the Vertex
training run completes; the CI gate at
``tests/test_weights_fetcher.py::test_pinned_sha256_is_non_empty``
skips while empty and asserts validity once populated.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Module-level config
# --------------------------------------------------------------------

# Pinned hash of ``model.safetensors`` from the Vertex training run.
# Populated by Phase 5 step D after `test_class_accuracy >= 0.75` passes.
# Empty during pre-training development; the SHA verification short-circuits
# (logs a warning) when this is empty so the test fixture path stays clean.
PINNED_SHA256: str = ""

_DEFAULT_HF_REPO = "apex-memory/myelin-contrastive-v1"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "myelin"


# --------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------


class WeightsFetchError(RuntimeError):
    """Raised when the weights cannot be obtained — network down, repo
    missing, or unrecognized version. Distinct from
    ``WeightsShaMismatchError`` (which is a security signal, not a
    fetch-availability issue)."""


class WeightsShaMismatchError(RuntimeError):
    """Raised when the downloaded weights file SHA256 does NOT match
    ``PINNED_SHA256``. This is a tamper / corruption signal and the
    caller must NOT load the weights."""


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def fetch_weights(version: str = "contrastive_v1") -> str:
    """Return the absolute path to a verified, cached weights directory.

    Parameters
    ----------
    version : str
        The weights version label. Only ``"contrastive_v1"`` is
        recognized at present; other values raise ``WeightsFetchError``.
    """
    if version != "contrastive_v1":
        raise WeightsFetchError(
            f"unrecognized weights version {version!r}; "
            "only 'contrastive_v1' is supported"
        )

    repo_id = os.environ.get("MYELIN_MR1_HF_REPO", _DEFAULT_HF_REPO)
    cache_root = _resolve_cache_root()
    pkg_dir = cache_root / repo_id.replace("/", os.sep)

    weights_file = pkg_dir / "model.safetensors"
    if weights_file.exists() and (pkg_dir / "tokenizer.json").exists():
        _verify_sha256(weights_file)
        return str(pkg_dir)

    # Cache miss: pull from HF Hub.
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise WeightsFetchError(
            "huggingface_hub is required to fetch weights — install "
            "'pip install myelin[contrastive]'"
        ) from e

    pkg_dir.mkdir(parents=True, exist_ok=True)
    try:
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=str(pkg_dir),
            allow_patterns=[
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "metadata.json",
                "*.onnx",
            ],
        )
    except OSError as e:
        raise WeightsFetchError(
            f"hf_hub snapshot_download failed for {repo_id!r}: {e}"
        ) from e
    except Exception as e:
        raise WeightsFetchError(
            f"unexpected error fetching {repo_id!r}: {type(e).__name__}: {e}"
        ) from e

    _verify_sha256(weights_file)
    return str(pkg_dir)


def _resolve_cache_root() -> Path:
    env = os.environ.get("MYELIN_WEIGHTS_CACHE")
    if env:
        return Path(env).expanduser()
    return _DEFAULT_CACHE_DIR


def _verify_sha256(weights_file: Path) -> None:
    """Verify ``weights_file`` matches ``PINNED_SHA256``.

    Behavior matrix:
    - PINNED_SHA256 empty → log a warning (pre-release dev path).
    - PINNED_SHA256 set + match → return silently.
    - PINNED_SHA256 set + mismatch → raise ``WeightsShaMismatchError``.
    """
    if not PINNED_SHA256:
        logger.warning(
            "weights_fetcher.PINNED_SHA256 is empty — skipping verification "
            "(dev path only; release builds MUST pin a SHA)"
        )
        return
    actual = _sha256_file(weights_file)
    if actual != PINNED_SHA256:
        raise WeightsShaMismatchError(
            f"weights SHA256 mismatch — expected {PINNED_SHA256!r}, "
            f"got {actual!r} for {weights_file}"
        )


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "PINNED_SHA256",
    "WeightsFetchError",
    "WeightsShaMismatchError",
    "fetch_weights",
]
