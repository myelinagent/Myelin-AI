"""Tests for weights_fetcher — HF Hub fetch + SHA256 verify + cache.

Per SPARC spec Phase 3: lazy HF Hub fetch with SHA256 verification,
``MYELIN_WEIGHTS_CACHE`` env override, offline cache hit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

try:
    from myelin._internal import weights_fetcher as wf  # type: ignore[import-not-found]
    from myelin._internal.weights_fetcher import (  # type: ignore[import-not-found]
        WeightsFetchError,
        WeightsShaMismatchError,
        fetch_weights,
    )
    _HAS = True
except ImportError:
    _HAS = False


pytestmark = pytest.mark.skipif(not _HAS, reason="weights_fetcher not yet implemented")


# ----- fixtures ---------------------------------------------------


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MYELIN_WEIGHTS_CACHE", str(tmp_path))
    return tmp_path


def _write_fake_pkg(d: Path, payload: bytes = b"fake-weights") -> str:
    """Write a 3-file fake model dir and return the safetensors SHA256."""
    d.mkdir(parents=True, exist_ok=True)
    weights = d / "model.safetensors"
    weights.write_bytes(payload)
    (d / "tokenizer.json").write_text("{}")
    (d / "metadata.json").write_text(
        json.dumps({"version": "contrastive_v1"})
    )
    return hashlib.sha256(payload).hexdigest()


# ----- tests ------------------------------------------------------


def test_fetch_returns_cached_dir_if_present(fake_cache, monkeypatch):
    """Cache hit path — no HF Hub call, no SHA check failure."""
    pkg_dir = fake_cache / "apex-memory" / "myelin-contrastive-v1"
    sha = _write_fake_pkg(pkg_dir)
    monkeypatch.setattr(wf, "PINNED_SHA256", sha)

    def _fail_download(*_a, **_kw):  # noqa: ANN001
        raise AssertionError("hf_hub_download must NOT be called on cache hit")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _fail_download,
        raising=False,
    )
    out = fetch_weights("contrastive_v1")
    assert Path(out).exists()
    assert (Path(out) / "model.safetensors").exists()


def test_fetch_downloads_when_cache_empty(fake_cache, monkeypatch):
    """Cache miss → calls snapshot_download exactly once."""
    pkg_dir = fake_cache / "apex-memory" / "myelin-contrastive-v1"
    download_calls = {"n": 0}

    def _fake_download(repo_id, local_dir, **_kw):  # noqa: ANN001
        download_calls["n"] += 1
        _write_fake_pkg(Path(local_dir))
        return local_dir
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _fake_download,
        raising=False,
    )
    # First fetch to know the SHA (we'll write fake bytes and read them).
    monkeypatch.setattr(wf, "PINNED_SHA256", hashlib.sha256(b"fake-weights").hexdigest())
    out = fetch_weights("contrastive_v1")
    assert download_calls["n"] == 1
    assert Path(out).exists()


def test_fetch_raises_on_sha_mismatch(fake_cache, monkeypatch):
    """Tampered weights file → WeightsShaMismatchError."""
    pkg_dir = fake_cache / "apex-memory" / "myelin-contrastive-v1"
    _write_fake_pkg(pkg_dir, payload=b"tampered")
    # Pin a SHA that does NOT match the tampered payload.
    monkeypatch.setattr(wf, "PINNED_SHA256", "0" * 64)

    with pytest.raises(WeightsShaMismatchError):
        fetch_weights("contrastive_v1")


def test_fetch_respects_env_override(tmp_path, monkeypatch):
    """MYELIN_WEIGHTS_CACHE overrides ~/.cache/myelin/."""
    custom = tmp_path / "custom-cache"
    monkeypatch.setenv("MYELIN_WEIGHTS_CACHE", str(custom))
    pkg = custom / "apex-memory" / "myelin-contrastive-v1"
    sha = _write_fake_pkg(pkg)
    monkeypatch.setattr(wf, "PINNED_SHA256", sha)
    out = fetch_weights("contrastive_v1")
    assert str(custom) in str(out)


def test_fetch_raises_on_offline_no_cache(fake_cache, monkeypatch):
    """Offline + empty cache → WeightsFetchError, not silent stub."""
    def _network_error(*_a, **_kw):  # noqa: ANN001
        raise OSError("no internet")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _network_error,
        raising=False,
    )
    with pytest.raises(WeightsFetchError):
        fetch_weights("contrastive_v1")


def test_pinned_sha256_is_non_empty():
    """CI gate: the pinned SHA constant MUST be set on release."""
    # During development this is a known placeholder; we assert
    # it's a valid 64-char hex when populated.
    sha = wf.PINNED_SHA256
    if sha == "":
        pytest.skip("PINNED_SHA256 not yet populated (pre-training)")
    assert len(sha) == 64
    int(sha, 16)  # hex validity
