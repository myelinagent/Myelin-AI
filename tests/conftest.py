"""Shared pytest fixtures for v1.2 tests.

The SessionRegistry is a process-singleton with a daemon sweeper.
Tests that touch working memory MUST reset the singleton between
runs so they don't observe state leaked from a prior test. This
conftest auto-resets via an autouse fixture.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_session_registry():
    """Reset the SessionRegistry singleton before and after each test
    so working-memory tests don't observe state from prior runs."""
    from myelin._internal.session import SessionRegistry
    SessionRegistry.reset_for_testing()
    yield
    SessionRegistry.reset_for_testing()
