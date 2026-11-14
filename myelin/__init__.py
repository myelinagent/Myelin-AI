"""Myelin — drop-in memory layer for AI agent harnesses.

Ships three validated mechanisms in a 7-method API:
- Event-anchored bi-temporal facts (P1 H044 v2 — parity with APEX-MEM §3.1)
- Dream-synthesis consolidation (F-ML-2 — counterfactual self-test)
- Apoptosis-based forgetting (P3 — quarantine -> tombstone state machine)

Quickstart:

    from myelin import MemoryClient

    client = MemoryClient()
    client.add("Scott prefers tea.")
    client.search("What does Scott like?")

See README for full API + cross-references to the validated mechanisms.
"""
from myelin.client import MemoryClient
from myelin.context import estimate_tokens, excerpt, fit_to_budget, format_context
from myelin.prompts import build_answer_prompt, parse_used, strip_used
from myelin.types import (
    ApoptosisReport,
    ConsolidationReport,
    Event,
    Memory,
    ObservedTurn,
    SearchResult,
    SearchTelemetry,
    SessionHandle,
)

__all__ = [
    "ApoptosisReport",
    "ConsolidationReport",
    "Event",
    "Memory",
    "MemoryClient",
    "ObservedTurn",
    "SearchResult",
    "SearchTelemetry",
    "SessionHandle",
    "build_answer_prompt",
    "estimate_tokens",
    "excerpt",
    "fit_to_budget",
    "format_context",
    "parse_used",
    "strip_used",
]

__version__ = "1.4.0"
