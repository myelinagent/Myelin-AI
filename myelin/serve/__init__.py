"""HTTP serving surface for myelin.

Wraps :class:`myelin.MemoryClient` as a JSON-over-HTTP service so non-Python
callers (or sidecar-style architectures like the Node OpenClaw bridge) can
drive memory operations remotely.

Run as a module::

    python -m myelin.serve --host 127.0.0.1 --port 8765

Endpoints mirror the OpenClaw memory-plugin convention (see ADR-017):

  GET  /health                          — service healthcheck
  POST /admin/reset                     — clear + recreate MemoryClient
  POST /hooks/before_agent_start        — recall + return as system context
  POST /hooks/agent_end                 — capture turn (memory_store)
  POST /tools/memory_search             — semantic search
  POST /tools/memory_get                — get by id
  POST /tools/memory_store              — add fact
  POST /tools/memory_update             — in-place edit
  POST /tools/memory_forget             — hard delete
  POST /tools/memory_consolidate        — F-ML-2 dream-synthesis
  POST /tools/memory_prune              — P3 apoptosis

Security: loopback-only by default. No auth. If exposed beyond the local
host, wrap behind a reverse proxy with token auth.
"""
from myelin.serve.sidecar import build_server, main

__all__ = ["build_server", "main"]
