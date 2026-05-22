"""Drop-in adapters for popular agent-memory libraries.

Each adapter wraps the Phase-A :class:`myelin.MemoryClient` and exposes
the target library's surface, so existing code that imports e.g.
``mem0.MemoryClient`` can switch to ``myelin.adapters.mem0.MemoryClient``
with a single line change and pick up the event-anchored / dream-synthesis
/ apoptosis substrate without further refactoring.

Adapters available:

* :mod:`myelin.adapters.mem0` — Mem0 (`mem0ai`) compatible surface.
"""
