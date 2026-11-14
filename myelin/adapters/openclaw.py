"""OpenClaw memory-plugin adapter around `myelin.client.MemoryClient`.

This module wraps the standalone Phase-A `MemoryClient` in the OpenClaw
memory-plugin protocol so a Myelin substrate can be dropped into an
OpenClaw harness in the same slot TencentDB Agent Memory / Mem0 /
MemOS occupy.

The OpenClaw memory-plugin protocol is convention-driven (see ADR-017):

- The plugin claims the exclusive ``plugins.slots.memory`` slot.
- It implements two lifecycle hooks: ``before_agent_start`` (recall +
  inject system context) and ``agent_end`` (capture the turn).
- It exposes agent-callable tools, each with a JSON-Schema input spec,
  so an LLM agent can invoke them inside its reasoning loop.

Myelin's novel ``consolidate()`` (F-ML-2 dream-synthesis) and ``prune()``
(P3 apoptosis) mechanisms are exposed as additional tools — that's the
differentiating advantage over Mem0 / TencentDB plugins which expose
only CRUD.

Usage::

    from myelin import MemoryClient
    from myelin.adapters.openclaw import OpenClawMemoryPlugin

    client = MemoryClient(db_path=":memory:")
    plugin = OpenClawMemoryPlugin(client=client, name="myelin")

    # Inside an OpenClaw harness:
    plugin.before_agent_start({"user_id": "u1", "query": "what does Scott like?"})
    # ...agent runs, produces a turn...
    plugin.agent_end({"user_id": "u1", "turn": {"content": "Scott prefers tea."}})

    # Tools are discoverable via .tools — JSON-Schema-typed agent actions.
    for tool in plugin.tools:
        print(tool["name"], tool["input_schema"])
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Callable

from myelin.client import MemoryClient
from myelin.types import (
    ApoptosisReport,
    ConsolidationReport,
    Event,
    Memory,
    SearchResult,
)

__all__ = [
    "OpenClawMemoryPlugin",
    "OPENCLAW_TOOL_NAMES",
    "OPENCLAW_LIFECYCLE_HOOKS",
]


# Tool names — kept stable so harness configs / tests can pin them.
OPENCLAW_TOOL_NAMES: tuple[str, ...] = (
    "memory_search",  # core: semantic recall
    "memory_get",  # core: targeted read
    "memory_store",  # mem0-extension: write
    "memory_update",  # CRUD completion
    "memory_forget",  # mem0-extension: delete
    "memory_consolidate",  # Myelin novel: F-ML-2 dream-synthesis
    "memory_prune",  # Myelin novel: P3 apoptosis
)

# Lifecycle hooks — mirrors MemOS-Cloud-OpenClaw-Plugin convention.
OPENCLAW_LIFECYCLE_HOOKS: tuple[str, ...] = (
    "before_agent_start",
    "agent_end",
)

# Slot the plugin claims.
_MEMORY_SLOT = "memory"


class OpenClawMemoryPlugin:
    """OpenClaw memory plugin wrapping a `MemoryClient`.

    Construct once per process. Safe to share across requests — the
    underlying ``MemoryClient`` is thread-safe via its SqliteBackend
    lock.

    Args:
        client: A constructed `MemoryClient` (caller owns it).
        name: Plugin name surfaced in the manifest; defaults to
            ``"myelin"``. Must be ASCII-safe (only ``[A-Za-z0-9_-]``).
        recall_top_k: Default `top_k` used by the auto-recall
            ``before_agent_start`` hook.
        auto_capture: When True, ``agent_end`` writes the durable
            content from the turn into memory. When False, the hook is
            a no-op (the agent itself can still invoke ``memory_store``
            as a tool).
    """

    def __init__(
        self,
        *,
        client: MemoryClient,
        name: str = "myelin",
        recall_top_k: int = 5,
        auto_capture: bool = True,
    ) -> None:
        if not isinstance(client, MemoryClient):
            raise TypeError(
                f"client must be MemoryClient, got {type(client).__name__}"
            )
        if not name or not all(
            c.isalnum() or c in ("-", "_") for c in name
        ):
            raise ValueError(
                f"name must be non-empty and ASCII-safe ([A-Za-z0-9_-]); "
                f"got {name!r}"
            )
        if recall_top_k <= 0:
            raise ValueError(f"recall_top_k must be > 0, got {recall_top_k}")

        self._client = client
        self._name = name
        self._recall_top_k = recall_top_k
        self._auto_capture = auto_capture

        # Build tool table once. Each entry is a dict of:
        #   name, description, input_schema (JSON-Schema), handler (callable)
        self._tools: list[dict[str, Any]] = self._build_tools()
        self._tool_by_name: dict[str, dict[str, Any]] = {
            t["name"]: t for t in self._tools
        }

    # ---- Public surface: protocol identity ----------------------

    @property
    def name(self) -> str:
        """The plugin's slot-registration name."""
        return self._name

    @property
    def slot(self) -> str:
        """The OpenClaw plugin slot — always ``"memory"`` for this adapter."""
        return _MEMORY_SLOT

    @property
    def tools(self) -> list[dict[str, Any]]:
        """List of agent-callable tool specs.

        Each entry is shaped::

            {
                "name": str,
                "description": str,
                "input_schema": {... JSON Schema Draft 7 ...},
            }

        The ``handler`` callable is intentionally **not** included in
        the returned dicts so the list is JSON-serializable for
        introspection / harness config emission. Use ``invoke_tool()`` to
        actually call a tool.
        """
        # Return a deep-ish copy without the handler so this is safe
        # for ``json.dumps`` (one of the hard test requirements).
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self._tools
        ]

    def manifest(self) -> dict[str, Any]:
        """The ``openclaw.plugin.json`` manifest contents.

        Returns a JSON-serializable dict the harness can write to
        ``openclaw.plugin.json`` (or merge into a larger plugin
        registration document). Shape follows the convergent contract
        documented in ADR-017.
        """
        return {
            "name": self._name,
            "version": "0.1.0",
            "slot": self.slot,
            "description": (
                "Myelin memory plugin — event-anchored bi-temporal facts, "
                "dream-synthesis consolidation, apoptosis-based forgetting."
            ),
            "tools": [t["name"] for t in self._tools],
            "hooks": list(OPENCLAW_LIFECYCLE_HOOKS),
            "config_schema": {
                "type": "object",
                "properties": {
                    "recall_top_k": {"type": "integer", "minimum": 1},
                    "auto_capture": {"type": "boolean"},
                },
                "required": [],
            },
        }

    # ---- Public surface: tool invocation ------------------------

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered tool by name with JSON-shaped arguments.

        This is the entry point an OpenClaw harness uses when the agent
        emits a tool-call. The returned value is JSON-serializable so
        the harness can ship it straight back to the LLM.

        Raises:
            KeyError: If ``name`` is not a registered tool.
            ValueError: If required arguments are missing.
        """
        if name not in self._tool_by_name:
            raise KeyError(
                f"unknown tool {name!r}; registered: "
                f"{sorted(self._tool_by_name)}"
            )
        if not isinstance(arguments, dict):
            raise TypeError(
                f"arguments must be dict, got {type(arguments).__name__}"
            )

        handler: Callable[[dict[str, Any]], Any] = self._tool_by_name[name][
            "_handler"
        ]
        result = handler(arguments)
        return _to_json_safe(result)

    # ---- Public surface: lifecycle hooks ------------------------

    def before_agent_start(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Recall hook — called before the agent's reasoning turn.

        Reads the user query from ``context["query"]`` (or
        ``context["user_message"]``), searches the substrate, and
        returns a payload the harness prepends to the agent's system
        context.

        Returns a dict with both shapes the loaders we know expect:

        - ``messages`` — list of ``{role, content}`` (MemOS-Cloud shape)
        - ``system_context`` — a flat string (basic-memory / mem0 shape)
        - ``hits`` — the raw `SearchResult` list as dicts (for debugging)

        Empty payload when ``query`` is missing or no hits.
        """
        if not isinstance(context, dict):
            raise TypeError(
                f"context must be dict, got {type(context).__name__}"
            )

        query = context.get("query") or context.get("user_message") or ""
        if not query:
            return {"messages": [], "system_context": "", "hits": []}

        user_id = context.get("user_id")
        top_k = int(context.get("top_k", self._recall_top_k))
        hits = self._client.search(
            query=query, user_id=user_id, top_k=top_k
        )
        if not hits:
            return {"messages": [], "system_context": "", "hits": []}

        # Render a compact system-context block.
        lines = [
            f"- {h.memory.content}  (score={h.score:.3f})" for h in hits
        ]
        block = "Relevant memories:\n" + "\n".join(lines)

        return {
            "messages": [{"role": "system", "content": block}],
            "system_context": block,
            "hits": [_to_json_safe(h) for h in hits],
        }

    def agent_end(self, turn: dict[str, Any]) -> dict[str, Any]:
        """Capture hook — called after the agent's turn ends.

        Reads the durable content from
        ``turn["content"]`` (or ``turn["assistant_message"]``) and
        writes it via ``MemoryClient.add``. When ``auto_capture=False``,
        this is a no-op (returns ``{"stored": False, "memory_id": None}``).

        Returns ``{"stored": bool, "memory_id": str|None}``.
        """
        if not isinstance(turn, dict):
            raise TypeError(
                f"turn must be dict, got {type(turn).__name__}"
            )

        if not self._auto_capture:
            return {"stored": False, "memory_id": None}

        content = turn.get("content") or turn.get("assistant_message") or ""
        if not content or not content.strip():
            return {"stored": False, "memory_id": None}

        mem = self._client.add(
            content,
            user_id=turn.get("user_id"),
            metadata=turn.get("metadata"),
        )
        return {"stored": True, "memory_id": mem.id}

    # ---- Internal: tool table -----------------------------------

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": (
                    "Semantic recall over the agent's long-term memory. "
                    "Returns the top_k most relevant facts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5,
                        },
                        "user_id": {
                            "type": "string",
                            "description": (
                                "Optional scope override. Defaults to the "
                                "client's bound user_id."
                            ),
                        },
                        "as_of": {
                            "type": "string",
                            "format": "date-time",
                            "description": (
                                "ISO-8601 timestamp; bi-temporal filter "
                                "returning only facts valid at that time."
                            ),
                        },
                    },
                    "required": ["query"],
                },
                "_handler": self._tool_search,
            },
            {
                "name": "memory_get",
                "description": (
                    "Fetch a single memory by id. Returns null if "
                    "absent or tombstoned."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                    },
                    "required": ["memory_id"],
                },
                "_handler": self._tool_get,
            },
            {
                "name": "memory_store",
                "description": (
                    "Persist a new durable fact. Optionally event-anchor "
                    "the fact for bi-temporal provenance."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "user_id": {"type": "string"},
                        "metadata": {"type": "object"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                        "event": {
                            "type": "object",
                            "description": (
                                "Optional event-anchor for bi-temporal "
                                "provenance (P1 H044 v2)."
                            ),
                            "properties": {
                                "event_id": {"type": "string"},
                                "type": {"type": "string"},
                                "anchor_datetime": {
                                    "type": "string",
                                    "format": "date-time",
                                },
                                "location": {"type": "string"},
                                "participant_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence_turn_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["event_id"],
                        },
                    },
                    "required": ["content"],
                },
                "_handler": self._tool_store,
            },
            {
                "name": "memory_update",
                "description": "Edit an existing memory's content / metadata in-place.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["memory_id"],
                },
                "_handler": self._tool_update,
            },
            {
                "name": "memory_forget",
                "description": (
                    "Hard-delete a memory by id. For audit-preserving "
                    "forgetting use memory_prune instead."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                    },
                    "required": ["memory_id"],
                },
                "_handler": self._tool_forget,
            },
            {
                "name": "memory_consolidate",
                "description": (
                    "Run dream-synthesis self-test (F-ML-2). Samples "
                    "healthy facts, generates dream queries, grades the "
                    "agent's recall, surfaces gaps."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "max_samples": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1024,
                            "default": 32,
                        },
                    },
                    "required": [],
                },
                "_handler": self._tool_consolidate,
            },
            {
                "name": "memory_prune",
                "description": (
                    "Forget on purpose: retire outdated facts and "
                    "quarantine unhealthy ones (hidden from search, kept "
                    "in the store)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "now": {
                            "type": "string",
                            "format": "date-time",
                            "description": (
                                "Optional ISO-8601 'now'; defaults to UTC "
                                "now. Useful for deterministic tests."
                            ),
                        },
                        "caspase_window_days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 365,
                            "default": 7,
                        },
                        "tombstone": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Also remove facts that have sat in "
                                "quarantine past the window. Off by "
                                "default: quarantined facts are kept."
                            ),
                        },
                    },
                    "required": [],
                },
                "_handler": self._tool_prune,
            },
        ]

    # ---- Internal: tool handlers --------------------------------

    def _tool_search(self, args: dict[str, Any]) -> list[Any]:
        query = _require_str(args, "query")
        as_of_raw = args.get("as_of")
        as_of: datetime | None = (
            _parse_iso(as_of_raw) if as_of_raw else None
        )
        return self._client.search(
            query=query,
            user_id=args.get("user_id"),
            top_k=int(args.get("top_k", self._recall_top_k)),
            as_of=as_of,
        )

    def _tool_get(self, args: dict[str, Any]) -> Memory | None:
        return self._client.get(_require_str(args, "memory_id"))

    def _tool_store(self, args: dict[str, Any]) -> Memory:
        content = _require_str(args, "content")
        return self._client.add(
            content,
            user_id=args.get("user_id"),
            metadata=args.get("metadata"),
            event=args.get("event"),
            confidence=float(args.get("confidence", 0.5)),
            importance=float(args.get("importance", 0.5)),
        )

    def _tool_update(self, args: dict[str, Any]) -> Memory:
        return self._client.update(
            _require_str(args, "memory_id"),
            content=args.get("content"),
            metadata=args.get("metadata"),
        )

    def _tool_forget(self, args: dict[str, Any]) -> dict[str, bool]:
        return {
            "deleted": self._client.delete(
                _require_str(args, "memory_id")
            )
        }

    def _tool_consolidate(
        self, args: dict[str, Any]
    ) -> ConsolidationReport:
        return self._client.consolidate(
            user_id=args.get("user_id"),
            max_samples=int(args.get("max_samples", 32)),
        )

    def _tool_prune(self, args: dict[str, Any]) -> ApoptosisReport:
        now_raw = args.get("now")
        now: datetime | None = (
            _parse_iso(now_raw) if now_raw else None
        )
        return self._client.prune(
            user_id=args.get("user_id"),
            now=now,
            caspase_window_days=int(args.get("caspase_window_days", 7)),
            tombstone=bool(args.get("tombstone", False)),
        )


# ---- Module helpers ---------------------------------------------


def _require_str(d: dict[str, Any], key: str) -> str:
    v = d.get(key)
    if v is None:
        raise ValueError(f"missing required argument {key!r}")
    if not isinstance(v, str):
        raise TypeError(
            f"argument {key!r} must be str, got {type(v).__name__}"
        )
    return v


def _parse_iso(s: str) -> datetime:
    if not isinstance(s, str):
        raise TypeError(
            f"datetime string expected; got {type(s).__name__}"
        )
    # datetime.fromisoformat handles "Z" suffix as of 3.11+.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _to_json_safe(obj: Any) -> Any:
    """Convert dataclasses / datetimes / Events into JSON-safe shapes.

    Used as the boundary serializer between the adapter and the
    OpenClaw harness so tool-call results are immediately
    ``json.dumps``-able.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, SearchResult):
        return {
            "memory": _to_json_safe(obj.memory),
            "score": float(obj.score),
        }
    if isinstance(obj, Memory):
        return {
            "id": obj.id,
            "content": obj.content,
            "user_id": obj.user_id,
            "t_valid": _to_json_safe(obj.t_valid),
            "t_invalid": _to_json_safe(obj.t_invalid),
            "metadata": dict(obj.metadata or {}),
            "event": _to_json_safe(obj.event) if obj.event else None,
            "confidence": float(obj.confidence),
            "importance": float(obj.importance),
            "health_state": obj.health_state,
            "quarantine_entered_at": _to_json_safe(
                obj.quarantine_entered_at
            ),
        }
    if isinstance(obj, Event):
        return {
            "event_id": obj.event_id,
            "type": obj.type,
            "anchor_datetime": _to_json_safe(obj.anchor_datetime),
            "location": obj.location,
            "participant_ids": list(obj.participant_ids),
            "evidence_turn_ids": list(obj.evidence_turn_ids),
        }
    if is_dataclass(obj):
        return _to_json_safe(asdict(obj))
    # Last resort: rely on json's behavior.
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        return repr(obj)
    return obj
