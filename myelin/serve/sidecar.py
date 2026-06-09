"""HTTP sidecar that exposes a :class:`myelin.MemoryClient` as JSON-over-HTTP.

Promoted from ``scripts/benchmarks/personamem/myelin_sidecar.py`` (validated
on PersonaMem n=589 benches v1.6 / v1.7 / v1.9). Use ``python -m myelin.serve``
to run; use :func:`build_server` to embed in another process.

Endpoint contract is intentionally compatible with the OpenClaw memory-plugin
convention (see ADR-017) so the Node bridge plugin under
``tools/openclaw-myelin-bridge/`` can talk to either.

Loopback only by default. No auth on this layer — wrap behind a reverse
proxy with token auth if you expose it beyond the local host.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from myelin import MemoryClient

logger = logging.getLogger(__name__)


# ---- state ---------------------------------------------------------------

_CLIENT_LOCK = threading.Lock()
_CLIENT: MemoryClient | None = None
_DEFAULT_USER_ID = "myelin-sidecar-default"

# Tool-call counter for vacuous-PASS guards (preflight integration checks).
# Increments on every successful /tools/* and /hooks/* invocation.
_TOOL_CALL_COUNTER_LOCK = threading.Lock()
_TOOL_CALL_COUNTER: dict[str, int] = {"tool_invocations": 0, "reset_count": 0}


def _bump_counter(key: str = "tool_invocations") -> None:
    with _TOOL_CALL_COUNTER_LOCK:
        _TOOL_CALL_COUNTER[key] = _TOOL_CALL_COUNTER.get(key, 0) + 1


def _snapshot_counter() -> dict[str, int]:
    with _TOOL_CALL_COUNTER_LOCK:
        return dict(_TOOL_CALL_COUNTER)


def _get_client() -> MemoryClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = MemoryClient(db_path=":memory:", user_id=_DEFAULT_USER_ID)
        return _CLIENT


def _reset_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = MemoryClient(db_path=":memory:", user_id=_DEFAULT_USER_ID)


# ---- serializer ----------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


# ---- tool handlers (return JSON-safe dicts) ------------------------------


def tool_memory_search(payload: dict) -> Any:
    client = _get_client()
    hits = client.search(
        query=payload["query"],
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
        top_k=int(payload.get("top_k", 5)),
    )
    return {"hits": _to_jsonable(hits)}


def tool_memory_get(payload: dict) -> Any:
    client = _get_client()
    mem = client.get(payload["memory_id"])
    return {"memory": _to_jsonable(mem)}


def tool_memory_store(payload: dict) -> Any:
    client = _get_client()
    mem = client.add(
        content=payload["content"],
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
        metadata=payload.get("metadata") or {},
        confidence=float(payload.get("confidence", 0.5)),
        importance=float(payload.get("importance", 0.5)),
    )
    return {"memory": _to_jsonable(mem)}


def tool_memory_update(payload: dict) -> Any:
    client = _get_client()
    mem = client.update(
        memory_id=payload["memory_id"],
        content=payload.get("content"),
        metadata=payload.get("metadata"),
    )
    return {"memory": _to_jsonable(mem)}


def tool_memory_forget(payload: dict) -> Any:
    client = _get_client()
    deleted = client.delete(payload["memory_id"])
    return {"deleted": bool(deleted)}


def tool_memory_consolidate(payload: dict) -> Any:
    client = _get_client()
    report = client.consolidate(
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
        max_samples=int(payload.get("max_samples", 32)),
    )
    return {"report": _to_jsonable(report)}


def tool_memory_prune(payload: dict) -> Any:
    client = _get_client()
    report = client.prune(
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
    )
    return {"report": _to_jsonable(report)}


# ---- lifecycle hooks -----------------------------------------------------


def hook_before_agent_start(payload: dict) -> Any:
    """Recall + inject as system context.

    Payload: {"user_id": str, "query": str, "top_k"?: int}
    Returns: {"messages": [{"role":"system","content": "..."}]}
    """
    query = payload.get("query") or ""
    if not query:
        return {"messages": []}
    client = _get_client()
    hits = client.search(
        query=query,
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
        top_k=int(payload.get("top_k", 5)),
    )
    if not hits:
        return {"messages": []}

    def _hit_text(h: Any) -> str:
        mem = getattr(h, "memory", None) or h
        return getattr(mem, "content", str(mem))

    bullets = "\n".join(f"- {_hit_text(h)}" for h in hits)
    content = f"Relevant memories from this user's history:\n{bullets}"
    return {"messages": [{"role": "system", "content": content}]}


def hook_agent_end(payload: dict) -> Any:
    """Capture turn (memory_store).

    Payload: {"user_id": str, "turn": {"content": str, ...}}
    """
    turn = payload.get("turn") or {}
    content = turn.get("content") or ""
    if not content:
        return {"stored": None, "reason": "empty turn content"}
    client = _get_client()
    mem = client.add(
        content=content,
        user_id=payload.get("user_id") or _DEFAULT_USER_ID,
        metadata=turn.get("metadata") or {},
    )
    return {"stored": _to_jsonable(mem)}


# ---- route table ---------------------------------------------------------


_ROUTES = {
    "/tools/memory_search": tool_memory_search,
    "/tools/memory_get": tool_memory_get,
    "/tools/memory_store": tool_memory_store,
    "/tools/memory_update": tool_memory_update,
    "/tools/memory_forget": tool_memory_forget,
    "/tools/memory_consolidate": tool_memory_consolidate,
    "/tools/memory_prune": tool_memory_prune,
    "/hooks/before_agent_start": hook_before_agent_start,
    "/hooks/agent_end": hook_agent_end,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)

    def _write_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"ok": True, "service": "myelin-sidecar"})
            return
        if self.path == "/metrics":
            self._write_json(200, _snapshot_counter())
            return
        self._write_json(404, {"error": f"unknown route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            payload = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": f"bad json: {exc}"})
            return

        if self.path == "/admin/reset":
            _reset_client()
            _bump_counter("reset_count")
            self._write_json(200, {"ok": True, "reset": True})
            return

        handler = _ROUTES.get(self.path)
        if not handler:
            self._write_json(404, {"error": f"unknown route {self.path}"})
            return

        try:
            result = handler(payload)
            _bump_counter("tool_invocations")
            self._write_json(200, result)
        except KeyError as exc:
            self._write_json(400, {"error": f"missing field: {exc}"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("handler error on %s", self.path)
            self._write_json(500, {"error": type(exc).__name__, "message": str(exc)})


def build_server(host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    """Build (but do not start) an HTTPServer wrapping the myelin sidecar.

    Caller is responsible for ``server.serve_forever()`` and
    ``server.server_close()``. Useful for embedding in another process or
    running on a background thread for tests.
    """
    _get_client()  # prime the singleton
    return HTTPServer((host, port), _Handler)


def main() -> None:
    """CLI entrypoint — ``python -m myelin.serve``."""
    ap = argparse.ArgumentParser(prog="python -m myelin.serve")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default loopback)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    server = build_server(args.host, args.port)
    logger.info("myelin-sidecar listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
