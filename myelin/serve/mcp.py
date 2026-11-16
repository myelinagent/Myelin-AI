"""MCP server: Myelin-AI as long-term memory for an assistant such as Claude Code.

    myelin-mcp                         # memories in ~/.myelin/memory.db
    myelin-mcp --db-path ./memory.db   # one store per project

Speaks the Model Context Protocol over stdio (one JSON-RPC message per
line), using only the standard library. Five tools:

* ``remember``   store something worth keeping
* ``recall``     the best-matching memories that fit a token budget
* ``mark_used``  say which recalled memories the answer relied on
* ``forget``     delete one memory
* ``memory_stats`` how much is stored

``mark_used`` is what lets the store learn: memories that keep being
recalled and not used rank lower next time.

Hybrid search is on when the ``static`` extra is installed
(``pip install "myelin-ai[static]"``); otherwise search is lexical.
Nothing is pruned automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any

from myelin import MemoryClient, __version__, format_context
from myelin.types import SearchResult

DEFAULT_DB_PATH = "~/.myelin/memory.db"
DEFAULT_USER_ID = "default"
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# How many recalls are kept so that ``mark_used`` can refer back to one.
RECALLS_KEPT = 32

INSTRUCTIONS = (
    "Long-term memory that persists between sessions. "
    "Call `recall` at the start of a task, and whenever earlier context, "
    "preferences or decisions might matter. "
    "Call `remember` when the user states a lasting preference, fact or "
    "decision, or when you learn something a future session would need; "
    "store one self-contained statement per call. "
    "After answering with the help of recalled memories, call `mark_used` "
    "with the numbers of the ones you relied on (an empty list if none), "
    "so that useful memories rank higher next time. "
    "Do not store secrets such as passwords or API keys."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Store one lasting fact, preference or decision so that future "
            "sessions can recall it. Write a self-contained sentence that "
            "names who or what it is about. Never store secrets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember."},
                "importance": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "0 to 1; default 0.5. Higher is kept longer.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Find stored memories relevant to a question or topic. Returns "
            "numbered memories, oldest first, and a recall_id for mark_used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to know."},
                "token_budget": {
                    "type": "integer", "minimum": 100,
                    "description": "Most tokens of memory to return; default 2000.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "mark_used",
        "description": (
            "Report which memories from a recall your answer relied on, by "
            "their numbers. Pass an empty list if none were useful."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "recall_id": {"type": "string"},
                "used": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            },
            "required": ["recall_id", "used"],
        },
    },
    {
        "name": "forget",
        "description": "Delete one memory by its id (ids are listed by recall).",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_stats",
        "description": "How many memories are stored, where, and how search works.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class ToolError(Exception):
    """A tool call that cannot be carried out; reported to the model."""


class MemoryServer:
    """The tools, over one :class:`MemoryClient`."""

    def __init__(self, client: MemoryClient, *, db_path: str = ":memory:") -> None:
        self.client = client
        self.db_path = db_path
        self._recalls: OrderedDict[str, tuple[str, list[SearchResult]]] = OrderedDict()
        self._recall_count = 0

    # ---- tools -----------------------------------------------------------

    def remember(self, args: dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            raise ToolError("content is empty")
        importance = float(args.get("importance", 0.5))
        if not 0.0 <= importance <= 1.0:
            raise ToolError("importance must be between 0 and 1")
        memory = self.client.add(content, importance=importance)
        return f"Remembered (id {memory.id})."

    def recall(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query is empty")
        budget = max(100, int(args.get("token_budget", 2000)))
        hits = self.client.recall(query, token_budget=budget)
        if not hits:
            return "No memories match."
        self._recall_count += 1
        recall_id = f"r{self._recall_count}"
        self._recalls[recall_id] = (query, hits)
        while len(self._recalls) > RECALLS_KEPT:
            self._recalls.popitem(last=False)
        ids = "\n".join(f"[{n}] {hit.memory.id}" for n, hit in enumerate(hits, start=1))
        return (
            f"{format_context(hits, numbered=True)}\n\n"
            f"Memory ids:\n{ids}\n\n"
            f"recall_id: {recall_id}. After answering, call mark_used with "
            "this recall_id and the numbers you relied on."
        )

    def mark_used(self, args: dict[str, Any]) -> str:
        recall_id = str(args.get("recall_id") or "")
        if recall_id not in self._recalls:
            raise ToolError(f"unknown recall_id {recall_id!r}; call recall again")
        query, hits = self._recalls[recall_id]
        used = sorted({int(n) for n in args.get("used") or []})
        if any(n < 1 or n > len(hits) for n in used):
            raise ToolError(f"used must be numbers from 1 to {len(hits)}")
        counts = self.client.cite_used(hits, used, query=query)
        return f"Recorded: {counts['used']} used, {counts['unused']} not used."

    def forget(self, args: dict[str, Any]) -> str:
        memory_id = str(args.get("memory_id") or "")
        if not memory_id:
            raise ToolError("memory_id is empty")
        if not self.client.delete(memory_id):
            raise ToolError(f"no memory with id {memory_id!r}")
        return "Forgotten."

    def memory_stats(self, args: dict[str, Any]) -> str:
        total = self.client.count()
        search = "hybrid (words and meaning)" if self.client.hybrid_search else "lexical (words)"
        return f"{total} memories in {self.db_path}. Search: {search}."

    # ---- protocol --------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one JSON-RPC message; ``None`` for a notification."""
        method = message.get("method")
        if "id" not in message:
            return None
        request_id = message["id"]
        params = message.get("params") or {}
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                result: dict[str, Any] = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "myelin-ai", "version": __version__},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(params)
            else:
                return _error(request_id, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001 - a broken request must not stop the server
            return _error(request_id, -32603, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = {t["name"] for t in TOOLS}
        if name not in tool:
            return _text(f"unknown tool: {name}", error=True)
        try:
            return _text(getattr(self, name)(params.get("arguments") or {}))
        except ToolError as exc:
            return _text(str(exc), error=True)
        except (TypeError, ValueError) as exc:
            return _text(f"bad arguments: {exc}", error=True)


def _text(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(server: MemoryServer, stdin: IO[str], stdout: IO[str]) -> None:
    """Read messages from ``stdin`` until it closes, answering on ``stdout``."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            replies: list[dict[str, Any]] = [_error(None, -32700, "parse error")]
        else:
            batch = message if isinstance(message, list) else [message]
            replies = [
                reply for item in batch
                if isinstance(item, dict) and (reply := server.handle(item)) is not None
            ]
        for reply in replies:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


def _embedder() -> Any:
    """The bundled CPU embedder when its extra is installed, else None."""
    try:
        from myelin.embedders import StaticEmbedder

        return StaticEmbedder()
    except Exception as exc:  # noqa: BLE001 - any failure to load means lexical search
        print(f"myelin-mcp: lexical search only ({exc})", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="myelin-mcp", description="Myelin-AI memory as an MCP server over stdio."
    )
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"SQLite file holding the memories (default {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--user-id", default=DEFAULT_USER_ID,
        help="whose memories these are; separate ids do not see each other",
    )
    parser.add_argument(
        "--no-embedder", action="store_true",
        help="lexical search only, even if the static extra is installed",
    )
    args = parser.parse_args(argv)

    # Only protocol messages may reach stdout; anything a library prints
    # goes to stderr.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    db_path = args.db_path
    if db_path != ":memory:":
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        db_path = str(path)
    client = MemoryClient(
        db_path=db_path,
        user_id=args.user_id,
        embedder=None if args.no_embedder else _embedder(),
        cite_learning=True,
    )
    try:
        serve(MemoryServer(client, db_path=db_path), sys.stdin, protocol_out)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
