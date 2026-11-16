"""The Myelin-AI memory tools over HTTP, for agents that run elsewhere.

    MYELIN_MCP_TOKEN=<long random string> myelin-mcp --http --host 0.0.0.0

Implements the MCP "Streamable HTTP" transport in its stateless form: a
client POSTs a JSON-RPC message to ``/mcp`` and gets the reply as JSON.
Standard library only.

Who may call it
---------------
Every request needs an access token, sent either as
``Authorization: Bearer <token>`` or, for clients that cannot set
headers, as the last part of the URL: ``/mcp/<token>``. A token names a
user, and each user sees only their own memories.

* One person: set ``MYELIN_MCP_TOKEN``; the memories belong to
  ``--user-id``.
* Several people: ``--tokens-file tokens.json`` holding
  ``{"<token>": "<user id>", ...}``.

Without any token the server starts only on a loopback address, for
local testing. It never starts open to the network.

This server does not encrypt traffic. Put it behind something that
terminates HTTPS (a reverse proxy or a tunnel) before exposing it.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from myelin import MemoryClient
from myelin.serve.mcp import MemoryServer, _error

ENDPOINT = "/mcp"
MAX_BODY_BYTES = 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 30
MIN_TOKEN_LENGTH = 16


class ConfigError(Exception):
    """The server was asked to start in a way that is not safe or valid."""


def load_tokens(tokens_file: str, user_id: str) -> dict[str, str]:
    """Access tokens and the user each one names."""
    tokens: dict[str, str] = {}
    single = os.environ.get("MYELIN_MCP_TOKEN", "").strip()
    if single:
        tokens[single] = user_id
    if tokens_file:
        try:
            loaded = json.loads(Path(tokens_file).expanduser().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read tokens file: {exc}") from exc
        if not isinstance(loaded, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in loaded.items()
        ):
            raise ConfigError('tokens file must be a JSON object of {"token": "user id"}')
        tokens.update(loaded)
    short = [user for token, user in tokens.items() if len(token) < MIN_TOKEN_LENGTH]
    if short:
        raise ConfigError(
            f"access token for {', '.join(sorted(short))} is shorter than "
            f"{MIN_TOKEN_LENGTH} characters; use a long random string"
        )
    return tokens


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class MemoryHTTPServer(ThreadingHTTPServer):
    """One shared store; one set of tools per user."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        client: MemoryClient,
        *,
        tokens: dict[str, str],
        default_user: str,
        allowed_origins: list[str] | None = None,
    ) -> None:
        if not tokens and not is_loopback(address[0]):
            raise ConfigError(
                "refusing to listen on a network address without an access token: "
                "set MYELIN_MCP_TOKEN or pass --tokens-file"
            )
        super().__init__(address, _Handler)
        self.client = client
        self.tokens = tokens
        self.default_user = default_user
        self.allowed_origins = set(allowed_origins or [])
        self._tools: dict[str, MemoryServer] = {}
        self._tools_lock = threading.Lock()

    def user_for(self, token: str | None) -> str | None:
        """The user a token names; None if it names no one."""
        if not self.tokens:
            return self.default_user  # loopback only, enforced above
        if not token:
            return None
        found = None
        for known, user in self.tokens.items():  # check all: constant time
            if hmac.compare_digest(known.encode(), token.encode()):
                found = user
        return found

    def tools_for(self, user: str) -> MemoryServer:
        with self._tools_lock:
            if user not in self._tools:
                self._tools[user] = MemoryServer(self.client, user_id=user)
            return self._tools[user]


class _Handler(BaseHTTPRequestHandler):
    server: MemoryHTTPServer
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT_SECONDS

    def log_message(self, format: str, *args: Any) -> None:
        # The default log line includes the request path, which can hold a
        # token. Log the method and status only.
        status = args[1] if len(args) > 1 else ""
        print(f"myelin-mcp http: {self.command} {status}", file=sys.stderr)

    # ---- replies ---------------------------------------------------------

    def _send(self, status: int, body: Any = None, headers: dict[str, str] | None = None) -> None:
        data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _refuse(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
        self.close_connection = True
        self._send(status, {"error": message}, headers)

    # ---- request ---------------------------------------------------------

    def _path_token(self) -> tuple[bool, str | None]:
        """Whether the path is the MCP endpoint, and a token carried in it."""
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == ENDPOINT:
            return True, None
        if path.startswith(ENDPOINT + "/") and "/" not in path[len(ENDPOINT) + 1:]:
            return True, path[len(ENDPOINT) + 1:]
        return False, None

    def _user(self) -> str | None:
        """Check the path, origin and token; reply and return None if refused."""
        is_endpoint, path_token = self._path_token()
        if not is_endpoint:
            self._refuse(404, "not found")
            return None
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.allowed_origins:
            self._refuse(403, "origin not allowed")
            return None
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else path_token
        user = self.server.user_for(token)
        if user is None:
            self._refuse(401, "missing or invalid access token", {"WWW-Authenticate": "Bearer"})
        return user

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._send(200, {"status": "ok"})
            return
        if self._user() is not None:
            # No server-initiated messages, so there is no stream to open.
            self._refuse(405, "use POST", {"Allow": "POST"})

    def do_DELETE(self) -> None:
        if self._user() is not None:
            self._refuse(405, "sessions are not used", {"Allow": "POST"})

    def do_POST(self) -> None:
        user = self._user()
        if user is None:
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._refuse(411, "Content-Length required")
            return
        if length > MAX_BODY_BYTES:
            self._refuse(413, "request too large")
            return
        try:
            message = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, _error(None, -32700, "parse error"))
            return
        tools = self.server.tools_for(user)
        batch = message if isinstance(message, list) else [message]
        replies = [
            reply for item in batch
            if isinstance(item, dict) and (reply := tools.handle(item)) is not None
        ]
        if not replies:
            self._send(202)  # only notifications: accepted, nothing to say
        elif isinstance(message, list):
            self._send(200, replies)
        else:
            self._send(200, replies[0])


def run(
    client: MemoryClient,
    *,
    host: str,
    port: int,
    user_id: str,
    tokens_file: str = "",
    allowed_origins: list[str] | None = None,
) -> int:
    """Serve until interrupted. Returns a process exit code."""
    try:
        tokens = load_tokens(tokens_file, user_id)
        server = MemoryHTTPServer(
            (host, port), client, tokens=tokens, default_user=user_id,
            allowed_origins=allowed_origins,
        )
    except ConfigError as exc:
        print(f"myelin-mcp: {exc}", file=sys.stderr)
        return 2
    who = f"{len(tokens)} access token(s)" if tokens else "NO access token (loopback only)"
    print(
        f"myelin-mcp: MCP over HTTP at http://{host}:{server.server_address[1]}{ENDPOINT} "
        f"with {who}. This server does not encrypt traffic; put HTTPS in front of it.",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
