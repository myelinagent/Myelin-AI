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

Where a memory came from
------------------------
Several agents can share one user's memory. Each memory records the
agent that stored it, and ``recall`` shows it. The label comes from, in
order: the token's entry in the tokens file, written as
``{"<token>": {"user": "<user id>", "source": "<agent>"}}`` (one token
per agent, all naming the same user); a ``?source=<agent>`` on the URL;
an ``X-Myelin-Source`` header.

OAuth, one identity per bot
---------------------------
With ``--bots-file`` the server is also its own small OAuth
authorization server. Each bot trades its client id and secret for a
short-lived token at ``POST /oauth/token`` (client credentials grant)
and sends that as the bearer token. The bot's name is the source label
on what it stores, and a bot may be limited to reading. Clients find
the token endpoint through ``/.well-known/oauth-protected-resource``,
which a 401 points to. See :mod:`myelin.serve.mcp_oauth`.

Without any token or bot the server starts only on a loopback address,
for local testing. It never starts open to the network.

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
from urllib.parse import parse_qs

from myelin import MemoryClient
from myelin.serve import mcp_oauth
from myelin.serve.mcp import MemoryServer, _error, clean_source

ENDPOINT = "/mcp"
MAX_BODY_BYTES = 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 30
MIN_TOKEN_LENGTH = 16


class ConfigError(Exception):
    """The server was asked to start in a way that is not safe or valid."""


Grant = tuple[str, str | None]  # (user id, source label or None)
Caller = tuple[str, str | None, frozenset[str]]  # a grant, plus what it may do
FULL_ACCESS = frozenset(mcp_oauth.ALL_SCOPES)


def _only_allowed_tools(reply: dict[str, Any], scopes: frozenset[str]) -> dict[str, Any]:
    """A tools/list reply without the tools these scopes cannot call."""
    result = reply.get("result")
    if scopes >= FULL_ACCESS or not isinstance(result, dict) or "tools" not in result:
        return reply
    allowed = [t for t in result["tools"] if mcp_oauth.scope_for(t["name"]) in scopes]
    return {**reply, "result": {**result, "tools": allowed}}


def load_tokens(tokens_file: str, user_id: str) -> dict[str, Grant]:
    """Access tokens, with the user and optional source each one names."""
    tokens: dict[str, Grant] = {}
    single = os.environ.get("MYELIN_MCP_TOKEN", "").strip()
    if single:
        tokens[single] = (user_id, None)
    if tokens_file:
        try:
            loaded = json.loads(Path(tokens_file).expanduser().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read tokens file: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError('tokens file must be a JSON object of {"token": "user id"}')
        for token, entry in loaded.items():
            if isinstance(entry, str) and entry:
                tokens[token] = (entry, None)
            elif isinstance(entry, dict) and isinstance(entry.get("user"), str) and entry["user"]:
                tokens[token] = (entry["user"], clean_source(entry.get("source")))
            else:
                raise ConfigError(
                    'each tokens file entry must be "user id" or '
                    '{"user": "user id", "source": "agent name"}'
                )
    short = [user for token, (user, _) in tokens.items() if len(token) < MIN_TOKEN_LENGTH]
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
        tokens: dict[str, Grant],
        default_user: str,
        allowed_origins: list[str] | None = None,
        oauth: mcp_oauth.Tokens | None = None,
        public_url: str = "",
    ) -> None:
        if not tokens and oauth is None and not is_loopback(address[0]):
            raise ConfigError(
                "refusing to listen on a network address without an access token: "
                "set MYELIN_MCP_TOKEN or pass --tokens-file or --bots-file"
            )
        super().__init__(address, _Handler)
        self.client = client
        self.tokens = tokens
        self.oauth = oauth
        self.public_url = public_url.rstrip("/")
        self.default_user = default_user
        self.allowed_origins = set(allowed_origins or [])
        self._tools: dict[Grant, MemoryServer] = {}
        self._tools_lock = threading.Lock()

    def grant_for(self, token: str | None) -> Grant | None:
        """The user and source a token names; None if it names no one."""
        if not self.tokens and self.oauth is None:
            return (self.default_user, None)  # loopback only, enforced above
        if not token:
            return None
        found = None
        for known, grant in self.tokens.items():  # check all: constant time
            if hmac.compare_digest(known.encode(), token.encode()):
                found = grant
        return found

    def tools_for(self, user: str, source: str | None) -> MemoryServer:
        with self._tools_lock:
            if len(self._tools) > 1024:  # labels come from callers; keep this bounded
                self._tools.clear()
            key = (user, source)
            if key not in self._tools:
                self._tools[key] = MemoryServer(
                    self.client, user_id=user, source=source, adopt_client_name=False
                )
            return self._tools[key]


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
        if self.close_connection:
            self.send_header("Connection", "close")  # tell the client, do not just hang up
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

    def _base_url(self) -> str:
        """This server's address as callers see it."""
        if self.server.public_url:
            return self.server.public_url
        host = self.headers.get("Host") or "{}:{}".format(*self.server.server_address[:2])
        return f"http://{host}"

    def _challenge(self, problem: str = "") -> dict[str, str]:
        """The WWW-Authenticate header; with OAuth on, it says where to get a token."""
        parts = []
        if self.server.oauth is not None:
            parts.append(
                f'resource_metadata="{self._base_url()}{mcp_oauth.RESOURCE_METADATA_PATH}"'
            )
        if problem:
            parts.append(problem)
        return {"WWW-Authenticate": ("Bearer " + ", ".join(parts)).strip()}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.allowed_origins:
            self._refuse(403, "origin not allowed")
            return False
        return True

    def _user(self) -> Caller | None:
        """Check the path, origin and token; reply and return None if refused."""
        is_endpoint, path_token = self._path_token()
        if not is_endpoint:
            self._refuse(404, "not found")
            return None
        if not self._origin_ok():
            return None
        header = self.headers.get("Authorization", "")
        bearer = header[7:].strip() if header[:7].lower() == "bearer " else None
        if bearer and self.server.oauth is not None:  # OAuth tokens: header only
            access = self.server.oauth.access(bearer)
            if access is not None:
                return access.user, clean_source(access.bot), access.scopes
        grant = self.server.grant_for(bearer or path_token)
        if grant is None:
            self._refuse(401, "missing or invalid access token", self._challenge())
            return None
        user, source = grant
        if source is None:  # the token does not fix it, so the caller may say
            query = parse_qs(self.path.partition("?")[2])
            source = clean_source(
                (query.get("source") or [None])[0] or self.headers.get("X-Myelin-Source")
            )
        return user, source, FULL_ACCESS

    def _body(self) -> bytes | None:
        """The request body; replies and returns None if it cannot be read."""
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._refuse(411, "Content-Length required")
            return None
        if length > MAX_BODY_BYTES:
            self._refuse(413, "request too large")
            return None
        return self.rfile.read(length)

    def _token_request(self) -> None:
        oauth = self.server.oauth
        if oauth is None:
            self._refuse(404, "not found")
            return
        if not self._origin_ok():
            return
        body = self._body()
        if body is None:
            return
        no_store = {"Cache-Control": "no-store", "Pragma": "no-cache"}
        try:
            reply = mcp_oauth.grant_token(oauth, self.headers.get("Authorization", ""), body)
        except mcp_oauth.OAuthError as refusal:
            if refusal.status == 401:
                no_store["WWW-Authenticate"] = 'Basic realm="myelin-mcp"'
            self._send(
                refusal.status,
                {"error": refusal.code, "error_description": refusal.description},
                no_store,
            )
            return
        self._send(200, reply, no_store)

    def _metadata(self, path: str) -> bool:
        """Answer an OAuth discovery request; False if the path is not one."""
        if self.server.oauth is None:
            return False
        base = self._base_url()
        if path in (
            mcp_oauth.RESOURCE_METADATA_PATH,
            mcp_oauth.RESOURCE_METADATA_PATH + ENDPOINT,
        ):
            self._send(200, mcp_oauth.resource_metadata(base, ENDPOINT))
            return True
        if path == mcp_oauth.SERVER_METADATA_PATH:
            self._send(200, mcp_oauth.server_metadata(base))
            return True
        if path == mcp_oauth.AUTHORIZE_PATH:
            self._send(400, {
                "error": "unsupported_response_type",
                "error_description": "there is no sign-in page; bots get a token from "
                f"{base}{mcp_oauth.TOKEN_PATH} with the client_credentials grant",
            })
            return True
        return False

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, {"status": "ok"})
            return
        if self._metadata(path):
            return
        if self._user() is not None:
            # No server-initiated messages, so there is no stream to open.
            self._refuse(405, "use POST", {"Allow": "POST"})

    def do_DELETE(self) -> None:
        if self._user() is not None:
            self._refuse(405, "sessions are not used", {"Allow": "POST"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] == mcp_oauth.TOKEN_PATH:
            self._token_request()
            return
        caller = self._user()
        if caller is None:
            return
        user, source, scopes = caller
        body = self._body()
        if body is None:
            return
        try:
            message = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, _error(None, -32700, "parse error"))
            return
        batch = message if isinstance(message, list) else [message]
        needed = {
            mcp_oauth.scope_for(str((item.get("params") or {}).get("name")))
            for item in batch
            if isinstance(item, dict)
            and item.get("method") == "tools/call"
            and isinstance(item.get("params") or {}, dict)
        } - scopes
        if needed:
            wanted = " ".join(sorted(needed))
            self._send(
                403,
                {"error": f"this token lacks the scope {wanted}"},
                self._challenge(f'error="insufficient_scope", scope="{wanted}"'),
            )
            return
        tools = self.server.tools_for(user, source)
        replies = [
            _only_allowed_tools(reply, scopes) for item in batch
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
    bots_file: str = "",
    public_url: str = "",
    token_seconds: int = mcp_oauth.DEFAULT_TOKEN_SECONDS,
) -> int:
    """Serve until interrupted. Returns a process exit code."""
    try:
        tokens = load_tokens(tokens_file, user_id)
        oauth = None
        if bots_file:
            if token_seconds < 60:
                raise ConfigError("--token-seconds must be at least 60")
            oauth = mcp_oauth.Tokens(mcp_oauth.Bots(bots_file), lifetime=token_seconds)
        server = MemoryHTTPServer(
            (host, port), client, tokens=tokens, default_user=user_id,
            allowed_origins=allowed_origins, oauth=oauth, public_url=public_url,
        )
    except (ConfigError, mcp_oauth.BotsFileError) as exc:
        print(f"myelin-mcp: {exc}", file=sys.stderr)
        return 2
    who = f"{len(tokens)} access token(s)" if tokens else "NO access token (loopback only)"
    if oauth is not None:
        who = f"{len(oauth.bots)} OAuth bot(s)" + (f" and {who}" if tokens else "")
    print(
        f"myelin-mcp: MCP over HTTP at http://{host}:{server.server_address[1]}{ENDPOINT} "
        f"with {who}. This server does not encrypt traffic; put HTTPS in front of it.",
        file=sys.stderr,
    )
    if oauth is not None and not public_url and not is_loopback(host):
        print(
            "myelin-mcp: no --public-url given, so the OAuth metadata will name this "
            "server by the Host header over plain http; pass --public-url https://...",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
