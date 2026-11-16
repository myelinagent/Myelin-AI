"""OAuth for the HTTP memory server: one identity per bot.

Each agent ("bot") that may use the memory is an OAuth client with its
own id and secret. It trades them for a short-lived access token at
``POST /oauth/token`` (the OAuth 2.1 client credentials grant), then
calls ``/mcp`` with that token. The token carries the bot's identity:
whose memories it works on, the source label recorded on what it
stores, and whether it may write.

Bots live in a JSON file. It never holds a usable secret, only the
SHA-256 of one, under the key ``secret_sha256``, next to the bot's
``user`` and its ``scopes`` (``memory:read``, ``memory:write``).

``myelin-mcp --add-bot codex --bots-file bots.json`` writes an entry and
prints the secret once. The file is re-read when it changes, so removing
a bot or replacing its secret takes effect at once: tokens issued to a
removed bot, or under a replaced secret, stop working.

Access tokens are random strings kept in memory (hashed), so a restart
ends them all and bots simply ask for a new one.

What this is not: the browser sign-in flow (authorization code). There
is no person to sign in here, only bots that hold a secret.

Standard library only.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote_plus

READ = "memory:read"
WRITE = "memory:write"
ALL_SCOPES = (READ, WRITE)
# What each tool needs. A tool not listed here needs WRITE.
READ_TOOLS = frozenset({"recall", "memory_stats"})

TOKEN_PATH = "/oauth/token"
AUTHORIZE_PATH = "/oauth/authorize"  # published, but always refuses: see below
RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"
SERVER_METADATA_PATH = "/.well-known/oauth-authorization-server"

DEFAULT_TOKEN_SECONDS = 3600
MAX_LIVE_TOKENS = 10_000
BOT_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
DIGEST_KEY = "secret_sha256"


class BotsFileError(Exception):
    """The bots file is missing, unreadable or malformed."""


@dataclass(frozen=True)
class Bot:
    name: str  # the OAuth client id, and the source label on its memories
    user: str
    digest: str  # SHA-256 of the bot's secret, in hex
    scopes: frozenset[str]


@dataclass(frozen=True)
class Access:
    """What a valid access token allows."""

    bot: str
    user: str
    scopes: frozenset[str]


def scope_for(tool: str) -> str:
    return READ if tool in READ_TOOLS else WRITE


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _check_name(name: str) -> None:
    if not BOT_NAME.fullmatch(name):
        raise BotsFileError(
            f"bot name {name!r} must be lower-case letters, digits, '.', '_' or '-'"
        )


def parse_bots(loaded: Any) -> dict[str, Bot]:
    if not isinstance(loaded, dict):
        raise BotsFileError("bots file must be a JSON object of bot name to settings")
    bots: dict[str, Bot] = {}
    for name, entry in loaded.items():
        _check_name(name)
        if not isinstance(entry, dict):
            raise BotsFileError(f"bot {name}: settings must be an object")
        user, digest = entry.get("user"), entry.get(DIGEST_KEY)
        if not isinstance(user, str) or not user:
            raise BotsFileError(f'bot {name}: "user" is required')
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BotsFileError(
                f'bot {name}: "{DIGEST_KEY}" must be the SHA-256 of its secret, in hex'
            )
        scopes = entry.get("scopes", list(ALL_SCOPES))
        if (
            not isinstance(scopes, list)
            or not scopes
            or any(scope not in ALL_SCOPES for scope in scopes)
        ):
            raise BotsFileError(
                f'bot {name}: "scopes" must be a list drawn from {", ".join(ALL_SCOPES)}'
            )
        bots[name] = Bot(name, user, digest, frozenset(scopes))
    return bots


class Bots:
    """The bots file, re-read when it changes."""

    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._bots: dict[str, Bot] = {}
        self._load(strict=True)

    def _load(self, *, strict: bool) -> None:
        try:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._stamp:
                return
            bots = parse_bots(json.loads(self.path.read_text()))
        except (OSError, json.JSONDecodeError, BotsFileError) as exc:
            if strict:
                raise BotsFileError(f"cannot use bots file {self.path}: {exc}") from exc
            # A half-written or broken edit must not lock every bot out, nor
            # let a removed one back in: keep what was last read.
            print(f"myelin-mcp: bots file not reloaded ({exc})", file=sys.stderr)
            return
        self._bots, self._stamp = bots, stamp

    def get(self, name: str) -> Bot | None:
        with self._lock:
            self._load(strict=False)
            return self._bots.get(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._bots)

    def check(self, name: str, proof: str) -> Bot | None:
        """The bot, if the name and the secret it presents are right."""
        bot = self.get(name)
        # Compare against something even when the bot is unknown, so the
        # answer takes as long either way.
        expected = bot.digest if bot else "0" * 64
        matches = hmac.compare_digest(expected, _hash(proof))
        return bot if bot and matches else None


class Tokens:
    """Access tokens that have been issued and have not expired."""

    def __init__(self, bots: Bots, lifetime: int = DEFAULT_TOKEN_SECONDS) -> None:
        self.bots = bots
        self.lifetime = lifetime
        self._lock = threading.Lock()
        # token hash -> (bot name, the bot's digest when issued, scopes, expiry)
        self._live: dict[str, tuple[str, str, frozenset[str], float]] = {}

    def issue(self, bot: Bot, scopes: frozenset[str]) -> str:
        issued = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            for key in [k for k, entry in self._live.items() if entry[3] <= now]:
                del self._live[key]
            while len(self._live) >= MAX_LIVE_TOKENS:  # oldest first
                del self._live[next(iter(self._live))]
            self._live[_hash(issued)] = (bot.name, bot.digest, scopes, now + self.lifetime)
        return issued

    def access(self, presented: str) -> Access | None:
        with self._lock:
            entry = self._live.get(_hash(presented))
        if entry is None or entry[3] <= time.monotonic():
            return None
        name, digest, scopes, _ = entry
        bot = self.bots.get(name)
        if bot is None or bot.digest != digest:
            return None  # removed, or its secret was replaced
        return Access(bot=name, user=bot.user, scopes=scopes & bot.scopes)


class OAuthError(Exception):
    """A refusal from the token endpoint, as RFC 6749 section 5.2 describes it."""

    def __init__(self, status: int, code: str, description: str) -> None:
        super().__init__(description)
        self.status = status
        self.code = code
        self.description = description


def _credentials(authorization: str, form: dict[str, list[str]]) -> tuple[str, str]:
    """The bot name and the secret it presents, from Basic auth or the form."""
    if authorization[:6].lower() == "basic ":
        try:
            decoded = base64.b64decode(authorization[6:].strip(), validate=True).decode()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise OAuthError(401, "invalid_client", "malformed Basic credentials") from exc
        name, colon, proof = decoded.partition(":")
        if not colon:
            raise OAuthError(401, "invalid_client", "malformed Basic credentials")
        return unquote_plus(name), unquote_plus(proof)
    name = (form.get("client_id") or [""])[0]
    proof = (form.get("client_secret") or [""])[0]
    if not name or not proof:
        raise OAuthError(401, "invalid_client", "client credentials are required")
    return name, proof


def grant_token(tokens: Tokens, authorization: str, body: bytes) -> dict[str, Any]:
    """Handle a token request; returns the JSON reply or raises OAuthError."""
    try:
        form = parse_qs(body.decode(), keep_blank_values=True, strict_parsing=bool(body))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OAuthError(400, "invalid_request", "body must be form-encoded") from exc
    grant_type = (form.get("grant_type") or [""])[0]
    if grant_type != "client_credentials":
        raise OAuthError(
            400, "unsupported_grant_type", "only the client_credentials grant is offered"
        )
    name, proof = _credentials(authorization, form)
    bot = tokens.bots.check(name, proof)
    if bot is None:
        raise OAuthError(401, "invalid_client", "unknown bot or wrong secret")
    # A client may ask for every scope the server advertises. Grant the part
    # this bot is allowed and say so in the reply (RFC 6749 section 3.3);
    # refuse only when nothing asked for is allowed.
    asked = frozenset((form.get("scope") or [""])[0].split())
    scopes = asked & bot.scopes if asked else bot.scopes
    if not scopes:
        raise OAuthError(400, "invalid_scope", "this bot is not allowed that scope")
    return {
        "access_token": tokens.issue(bot, scopes),
        "token_type": "Bearer",
        "expires_in": tokens.lifetime,
        "scope": " ".join(sorted(scopes)),
    }


def resource_metadata(base_url: str, endpoint: str) -> dict[str, Any]:
    """RFC 9728: where this resource's tokens come from."""
    return {
        "resource": base_url + endpoint,
        "authorization_servers": [base_url],
        "scopes_supported": list(ALL_SCOPES),
        "bearer_methods_supported": ["header"],
        "resource_name": "Myelin-AI memory",
    }


def server_metadata(base_url: str) -> dict[str, Any]:
    """RFC 8414: what the token endpoint offers."""
    return {
        "issuer": base_url,
        # No grant here uses this endpoint, and RFC 8414 lets it be left out
        # in that case, but widely used clients refuse metadata without it.
        "authorization_endpoint": base_url + AUTHORIZE_PATH,
        "token_endpoint": base_url + TOKEN_PATH,
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "response_types_supported": [],
        "scopes_supported": list(ALL_SCOPES),
    }


def add_bot(path: str, name: str, user: str, *, read_only: bool = False) -> str:
    """Add or replace a bot in the file; returns its new secret."""
    _check_name(name)
    file = Path(path).expanduser()
    loaded: Any = {}
    if file.exists():
        try:
            loaded = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BotsFileError(f"cannot read bots file {file}: {exc}") from exc
        parse_bots(loaded)  # refuse to build on a broken file
    fresh = secrets.token_urlsafe(32)
    loaded[name] = {
        "user": user,
        DIGEST_KEY: _hash(fresh),
        "scopes": [READ] if read_only else list(ALL_SCOPES),
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the file and swap, so a running server never reads half of it.
    partial = file.with_name(file.name + ".partial")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(loaded, handle, indent=2)
        handle.write("\n")
    os.replace(partial, file)
    return fresh
