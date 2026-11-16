"""OAuth on the HTTP memory server: one identity per bot."""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from myelin import MemoryClient
from myelin.serve import mcp, mcp_http, mcp_oauth

STATIC = "ana-static-token-0123456789"


def _send(url, *, data=None, headers=None, method="POST"):
    request = urllib.request.Request(url, data=data, method=method)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode()
            return response.status, json.loads(text) if text else None, response.headers
    except urllib.error.HTTPError as error:
        text = error.read().decode()
        return error.code, json.loads(text) if text else None, error.headers


def _token(url, name, proof, *, basic=False, **extra):
    form = {"grant_type": "client_credentials", **extra}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if basic:
        pair = base64.b64encode(f"{name}:{proof}".encode()).decode()
        headers["Authorization"] = f"Basic {pair}"
    else:
        form.update(client_id=name, client_secret=proof)
    return _send(
        url + "/oauth/token", data=urllib.parse.urlencode(form).encode(), headers=headers
    )


def _rpc(url, bearer, method, params=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return _send(
        url + "/mcp", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
    )


def _tool(url, bearer, name, **arguments):
    status, reply, headers = _rpc(url, bearer, "tools/call", {"name": name, "arguments": arguments})
    if status != 200:
        return status, headers
    return reply["result"]["content"][0]["text"]


class Served:
    def __init__(self, tmp_path, *, tokens=None, lifetime=3600):
        self.bots_file = str(tmp_path / "bots.json")
        self.proofs = {
            "codex": mcp_oauth.add_bot(self.bots_file, "codex", "ana"),
            "claude-code": mcp_oauth.add_bot(self.bots_file, "claude-code", "ana"),
            "reader": mcp_oauth.add_bot(self.bots_file, "reader", "ana", read_only=True),
            "bens-bot": mcp_oauth.add_bot(self.bots_file, "bens-bot", "ben"),
        }
        self.client = MemoryClient(user_id="shared", cite_learning=True)
        self.oauth = mcp_oauth.Tokens(mcp_oauth.Bots(self.bots_file), lifetime=lifetime)
        self.server = mcp_http.MemoryHTTPServer(
            ("127.0.0.1", 0), self.client, tokens=tokens or {}, default_user="shared",
            oauth=self.oauth,
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def bearer(self, name, **extra):
        status, reply, _ = _token(self.url, name, self.proofs[name], **extra)
        assert status == 200, reply
        return reply["access_token"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.client.close()


@pytest.fixture
def served(tmp_path):
    running = Served(tmp_path)
    yield running
    running.close()


def test_a_bot_signs_in_and_its_memories_carry_its_name(served):
    status, reply, headers = _token(served.url, "codex", served.proofs["codex"])
    assert status == 200
    assert reply["token_type"] == "Bearer"
    assert reply["expires_in"] == 3600
    assert reply["scope"] == "memory:read memory:write"
    assert headers["Cache-Control"] == "no-store"

    assert "Remembered" in _tool(
        served.url, reply["access_token"], "remember", content="The deploy runs at noon on Fridays"
    )
    # Another bot of the same person recalls it, and sees who stored it.
    found = _tool(served.url, served.bearer("claude-code"), "recall", query="when is the deploy")
    assert "noon on Fridays" in found
    assert "(from codex)" in found
    # A bot of a different person does not.
    assert "noon on Fridays" not in _tool(
        served.url, served.bearer("bens-bot"), "recall", query="when is the deploy"
    )


def test_credentials_can_be_sent_as_basic_auth(served):
    status, reply, _ = _token(served.url, "codex", served.proofs["codex"], basic=True)
    assert status == 200
    assert _rpc(served.url, reply["access_token"], "tools/list")[0] == 200


def test_wrong_credentials_get_no_token(served):
    for name, proof in [("codex", "wrong"), ("nobody", served.proofs["codex"]), ("codex", "")]:
        status, reply, headers = _token(served.url, name, proof)
        assert status == 401
        assert reply["error"] == "invalid_client"
        assert headers["WWW-Authenticate"].startswith("Basic")
    status, reply, _ = _send(
        served.url + "/oauth/token", data=b"grant_type=password&username=a&password=b"
    )
    assert (status, reply["error"]) == (400, "unsupported_grant_type")
    status, reply, _ = _send(
        served.url + "/oauth/token", data=b"grant_type=client_credentials",
        headers={"Authorization": "Basic not-base64!"},
    )
    assert (status, reply["error"]) == (401, "invalid_client")
    # A browser page may not ask for tokens.
    status, _, _ = _send(
        served.url + "/oauth/token", data=b"grant_type=client_credentials",
        headers={"Origin": "https://evil.example"},
    )
    assert status == 403


def test_a_read_only_bot_can_recall_but_not_change_anything(served):
    _tool(served.url, served.bearer("codex"), "remember", content="The office is on Baker Lane")
    reader = served.bearer("reader")

    assert "Baker Lane" in _tool(served.url, reader, "recall", query="where is the office")
    assert "memor" in _tool(served.url, reader, "memory_stats")
    for name, arguments in [
        ("remember", {"content": "x"}), ("forget", {"memory_id": "x"}), ("mark_used", {}),
    ]:
        status, headers = _tool(served.url, reader, name, **arguments)
        assert status == 403
        assert 'error="insufficient_scope"' in headers["WWW-Authenticate"]
        assert 'scope="memory:write"' in headers["WWW-Authenticate"]
    # It is only offered the tools it may call.
    _, reply, _ = _rpc(served.url, reader, "tools/list")
    assert {t["name"] for t in reply["result"]["tools"]} == {"recall", "memory_stats"}
    _, reply, _ = _rpc(served.url, served.bearer("codex"), "tools/list")
    assert len(reply["result"]["tools"]) == len(mcp.TOOLS)

    # It cannot ask for more than it was given.
    status, reply, _ = _token(served.url, "reader", served.proofs["reader"], scope="memory:write")
    assert (status, reply["error"]) == (400, "invalid_scope")
    # Asking for everything gets it only what it is allowed, and says so.
    status, reply, _ = _token(
        served.url, "reader", served.proofs["reader"], scope="memory:read memory:write"
    )
    assert (status, reply["scope"]) == (200, "memory:read")
    assert _tool(served.url, reply["access_token"], "remember", content="x")[0] == 403


def test_a_bot_can_ask_for_less_than_it_is_allowed(served):
    narrow = served.bearer("codex", scope="memory:read")
    status, _ = _tool(served.url, narrow, "remember", content="x")
    assert status == 403


def test_a_401_says_where_to_get_a_token(served):
    status, _, headers = _rpc(served.url, "not-a-token", "tools/list")
    assert status == 401
    challenge = headers["WWW-Authenticate"]
    assert challenge.startswith("Bearer ")
    link = challenge.split('resource_metadata="')[1].split('"')[0]
    assert link == served.url + "/.well-known/oauth-protected-resource"

    status, resource, _ = _send(link, method="GET")
    assert status == 200
    assert resource["resource"] == served.url + "/mcp"
    assert resource["authorization_servers"] == [served.url]
    assert _send(link + "/mcp", method="GET")[1] == resource

    status, issuer, _ = _send(served.url + "/.well-known/oauth-authorization-server", method="GET")
    assert status == 200
    assert issuer["issuer"] == served.url
    assert issuer["token_endpoint"] == served.url + "/oauth/token"
    assert issuer["grant_types_supported"] == ["client_credentials"]
    # Published because clients insist on it, but nobody can sign in there.
    status, reply, _ = _send(issuer["authorization_endpoint"], method="GET")
    assert (status, reply["error"]) == (400, "unsupported_response_type")


def test_the_public_url_is_what_the_metadata_names(tmp_path):
    bots_file = str(tmp_path / "bots.json")
    mcp_oauth.add_bot(bots_file, "codex", "ana")
    client = MemoryClient(user_id="shared")
    server = mcp_http.MemoryHTTPServer(
        ("127.0.0.1", 0), client, tokens={}, default_user="shared",
        oauth=mcp_oauth.Tokens(mcp_oauth.Bots(bots_file)),
        public_url="https://memory.example.com/",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        _, issuer, _ = _send(url + "/.well-known/oauth-authorization-server", method="GET")
        assert issuer["token_endpoint"] == "https://memory.example.com/oauth/token"
        _, _, headers = _rpc(url, "nope", "tools/list")
        assert "https://memory.example.com/.well-known/" in headers["WWW-Authenticate"]
    finally:
        server.shutdown()
        server.server_close()
        client.close()


def test_an_oauth_token_is_not_accepted_in_the_url(served):
    bearer = served.bearer("codex")
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    assert _send(f"{served.url}/mcp/{bearer}", data=body)[0] == 401


def test_removing_a_bot_or_replacing_its_secret_ends_its_tokens(served):
    codex, claude = served.bearer("codex"), served.bearer("claude-code")
    assert _rpc(served.url, codex, "tools/list")[0] == 200

    replaced = mcp_oauth.add_bot(served.bots_file, "codex", "ana")
    assert _rpc(served.url, codex, "tools/list")[0] == 401
    assert _token(served.url, "codex", served.proofs["codex"])[0] == 401
    assert _token(served.url, "codex", replaced)[0] == 200

    loaded = json.loads(Path(served.bots_file).read_text())
    del loaded["claude-code"]
    with open(served.bots_file, "w") as handle:
        json.dump(loaded, handle)
    assert _rpc(served.url, claude, "tools/list")[0] == 401
    assert _token(served.url, "claude-code", served.proofs["claude-code"])[0] == 401


def test_a_broken_edit_keeps_the_last_good_bots(served, capsys):
    bearer = served.bearer("codex")
    with open(served.bots_file, "w") as handle:
        handle.write("{ not json")
    assert _rpc(served.url, bearer, "tools/list")[0] == 200
    assert "not reloaded" in capsys.readouterr().err


def test_tokens_expire(tmp_path, monkeypatch):
    running = Served(tmp_path, lifetime=100)
    try:
        bearer = running.bearer("codex")
        assert _rpc(running.url, bearer, "tools/list")[0] == 200
        real = mcp_oauth.time.monotonic
        monkeypatch.setattr(mcp_oauth.time, "monotonic", lambda: real() + 101)
        assert _rpc(running.url, bearer, "tools/list")[0] == 401
    finally:
        running.close()


def test_static_tokens_still_work_beside_oauth(tmp_path):
    running = Served(tmp_path, tokens={STATIC: ("ana", None)})
    try:
        assert "Remembered" in _tool(running.url, STATIC, "remember", content="Static tokens still work")
        found = _tool(running.url, running.bearer("codex"), "recall", query="static tokens")
        assert "Static tokens still work" in found
    finally:
        running.close()


def test_with_bots_and_no_static_token_nothing_is_open(served):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    assert _send(served.url + "/mcp", data=body)[0] == 401


def test_without_bots_there_is_no_oauth(tmp_path):
    client = MemoryClient(user_id="shared")
    server = mcp_http.MemoryHTTPServer(
        ("127.0.0.1", 0), client, tokens={STATIC: ("ana", None)}, default_user="shared"
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert _send(url + "/oauth/token", data=b"grant_type=client_credentials")[0] == 404
        assert _send(url + "/.well-known/oauth-protected-resource", method="GET")[0] == 404
        _, _, headers = _rpc(url, "nope", "tools/list")
        assert headers["WWW-Authenticate"] == "Bearer"
    finally:
        server.shutdown()
        server.server_close()
        client.close()


def test_the_bots_file_holds_no_usable_secret_and_is_private(tmp_path):
    bots_file = tmp_path / "bots.json"
    proof = mcp_oauth.add_bot(str(bots_file), "codex", "ana")
    text = bots_file.read_text()
    assert proof not in text
    assert json.loads(text)["codex"]["scopes"] == ["memory:read", "memory:write"]
    if os.name == "posix":
        assert bots_file.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"Bad Name": {"user": "a", "secret_sha256": "' + "0" * 64 + '"}}',
        '{"codex": {"secret_sha256": "' + "0" * 64 + '"}}',
        '{"codex": {"user": "a", "secret_sha256": "short"}}',
        '{"codex": {"user": "a", "secret_sha256": "' + "0" * 64 + '", "scopes": ["admin"]}}',
        "{ not json",
    ],
)
def test_a_malformed_bots_file_stops_the_server_starting(tmp_path, content, capsys):
    bots_file = tmp_path / "bots.json"
    bots_file.write_text(content)
    client = MemoryClient(user_id="shared")
    try:
        code = mcp_http.run(
            client, host="127.0.0.1", port=0, user_id="shared", bots_file=str(bots_file)
        )
    finally:
        client.close()
    assert code == 2
    assert "bots file" in capsys.readouterr().err


def test_add_bot_from_the_command_line(tmp_path, capsys):
    bots_file = str(tmp_path / "bots.json")
    assert mcp.main(["--add-bot", "codex", "--bots-file", bots_file, "--user-id", "ana"]) == 0
    printed = capsys.readouterr().out
    assert "client id:     codex" in printed
    proof = printed.split("client secret: ")[1].split()[0]
    assert mcp_oauth.Bots(bots_file).check("codex", proof).user == "ana"

    assert mcp.main(
        ["--add-bot", "reader", "--bots-file", bots_file, "--user-id", "ana", "--read-only"]
    ) == 0
    assert mcp_oauth.Bots(bots_file).get("reader").scopes == {"memory:read"}
    assert mcp.main(["--add-bot", "Not Valid", "--bots-file", bots_file]) == 2
