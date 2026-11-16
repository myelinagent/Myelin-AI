"""The memory tools served over HTTP."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from myelin import MemoryClient
from myelin.serve import mcp_http

ANA = "ana-token-0123456789abcdef"
BEN = "ben-token-0123456789abcdef"


@pytest.fixture
def served():
    """A running server with two users; yields its base URL."""
    client = MemoryClient(user_id="shared", cite_learning=True)
    server = mcp_http.MemoryHTTPServer(
        ("127.0.0.1", 0), client, tokens={ANA: "ana", BEN: "ben"}, default_user="shared",
        allowed_origins=["https://app.example"],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    client.close()


def _send(url, body=None, *, token=None, method="POST", headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    request = urllib.request.Request(url, data=data, method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode()
            return response.status, json.loads(text) if text else None, response.headers
    except urllib.error.HTTPError as error:
        text = error.read().decode()
        return error.code, json.loads(text) if text else None, error.headers


def _tool(url, token, name, **arguments):
    status, reply, _ = _send(
        url + "/mcp",
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": name, "arguments": arguments}},
        token=token,
    )
    assert status == 200
    return reply["result"]["content"][0]["text"], reply["result"]["isError"]


def test_requests_need_a_valid_token(served):
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    status, body, headers = _send(served + "/mcp", ping)
    assert status == 401 and headers["WWW-Authenticate"] == "Bearer"
    assert "token" in body["error"]
    assert _send(served + "/mcp", ping, token="wrong-token-0123456789")[0] == 401
    assert _send(served + "/mcp", ping, token=ANA)[:2] == (
        200, {"jsonrpc": "2.0", "id": 1, "result": {}},
    )


def test_a_token_in_the_path_works_for_clients_without_headers(served):
    ping = {"jsonrpc": "2.0", "id": 7, "method": "ping"}
    assert _send(f"{served}/mcp/{ANA}", ping)[0] == 200
    assert _send(f"{served}/mcp/not-a-token", ping)[0] == 401
    assert _send(f"{served}/mcp/{ANA}/extra", ping)[0] == 404


def test_initialize_lists_tools_and_notifications_are_accepted(served):
    status, reply, _ = _send(
        served + "/mcp",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        token=ANA,
    )
    assert status == 200 and reply["result"]["serverInfo"]["name"] == "myelin-ai"
    status, body, _ = _send(
        served + "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"}, token=ANA
    )
    assert (status, body) == (202, None)
    status, reply, _ = _send(
        served + "/mcp",
        [{"jsonrpc": "2.0", "id": 2, "method": "ping"},
         {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}],
        token=ANA,
    )
    assert status == 200 and [r["id"] for r in reply] == [2, 3]
    assert len(reply[1]["result"]["tools"]) == 5


def test_each_token_sees_only_its_own_memories(served):
    _tool(served, ANA, "remember", content="Ana's cat is called Miso.")
    text, error = _tool(served, ANA, "recall", query="What is the cat called?")
    assert not error and "Miso" in text
    assert _tool(served, BEN, "recall", query="What is the cat called?") == (
        "No memories match.", False,
    )
    memory_id = text.split("Memory ids:\n[1] ")[1].split("\n")[0]
    assert _tool(served, BEN, "forget", memory_id=memory_id)[1] is True
    recall_id = text.split("recall_id: ")[1].split(".")[0]
    # A recall belongs to the user who made it.
    assert _tool(served, BEN, "mark_used", recall_id=recall_id, used=[1])[1] is True
    assert _tool(served, ANA, "mark_used", recall_id=recall_id, used=[1]) == (
        "Recorded: 1 used, 0 not used.", False,
    )
    # Nothing about where the store is kept leaves the server.
    assert _tool(served, ANA, "memory_stats")[0] == "1 memories. Search: lexical (words)."


def test_other_methods_paths_and_origins_are_refused(served):
    assert _send(served + "/health", method="GET")[:2] == (200, {"status": "ok"})
    status, _, headers = _send(served + "/mcp", method="GET", token=ANA)
    assert status == 405 and headers["Allow"] == "POST"
    assert _send(served + "/mcp", method="GET")[0] == 401
    assert _send(served + "/mcp", method="DELETE", token=ANA)[0] == 405
    assert _send(served + "/elsewhere", {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                 token=ANA)[0] == 404
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert _send(served + "/mcp", ping, token=ANA,
                 headers={"Origin": "https://evil.example"})[0] == 403
    assert _send(served + "/mcp", ping, token=ANA,
                 headers={"Origin": "https://app.example"})[0] == 200


def test_bad_bodies_are_refused(served):
    status, reply, _ = _send(served + "/mcp", raw=b"not json", token=ANA)
    assert status == 400 and reply["error"]["code"] == -32700
    too_big = b" " * (mcp_http.MAX_BODY_BYTES + 1)
    assert _send(served + "/mcp", raw=too_big, token=ANA)[0] == 413


def test_it_will_not_start_open_to_the_network():
    client = MemoryClient(user_id="shared")
    try:
        with pytest.raises(mcp_http.ConfigError, match="without an access token"):
            mcp_http.MemoryHTTPServer(("0.0.0.0", 0), client, tokens={}, default_user="shared")
        # On loopback with no token it serves the default user, for local testing.
        local = mcp_http.MemoryHTTPServer(
            ("127.0.0.1", 0), client, tokens={}, default_user="shared"
        )
        assert local.user_for(None) == "shared"
        local.server_close()
        assert mcp_http.run(client, host="0.0.0.0", port=0, user_id="shared") == 2
    finally:
        client.close()


def test_tokens_come_from_the_environment_and_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MYELIN_MCP_TOKEN", "solo-token-0123456789")
    assert mcp_http.load_tokens("", "scott") == {"solo-token-0123456789": "scott"}

    monkeypatch.delenv("MYELIN_MCP_TOKEN")
    assert mcp_http.load_tokens("", "scott") == {}
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({ANA: "ana", BEN: "ben"}))
    assert mcp_http.load_tokens(str(path), "scott") == {ANA: "ana", BEN: "ben"}

    path.write_text(json.dumps({"short": "ana"}))
    with pytest.raises(mcp_http.ConfigError, match="shorter than"):
        mcp_http.load_tokens(str(path), "scott")
    path.write_text("[1, 2]")
    with pytest.raises(mcp_http.ConfigError, match="JSON object"):
        mcp_http.load_tokens(str(path), "scott")
    with pytest.raises(mcp_http.ConfigError, match="cannot read"):
        mcp_http.load_tokens(str(tmp_path / "missing.json"), "scott")
