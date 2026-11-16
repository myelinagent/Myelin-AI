"""The MCP server that exposes Myelin as memory for an assistant."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from myelin import MemoryClient, __version__
from myelin.serve import mcp


@pytest.fixture
def server():
    client = MemoryClient(user_id="t", cite_learning=True)
    yield mcp.MemoryServer(client)
    client.close()


def _request(server, method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def _recall_id(text):
    return text.split("recall_id: ")[1].split(".")[0]


def _call(server, name, **arguments):
    reply = _request(server, "tools/call", {"name": name, "arguments": arguments})
    result = reply["result"]
    return result["content"][0]["text"], result["isError"]


def test_initialize_echoes_a_known_protocol_version_and_names_the_server(server):
    result = _request(server, "initialize", {"protocolVersion": "2024-11-05"})["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"] == {"name": "myelin-ai", "version": __version__}
    assert "recall" in result["instructions"]

    newest = _request(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert newest["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]


def test_tools_are_listed_with_schemas(server):
    tools = _request(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == [
        "remember", "recall", "mark_used", "forget", "memory_stats",
    ]
    for tool in tools:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_notifications_get_no_reply_and_unknown_methods_an_error(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert _request(server, "ping")["result"] == {}
    assert _request(server, "resources/list")["error"]["code"] == -32601


def test_remember_then_recall_then_forget(server):
    text, error = _call(server, "remember", content="Priya prefers tabs over spaces.")
    assert not error and text.startswith("Remembered (id ")
    _call(server, "remember", content="The staging database is called heron.")

    text, error = _call(server, "recall", query="What does Priya prefer?")
    assert not error
    assert "[1] " in text and "Priya prefers tabs over spaces." in text
    assert _recall_id(text).startswith("r")

    memory_id = text.split("Memory ids:\n[1] ")[1].split("\n")[0]
    assert _call(server, "forget", memory_id=memory_id) == ("Forgotten.", False)
    assert _call(server, "recall", query="Priya tabs") == ("No memories match.", False)
    text, error = _call(server, "forget", memory_id=memory_id)
    assert error and "no memory" in text


def test_mark_used_records_what_the_answer_relied_on(server):
    _call(server, "remember", content="Deploys go out on Tuesday mornings.")
    _call(server, "remember", content="Deploys need two approvals.")
    text, _ = _call(server, "recall", query="deploys")
    assert "[2] " in text
    recall_id = _recall_id(text)

    assert _call(server, "mark_used", recall_id=recall_id, used=[1]) == (
        "Recorded: 1 used, 1 not used.", False,
    )
    counted = server.client.cite_telemetry
    assert counted["cite_learning_enabled"]

    text, error = _call(server, "mark_used", recall_id=recall_id, used=[3])
    assert error and "1 to 2" in text
    text, error = _call(server, "mark_used", recall_id="nope", used=[])
    assert error and "unknown recall_id" in text


def test_bad_arguments_are_reported_to_the_model_not_raised(server):
    assert _call(server, "remember", content="   ") == ("content is empty", True)
    assert _call(server, "recall", query="")[1] is True
    assert _call(server, "remember", content="x", importance=2)[1] is True
    assert _call(server, "remember", content="x", importance="high")[1] is True
    text, error = _call(server, "no_such_tool")
    assert error and "unknown tool" in text


def test_stats_count_and_search_mode(server):
    _call(server, "remember", content="One thing.")
    text, error = _call(server, "memory_stats")
    assert not error
    assert text == "1 memories. Search: lexical (words)."


def test_only_recent_recalls_are_kept(server):
    _call(server, "remember", content="Nadia runs the weekly planning meeting.")
    ids = [
        _recall_id(_call(server, "recall", query="planning meeting")[0])
        for _ in range(mcp.RECALLS_KEPT + 1)
    ]
    assert len(set(ids)) == len(ids)
    assert _call(server, "mark_used", recall_id=ids[0], used=[])[1] is True
    assert _call(server, "mark_used", recall_id=ids[-1], used=[1])[1] is False


def test_serve_reads_lines_and_survives_garbage(server):
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        "",
        "not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps([{"jsonrpc": "2.0", "id": 2, "method": "ping"}, 7]),
    ]
    out = io.StringIO()
    mcp.serve(server, io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700


def test_the_command_keeps_memories_between_runs(tmp_path):
    db = tmp_path / "nested" / "memory.db"

    def run(*messages):
        done = subprocess.run(
            [sys.executable, "-m", "myelin.serve.mcp", "--db-path", str(db), "--no-embedder"],
            input="".join(json.dumps(m) + "\n" for m in messages),
            capture_output=True, text=True, timeout=60, check=True,
        )
        return [json.loads(line) for line in done.stdout.splitlines()]

    def call(name, **arguments):
        return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": arguments}}

    run(call("remember", content="Omar's team ships from the release branch."))
    (reply,) = run(call("recall", query="Where does Omar's team ship from?"))
    assert "release branch" in reply["result"]["content"][0]["text"]


def test_tools_are_scoped_to_one_user():
    client = MemoryClient(user_id="shared", cite_learning=True)
    try:
        ana = mcp.MemoryServer(client, user_id="ana")
        ben = mcp.MemoryServer(client, user_id="ben")
        _call(ana, "remember", content="Ana keeps the release notes in Notion.")
        assert _call(ben, "recall", query="release notes") == ("No memories match.", False)
        assert _call(ben, "memory_stats")[0].startswith("0 memories")

        text, _ = _call(ana, "recall", query="release notes")
        memory_id = text.split("Memory ids:\n[1] ")[1].split("\n")[0]
        # Another user cannot forget it, and is told only that it is not there.
        text, error = _call(ben, "forget", memory_id=memory_id)
        assert error and "no memory" in text
        assert _call(ana, "forget", memory_id=memory_id) == ("Forgotten.", False)
    finally:
        client.close()


def test_memories_are_tagged_with_the_agent_that_stored_them():
    client = MemoryClient(user_id="t")
    try:
        named = mcp.MemoryServer(client, source="My Agent!")
        assert named.source == "my-agent"
        _request(named, "initialize", {"clientInfo": {"name": "claude-code"}})
        assert named.source == "my-agent"  # an explicit label wins

        announced = mcp.MemoryServer(client)
        _call(announced, "remember", content="Stored before any client name is known.")
        _request(announced, "initialize", {"clientInfo": {"name": "Claude Code", "version": "2"}})
        _call(announced, "remember", content="Stored after the client introduced itself.")
        text, _ = _call(announced, "recall", query="stored client")
        ids = text.split("Memory ids:\n")[1].split("\n\n")[0].splitlines()
        assert sum("(from claude-code)" in line for line in ids) == 1
        assert sum("(from" not in line for line in ids) == 1
    finally:
        client.close()
