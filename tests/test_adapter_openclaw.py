"""Tests for `myelin.adapters.openclaw.OpenClawMemoryPlugin`.

These tests do NOT require `openclaw` or `tencentdb_agent_memory` to be
installed — they target the adapter's interface directly.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from myelin import MemoryClient
from myelin.adapters.openclaw import (
    OPENCLAW_LIFECYCLE_HOOKS,
    OPENCLAW_TOOL_NAMES,
    OpenClawMemoryPlugin,
)


# ---- Fixtures ----------------------------------------------------


@pytest.fixture
def client():
    c = MemoryClient(db_path=":memory:", user_id="alice")
    yield c
    c.close()


@pytest.fixture
def plugin(client):
    return OpenClawMemoryPlugin(client=client, name="myelin-test")


# ---- 1. Instantiation + protocol-required surface ----------------


def test_instantiation_requires_memory_client():
    with pytest.raises(TypeError):
        OpenClawMemoryPlugin(client="not-a-client")  # type: ignore[arg-type]


def test_instantiation_rejects_bad_name(client):
    with pytest.raises(ValueError):
        OpenClawMemoryPlugin(client=client, name="")
    with pytest.raises(ValueError):
        OpenClawMemoryPlugin(client=client, name="bad name with spaces")


def test_instantiation_rejects_nonpositive_top_k(client):
    with pytest.raises(ValueError):
        OpenClawMemoryPlugin(client=client, recall_top_k=0)


def test_protocol_required_methods_present(plugin):
    # Lifecycle hooks
    for hook in OPENCLAW_LIFECYCLE_HOOKS:
        assert callable(getattr(plugin, hook)), (
            f"missing lifecycle hook {hook!r}"
        )
    # Identity / manifest surface
    assert plugin.name == "myelin-test"
    assert plugin.slot == "memory"
    assert callable(plugin.manifest)
    assert callable(plugin.invoke_tool)


def test_all_documented_tools_are_registered(plugin):
    registered = {t["name"] for t in plugin.tools}
    assert registered == set(OPENCLAW_TOOL_NAMES), (
        f"tool set mismatch — registered={registered}, "
        f"expected={set(OPENCLAW_TOOL_NAMES)}"
    )


# ---- 2. Tool & manifest schemas JSON-serialize -------------------


def test_tools_list_is_json_serializable(plugin):
    # Critical: the tools list must serialize for harness emission.
    encoded = json.dumps(plugin.tools)
    decoded = json.loads(encoded)
    assert isinstance(decoded, list)
    assert {t["name"] for t in decoded} == set(OPENCLAW_TOOL_NAMES)
    for tool in decoded:
        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["input_schema"], dict)
        assert tool["input_schema"].get("type") == "object"


def test_tools_list_does_not_leak_handler(plugin):
    # Internal `_handler` callable must NOT appear in the public tools list
    # (otherwise json.dumps would fail and harnesses couldn't introspect).
    for tool in plugin.tools:
        assert "_handler" not in tool
        assert "handler" not in tool


def test_manifest_serializable_and_well_formed(plugin):
    manifest = plugin.manifest()
    json.dumps(manifest)  # must not raise
    assert manifest["name"] == "myelin-test"
    assert manifest["slot"] == "memory"
    assert set(manifest["tools"]) == set(OPENCLAW_TOOL_NAMES)
    assert set(manifest["hooks"]) == set(OPENCLAW_LIFECYCLE_HOOKS)
    assert "version" in manifest
    assert isinstance(manifest["config_schema"], dict)


# ---- 3. Tool happy-paths via invoke_tool() -----------------------


def test_invoke_unknown_tool_raises(plugin):
    with pytest.raises(KeyError):
        plugin.invoke_tool("not_a_tool", {})


def test_invoke_rejects_non_dict_arguments(plugin):
    with pytest.raises(TypeError):
        plugin.invoke_tool("memory_search", "not a dict")  # type: ignore[arg-type]


def test_tool_memory_store_then_search(plugin):
    stored = plugin.invoke_tool(
        "memory_store", {"content": "Scott prefers green tea."}
    )
    assert stored["id"]
    assert stored["content"] == "Scott prefers green tea."

    hits = plugin.invoke_tool(
        "memory_search", {"query": "What does Scott like?", "top_k": 3}
    )
    assert isinstance(hits, list)
    assert len(hits) >= 1
    assert any("green tea" in h["memory"]["content"] for h in hits)
    # Output is JSON-safe end-to-end.
    json.dumps(hits)


def test_tool_memory_store_missing_content_raises(plugin):
    with pytest.raises(ValueError):
        plugin.invoke_tool("memory_store", {})


def test_tool_memory_get_returns_none_for_unknown(plugin):
    assert (
        plugin.invoke_tool("memory_get", {"memory_id": "does-not-exist"})
        is None
    )


def test_tool_memory_update_then_get(plugin):
    stored = plugin.invoke_tool("memory_store", {"content": "v1"})
    mid = stored["id"]
    updated = plugin.invoke_tool(
        "memory_update", {"memory_id": mid, "content": "v2"}
    )
    assert updated["content"] == "v2"
    fetched = plugin.invoke_tool("memory_get", {"memory_id": mid})
    assert fetched["content"] == "v2"


def test_tool_memory_forget_then_get(plugin):
    stored = plugin.invoke_tool("memory_store", {"content": "delete me"})
    mid = stored["id"]
    out = plugin.invoke_tool("memory_forget", {"memory_id": mid})
    assert out == {"deleted": True}
    assert plugin.invoke_tool("memory_get", {"memory_id": mid}) is None


def test_tool_memory_store_with_event_anchor(plugin):
    stored = plugin.invoke_tool(
        "memory_store",
        {
            "content": "Scott met Mia at the market.",
            "event": {
                "event_id": "e-1",
                "type": "meeting",
                "anchor_datetime": "2026-05-01T12:00:00+00:00",
                "location": "market",
                "participant_ids": ["scott", "mia"],
                "evidence_turn_ids": ["t-7"],
            },
        },
    )
    assert stored["event"]["event_id"] == "e-1"
    assert stored["event"]["location"] == "market"


# ---- 4. user_id / agent_id scoping -------------------------------


def test_invoke_tool_respects_user_id_scope():
    c = MemoryClient(db_path=":memory:", user_id="alice")
    try:
        p = OpenClawMemoryPlugin(client=c)
        p.invoke_tool(
            "memory_store",
            {"content": "alice's secret", "user_id": "alice"},
        )
        p.invoke_tool(
            "memory_store",
            {"content": "bob's secret", "user_id": "bob"},
        )

        alice_hits = p.invoke_tool(
            "memory_search",
            {"query": "secret", "user_id": "alice"},
        )
        bob_hits = p.invoke_tool(
            "memory_search",
            {"query": "secret", "user_id": "bob"},
        )

        alice_contents = {h["memory"]["content"] for h in alice_hits}
        bob_contents = {h["memory"]["content"] for h in bob_hits}
        assert "alice's secret" in alice_contents
        assert "bob's secret" not in alice_contents
        assert "bob's secret" in bob_contents
        assert "alice's secret" not in bob_contents
    finally:
        c.close()


# ---- 5. Novel mechanism tools (consolidate + prune) --------------


def test_tool_memory_consolidate_no_llm_returns_well_formed_report(plugin):
    # Seed a few facts.
    for fact in ["one", "two", "three"]:
        plugin.invoke_tool("memory_store", {"content": fact})

    report = plugin.invoke_tool(
        "memory_consolidate", {"max_samples": 16}
    )
    # No llm_client → hit_rate 0, but report well-formed.
    assert report["hit_rate"] == 0.0
    assert report["llm_calls"] == 0
    assert report["facts_sampled"] >= 1
    json.dumps(report)


def test_tool_memory_prune_advances_state_machine():
    c = MemoryClient(db_path=":memory:", user_id="u")
    try:
        p = OpenClawMemoryPlugin(client=c)
        # Add a fact with low confidence/importance so health drops fast.
        p.invoke_tool(
            "memory_store",
            {
                "content": "shaky fact",
                "confidence": 0.05,
                "importance": 0.05,
            },
        )
        # Fast-forward 21 days so apoptosis fires.
        future = (datetime.now(UTC) + timedelta(days=21)).isoformat()
        report = p.invoke_tool(
            "memory_prune",
            {"now": future, "caspase_window_days": 7},
        )
        assert report["scanned"] >= 1
        # Either quarantined or tombstoned — both are valid transitions.
        assert (
            report["quarantined_count"] + report["tombstoned_count"] >= 1
        )
        json.dumps(report)
    finally:
        c.close()


# ---- 6. Lifecycle hooks: before_agent_start + agent_end ----------


def test_before_agent_start_empty_query_returns_empty(plugin):
    out = plugin.before_agent_start({"user_id": "alice"})
    assert out == {"messages": [], "system_context": "", "hits": []}


def test_before_agent_start_with_query_recalls_memories(plugin):
    plugin.invoke_tool("memory_store", {"content": "Scott prefers tea."})
    out = plugin.before_agent_start(
        {"user_id": "alice", "query": "what does Scott prefer?"}
    )
    assert len(out["hits"]) >= 1
    assert "Scott prefers tea" in out["system_context"]
    assert out["messages"][0]["role"] == "system"
    assert "Scott prefers tea" in out["messages"][0]["content"]
    json.dumps(out)  # serializable for harness shipment


def test_before_agent_start_rejects_non_dict(plugin):
    with pytest.raises(TypeError):
        plugin.before_agent_start("not a context")  # type: ignore[arg-type]


def test_agent_end_captures_durable_content(plugin):
    out = plugin.agent_end(
        {"user_id": "alice", "content": "Scott just bought a kettle."}
    )
    assert out["stored"] is True
    assert out["memory_id"]
    # Round-trip: that fact should now be searchable.
    hits = plugin.invoke_tool("memory_search", {"query": "kettle"})
    assert any("kettle" in h["memory"]["content"] for h in hits)


def test_agent_end_skips_empty_content(plugin):
    out = plugin.agent_end({"content": "   "})
    assert out == {"stored": False, "memory_id": None}


def test_agent_end_no_op_when_auto_capture_disabled(client):
    p = OpenClawMemoryPlugin(client=client, auto_capture=False)
    out = p.agent_end({"content": "this should not be stored"})
    assert out == {"stored": False, "memory_id": None}
    hits = p.invoke_tool("memory_search", {"query": "should not be stored"})
    assert hits == []


def test_agent_end_rejects_non_dict(plugin):
    with pytest.raises(TypeError):
        plugin.agent_end("not a turn")  # type: ignore[arg-type]
