"""Regression tests for crash, stall and data-loss paths."""

from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

import pytest

from myelin import MemoryClient
from myelin.adapters.mem0 import MemoryClient as Mem0Client
from myelin.agent.orchestrator import OrchestratedAgent
from myelin.serve import sidecar

# ---- search -----------------------------------------------------------


def test_naive_as_of_works_in_working_tier():
    c = MemoryClient(working_memory=True)
    try:
        t = c.observe("Melanie likes tea", session_id="s1")
        c.commit([t.turn_id], session_id="s1")
        hits = c.search(
            "Melanie",
            session_id="s1",
            modes=("working", "long_term"),
            as_of=datetime(2100, 1, 1),  # noqa: DTZ001 — naive on purpose
        )
        assert [h.memory.content for h in hits] == ["Melanie likes tea"]
        assert c.search(
            "Melanie", session_id="s1", modes=("working",),
            as_of=datetime(2000, 1, 1),  # noqa: DTZ001
        ) == []
    finally:
        c.close()


# ---- prune: a newer fact supersedes the older one ---------------------


def _fact(client: MemoryClient, city: str, *, days_ago: int = 0):
    mem = client.add(
        f"Scott lives in {city}",
        metadata={"subject": "scott", "predicate": "lives_in", "object": city},
    )
    if days_ago:
        stamp = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        backend = client._backend
        with backend._lock:
            backend._conn.execute(
                "UPDATE memories SET t_valid = ? WHERE id = ?", (stamp, mem.id)
            )
            backend._conn.commit()
    return mem


def test_prune_keeps_the_newer_of_two_contradicting_facts():
    c = MemoryClient(user_id="u")
    try:
        old = _fact(c, "Paris", days_ago=2)
        new = _fact(c, "Berlin")
        first = c.prune()
        assert (first.quarantined_count, first.retained_count) == (1, 1)
        later = c.prune(now=datetime.now(UTC) + timedelta(days=8))
        assert later.tombstoned_count == 1
        assert c.get(old.id) is None
        assert c.get(new.id).health_state == "healthy"
        hits = c.search("Where does Scott live")
        assert [h.memory.id for h in hits] == [new.id]
    finally:
        c.close()


def test_prune_keeps_only_the_latest_of_a_chain_of_updates():
    c = MemoryClient(user_id="u")
    try:
        paris = _fact(c, "Paris", days_ago=3)
        berlin = _fact(c, "Berlin", days_ago=2)
        back_to_paris = _fact(c, "Paris")
        c.prune()
        states = {
            m.id: m.health_state for m in c._backend.list_memories(user_id="u")
        }
        assert states == {
            paris.id: "quarantined",
            berlin.id: "quarantined",
            back_to_paris.id: "healthy",
        }
    finally:
        c.close()


# ---- commit -----------------------------------------------------------


def test_commit_failure_keeps_unstored_turns_retryable():
    c = MemoryClient(working_memory=True)
    try:
        turns = [
            c.observe(f"fact number {i} about tea", session_id="s")
            for i in range(3)
        ]
        ids = [t.turn_id for t in turns]
        real_add = c._backend.add_memory
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return real_add(**kwargs)

        c._backend.add_memory = flaky
        with pytest.raises(RuntimeError, match="disk full"):
            c.commit(ids, session_id="s")

        handle = c.session("s")
        assert handle.n_observed == 2
        assert handle.n_committed == 1

        c._backend.add_memory = real_add
        retried = c.commit(ids[1:], session_id="s")
        assert [m.content for m in retried] == [
            "fact number 1 about tea",
            "fact number 2 about tea",
        ]
        assert len(c._backend.list_memories(user_id="default")) == 3
        assert c.session("s").n_observed == 0
    finally:
        c.close()


def test_restored_turns_keep_observation_order():
    c = MemoryClient(working_memory=True)
    try:
        first = c.observe("first", session_id="s")
        second = c.observe("second", session_id="s")
        state = c._session_registry.lookup("s")
        drained = state.drain_observed([first.turn_id])
        state.restore_observed(drained)
        assert [t.turn_id for t in state.observed_buffer] == [
            first.turn_id,
            second.turn_id,
        ]
    finally:
        c.close()


def test_commit_stores_under_the_sessions_user():
    c = MemoryClient(working_memory=True, user_id="alice")
    try:
        t = c.observe("bob likes tea", session_id="sx", user_id="bob")
        (mem,) = c.commit([t.turn_id], session_id="sx")
        assert mem.user_id == "bob"
        assert [h.memory.id for h in c.search("tea", user_id="bob")] == [mem.id]
        assert c.search("tea", user_id="alice") == []
    finally:
        c.close()


def test_observe_defaults_to_client_user():
    c = MemoryClient(working_memory=True, user_id="alice")
    try:
        t = c.observe("alice likes tea", session_id="sa")
        (mem,) = c.commit([t.turn_id], session_id="sa")
        assert mem.user_id == "alice"
    finally:
        c.close()


def test_observe_rejects_a_second_user_on_one_session():
    c = MemoryClient(working_memory=True)
    try:
        c.observe("hello there", session_id="s", user_id="alice")
        with pytest.raises(ValueError, match="belongs to user 'alice'"):
            c.observe("hello again", session_id="s", user_id="bob")
    finally:
        c.close()


# ---- consolidate ------------------------------------------------------


def _spy_client(n_facts: int):
    prompts: list[str] = []

    def answerer(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    c = MemoryClient(llm_client=answerer)
    for i in range(n_facts):
        c.add(f"fact {i} mentions zebra{i}")
    return c, prompts


def test_consolidate_rotates_through_the_store():
    c, prompts = _spy_client(10)
    try:
        seen: list[set[str]] = []
        for _ in range(3):
            prompts.clear()
            c.consolidate(max_samples=4)
            seen.append(
                {m for p in prompts for m in re.findall(r"fact (\d+)", p)}
            )
        # Calls test facts 0-3, then 4-7, then 8-9 wrapping round to 0-1.
        assert [len(s) for s in seen] == [4, 4, 4]
        assert seen[0].isdisjoint(seen[1])
        assert seen[0] | seen[1] | seen[2] == {str(i) for i in range(10)}
    finally:
        c.close()


def test_consolidate_passes_retriever_through():
    c, prompts = _spy_client(1)
    try:
        report = c.consolidate(retriever=lambda q: ["zebra0 is the detail"])
        assert report.llm_calls == 3
        assert all(p.startswith("Remembered facts:") for p in prompts)
    finally:
        c.close()


def test_consolidate_treats_non_text_answers_as_misses():
    c = MemoryClient(llm_client=lambda prompt: {"text": "oolong"})
    try:
        c.add("Scott prefers oolong tea")
        report = c.consolidate()
        assert report.llm_calls == 3
        assert report.hit_rate == 0.0
    finally:
        c.close()


# ---- delete -----------------------------------------------------------


def test_delete_removes_tombstoned_content_and_cite_history():
    c = MemoryClient(cite_learning=True)
    try:
        mem = c.add("a secret worth forgetting")
        c.cite(mem.id, answer_correct=True)
        c._backend.transition_state(mem.id, new_state="tombstoned")
        assert c.delete(mem.id) is True
        assert c._backend.list_memories(
            user_id="default", include_tombstoned=True
        ) == []
        n_cites = c._backend._conn.execute(
            "SELECT COUNT(*) FROM cite_events"
        ).fetchone()[0]
        assert n_cites == 0
        assert c.delete(mem.id) is False
    finally:
        c.close()


def test_mem0_delete_all_requires_a_filter():
    m = Mem0Client()
    try:
        m.add("alice likes tea", user_id="alice")
        with pytest.raises(ValueError, match="at least one"):
            m.delete_all()
        assert len(m.get_all(user_id="alice")) == 1
    finally:
        m.close()


def test_mem0_delete_all_reaches_tombstones():
    m = Mem0Client()
    try:
        added = m.add("alice secret", user_id="alice")
        m.add("bob note", user_id="bob")
        backend = m._client._backend
        backend.transition_state(added["id"], new_state="tombstoned")
        assert m.delete_all(user_id="alice")["deleted_count"] == 1
        assert backend.list_memories(
            user_id="alice", include_tombstoned=True
        ) == []
        assert len(m.get_all(user_id="bob")) == 1
    finally:
        m.close()


# ---- orchestrator -----------------------------------------------------


@pytest.fixture
def agent():
    a = OrchestratedAgent.__new__(OrchestratedAgent)
    a._memory = MemoryClient()
    a._memory.add("Scott drinks oolong tea", user_id="u")
    yield a
    a._memory.close()


@pytest.mark.parametrize(
    "arguments",
    [
        '{"query": "tea", "top_k": null}',
        '{"query": "tea", "top_k": "five"}',
        '{"query": "tea", "top_k": -3}',
        '{"query": "tea", "top_k": 100000}',
        {"query": "tea"},
    ],
)
def test_memory_tool_tolerates_odd_top_k(agent, arguments):
    out = agent._execute_memory_tool("memory_search", arguments, user_id="u")
    assert "Scott drinks oolong tea" in out


@pytest.mark.parametrize(
    "arguments", ['["tea"]', "not json", '{"query": 7}', "null", None]
)
def test_memory_tool_reports_bad_arguments(agent, arguments):
    out = agent._execute_memory_tool("memory_search", arguments, user_id="u")
    assert out == "memory_search error: missing query"


# ---- sidecar ----------------------------------------------------------


@pytest.fixture
def server():
    srv = sidecar.build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv.server_address
    srv.shutdown()
    srv.server_close()
    sidecar._reset_client()


def _post(base: str, path: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(base + path, data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_sidecar_serves_others_while_one_client_stalls(server):
    base, address = server
    stalled = socket.create_connection(address)
    try:
        stalled.sendall(
            b"POST /tools/memory_search HTTP/1.1\r\n"
            b"Host: x\r\nContent-Length: 100\r\n\r\n"
        )
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
    finally:
        stalled.close()


def test_sidecar_round_trip(server):
    base, _ = server
    status, stored = _post(
        base, "/tools/memory_store",
        json.dumps({"content": "Scott drinks oolong tea"}).encode(),
    )
    assert status == 200
    status, found = _post(
        base, "/tools/memory_search", json.dumps({"query": "oolong"}).encode()
    )
    assert status == 200
    assert found["hits"][0]["memory"]["id"] == stored["memory"]["id"]


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        ("/tools/memory_search", b"[1, 2]", 400),
        ("/tools/memory_search", b"{not json", 400),
        ("/tools/memory_search", b"{}", 400),
        ("/tools/memory_update", b'{"memory_id": "nope", "content": "x"}', 404),
        ("/tools/nope", b"{}", 404),
    ],
)
def test_sidecar_rejects_bad_requests(server, path, body, expected):
    base, _ = server
    status, payload = _post(base, path, body)
    assert status == expected
    assert "error" in payload


def test_sidecar_refuses_oversized_body(server):
    _, address = server
    with socket.create_connection(address, timeout=5) as s:
        s.sendall(
            b"POST /tools/memory_store HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: 999999999\r\n\r\n"
        )
        assert s.recv(64).startswith(b"HTTP/1.0 413")
