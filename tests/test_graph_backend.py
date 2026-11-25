"""Tests for GraphBackend protocol and SqliteBackend knowledge-graph substrate."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

import myelin
from myelin import Edge, Entity, Flag, GraphStats, Mention
from myelin.backend import GraphBackend
from myelin.backends.sqlite import SqliteBackend, normalize_alias


def _dt(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, 0, tzinfo=UTC)


def test_public_exports_and_protocol_conformance():
    for symbol in ("Edge", "Entity", "Flag", "GraphStats", "Mention"):
        assert symbol in myelin.__all__
        assert symbol in myelin.types.__all__

    backend = SqliteBackend()
    try:
        assert isinstance(backend, GraphBackend)
    finally:
        backend.close()


def test_normalize_alias_cases():
    assert normalize_alias("  Dr. Alice's-Lab (v2)! ") == "dr alice s lab v2"
    assert normalize_alias("john_doe") == "john doe"
    assert normalize_alias("   ---!!!   ") == ""
    assert normalize_alias("Café-au-Lait") == "café au lait"


def test_entity_upsert_validation_and_alias_replacement():
    backend = SqliteBackend()
    try:
        with pytest.raises(ValueError):
            backend.upsert_entity(
                Entity(id="", user_id="u1", type="person", name="Alice", t_valid=_dt(1))
            )
        with pytest.raises(ValueError):
            backend.upsert_entity(
                Entity(id="e1", user_id="  ", type="person", name="Alice", t_valid=_dt(1))
            )
        with pytest.raises(ValueError):
            backend.upsert_entity(
                Entity(id="e1", user_id="u1", type="", name="Alice", t_valid=_dt(1))
            )
        with pytest.raises(ValueError):
            backend.upsert_entity(
                Entity(id="e1", user_id="u1", type="person", name=" ", t_valid=_dt(1))
            )

        ent = Entity(
            id="ent-alice",
            user_id="u1",
            type="person",
            name="Dr. Alice Smith",
            t_valid=_dt(1),
            summary="Research lead",
            page_path="entities/alice.md",
            page_commit="abc123",
            attrs={"role": "PI"},
        )
        stored = backend.upsert_entity(
            ent, aliases=["Ali", "Dr. Alice Smith", "Alice-S!", "   "]
        )
        assert stored.aliases == ("ali", "alice s", "dr alice smith")

        fetched = backend.get_entity("ent-alice", user_id="u1")
        assert fetched == stored
        assert fetched.attrs == {"role": "PI"}
        assert fetched.page_commit == "abc123"

        # Re-upsert replaces previous aliases completely.
        updated = backend.upsert_entity(
            Entity(
                id="ent-alice",
                user_id="u1",
                type="person",
                name="Alice Chen",
                t_valid=_dt(1),
                aliases=("A. Chen",),
            )
        )
        assert updated.aliases == ("a chen", "alice chen")
        assert backend.find_entities("Ali", user_id="u1") == []
        assert [e.id for e in backend.find_entities("a. chen!", user_id="u1")] == ["ent-alice"]
        assert [e.id for e in backend.find_entities("  ent-alice ", user_id="u1")] == ["ent-alice"]
        assert backend.find_entities("alice", user_id="u1") == []
        assert backend.find_entities("   ", user_id="u1") == []
    finally:
        backend.close()


def test_multi_user_isolation_for_entities_and_edges():
    backend = SqliteBackend()
    try:
        e_alice = Entity(
            id="shared-id", user_id="alice", type="person", name="Alice One", t_valid=_dt(1)
        )
        e_bob = Entity(
            id="shared-id", user_id="bob", type="org", name="Bob Corp", t_valid=_dt(1)
        )
        backend.upsert_entity(e_alice, aliases=["shared alias"])
        backend.upsert_entity(e_bob, aliases=["shared alias"])

        assert backend.get_entity("shared-id", user_id="alice").name == "Alice One"
        assert backend.get_entity("shared-id", user_id="bob").name == "Bob Corp"
        assert [e.name for e in backend.find_entities("shared alias", user_id="alice")] == ["Alice One"]
        assert [e.name for e in backend.find_entities("shared alias", user_id="bob")] == ["Bob Corp"]
        assert backend.alias_map(user_id="alice") == {
            "alice one": ["shared-id"],
            "shared alias": ["shared-id"],
        }

        edge_a = Edge(
            id="edge-1", user_id="alice", src_id="shared-id", predicate="likes",
            dst_id="tea", origin="stated", t_valid=_dt(1), recorded_at=_dt(1),
        )
        edge_b = Edge(
            id="edge-1", user_id="bob", src_id="shared-id", predicate="owns",
            dst_id="coffee", origin="observed", t_valid=_dt(1), recorded_at=_dt(1),
        )
        backend.upsert_edge(edge_a)
        backend.upsert_edge(edge_b)

        assert backend.get_edge("edge-1", user_id="alice").predicate == "likes"
        assert backend.get_edge("edge-1", user_id="bob").predicate == "owns"

        # Closing alice's entity must not affect bob's entity or edge with same ids.
        assert backend.close_entity("shared-id", user_id="alice", t_invalid=_dt(5)) is True
        assert backend.get_entity("shared-id", user_id="alice") is None
        assert backend.get_entity("shared-id", user_id="bob") is not None
        assert backend.get_edge("edge-1", user_id="alice").t_invalid == _dt(5)
        assert backend.get_edge("edge-1", user_id="bob").t_invalid is None
    finally:
        backend.close()


def test_edge_upsert_preserves_recorded_at_and_temporal_tz_boundaries():
    backend = SqliteBackend()
    try:
        with pytest.raises(ValueError):
            backend.upsert_edge(
                Edge(
                    id="e1", user_id="u1", src_id="a", predicate="p", dst_id="b",
                    origin="guessed", t_valid=_dt(1), recorded_at=_dt(1),
                )
            )
        with pytest.raises(ValueError):
            backend.upsert_edge(
                Edge(
                    id="e1", user_id="u1", src_id="", predicate="p", dst_id="b",
                    origin="stated", t_valid=_dt(1), recorded_at=_dt(1),
                )
            )

        # Non-UTC offset: 2026-01-05 15:30 +05:30 == 2026-01-05 10:00 UTC.
        ist = timezone(timedelta(hours=5, minutes=30))
        t_valid_ist = datetime(2026, 1, 5, 15, 30, tzinfo=ist)
        first = backend.upsert_edge(
            Edge(
                id="e1", user_id="u1", src_id="a", predicate="works_at", dst_id="b",
                origin="stated", t_valid=t_valid_ist, recorded_at=_dt(2, 12),
                confidence=0.7, attrs={"city": "NYC"},
            )
        )
        assert first.t_valid == datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        assert first.recorded_at == _dt(2, 12)

        # Re-upsert with different recorded_at must preserve original recorded_at.
        second = backend.upsert_edge(
            Edge(
                id="e1", user_id="u1", src_id="a", predicate="works_at", dst_id="b",
                origin="observed", t_valid=t_valid_ist, recorded_at=_dt(10, 8),
                confidence=0.95, attrs={"city": "SF"},
            )
        )
        assert second.recorded_at == _dt(2, 12)
        assert second.origin == "observed"
        assert second.confidence == 0.95
        assert second.attrs == {"city": "SF"}

        # Close at 2026-01-08 02:00 -04:00 == 2026-01-08 06:00 UTC.
        edt_minus4 = timezone(timedelta(hours=-4))
        t_close = datetime(2026, 1, 8, 2, 0, tzinfo=edt_minus4)
        assert backend.close_edge("e1", user_id="u1", t_invalid=t_close) is True
        assert backend.close_edge("e1", user_id="u1", t_invalid=_dt(9)) is False

        # Exact boundary checks: t_valid <= t and (t_invalid is None or t_invalid > t)
        assert backend.list_edges(user_id="u1", as_of=datetime(2026, 1, 5, 9, 59, tzinfo=UTC)) == []
        assert len(backend.list_edges(user_id="u1", as_of=datetime(2026, 1, 5, 10, 0, tzinfo=UTC))) == 1
        assert len(backend.list_edges(user_id="u1", as_of=datetime(2026, 1, 8, 5, 59, tzinfo=UTC))) == 1
        assert backend.list_edges(user_id="u1", as_of=datetime(2026, 1, 8, 6, 0, tzinfo=UTC)) == []
        assert len(backend.list_edges(user_id="u1", include_closed=True)) == 1
        assert backend.list_edges(user_id="u1") == []
    finally:
        backend.close()


def test_close_entity_cascades_open_edges_preserving_closed_edges():
    backend = SqliteBackend()
    try:
        for eid in ("a", "b", "c"):
            backend.upsert_entity(
                Entity(id=eid, user_id="u1", type="person", name=eid.upper(), t_valid=_dt(1))
            )
        backend.upsert_edge(
            Edge(
                id="e-open-out", user_id="u1", src_id="a", predicate="knows", dst_id="b",
                origin="stated", t_valid=_dt(1), recorded_at=_dt(1),
            )
        )
        backend.upsert_edge(
            Edge(
                id="e-open-in", user_id="u1", src_id="c", predicate="knows", dst_id="a",
                origin="stated", t_valid=_dt(1), recorded_at=_dt(1),
            )
        )
        backend.upsert_edge(
            Edge(
                id="e-already-closed", user_id="u1", src_id="a", predicate="knows", dst_id="c",
                origin="stated", t_valid=_dt(1), recorded_at=_dt(1), t_invalid=_dt(3),
            )
        )
        backend.upsert_edge(
            Edge(
                id="e-unrelated", user_id="u1", src_id="b", predicate="knows", dst_id="c",
                origin="stated", t_valid=_dt(1), recorded_at=_dt(1),
            )
        )

        assert backend.close_entity("a", user_id="u1", t_invalid=_dt(10)) is True
        # Closing again is a no-op returning False.
        assert backend.close_entity("a", user_id="u1", t_invalid=_dt(11)) is False

        assert backend.get_edge("e-open-out", user_id="u1").t_invalid == _dt(10)
        assert backend.get_edge("e-open-in", user_id="u1").t_invalid == _dt(10)
        assert backend.get_edge("e-already-closed", user_id="u1").t_invalid == _dt(3)
        assert backend.get_edge("e-unrelated", user_id="u1").t_invalid is None

        # Historical view before close still shows entity and both cascaded edges.
        assert backend.get_entity("a", user_id="u1", as_of=_dt(5)) is not None
        assert len(backend.list_entities(user_id="u1")) == 2
        assert len(backend.list_entities(user_id="u1", include_closed=True)) == 3
        assert len(backend.list_entities(user_id="u1", entity_type="person", as_of=_dt(5))) == 3
    finally:
        backend.close()


def test_neighbors_cte_directions_cycles_diamonds_as_of_and_limits():
    backend = SqliteBackend()
    try:
        with pytest.raises(ValueError):
            backend.neighbors("a", user_id="u1", direction="sideways")
        with pytest.raises(ValueError):
            backend.neighbors("a", user_id="u1", depth=3)
        with pytest.raises(ValueError):
            backend.neighbors("a", user_id="u1", depth=0)
        with pytest.raises(ValueError):
            backend.neighbors("a", user_id="u1", depth=True)  # type: ignore[arg-type]

        def add_e(eid: str, src: str, pred: str, dst: str, day: int, t_invalid: datetime | None = None, user: str = "u1"):
            backend.upsert_edge(
                Edge(
                    id=eid, user_id=user, src_id=src, predicate=pred, dst_id=dst,
                    origin="observed", t_valid=_dt(day), recorded_at=_dt(day),
                    t_invalid=t_invalid,
                )
            )

        # Diamond + cycle + cross-edge + other-user noise:
        # a -> b (e1, day 2), a -> c (e2, day 3)
        # b -> d (e3, day 4), c -> d (e4, day 5)
        # b -> c (e5, day 6), d -> a (e6, day 7 — cycle back to start)
        add_e("e1", "a", "links", "b", 2)
        add_e("e2", "a", "links", "c", 3)
        add_e("e3", "b", "links", "d", 4)
        add_e("e4", "c", "links", "d", 5)
        add_e("e5", "b", "links", "c", 6)
        add_e("e6", "d", "links", "a", 7)
        add_e("e-other-pred", "a", "blocks", "z", 8)
        add_e("e-other-user", "a", "links", "x", 9, user="u2")

        # Depth 1 out with predicate filter
        d1_out = backend.neighbors("a", user_id="u1", predicate="links", direction="out", depth=1)
        assert [(h, e.id) for h, e in d1_out] == [(1, "e2"), (1, "e1")]

        # Depth 1 in
        d1_in = backend.neighbors("a", user_id="u1", direction="in", depth=1)
        assert [(h, e.id) for h, e in d1_in] == [(1, "e6")]

        # Depth 2 out with predicate="links":
        # - depth 1: e2 (day 3), e1 (day 2)
        # - depth 2: e5 (day 6), e4 (day 5), e3 (day 4)
        # Note e6 (d -> a) re-enters start node 'a' and must NOT loop back.
        d2_out = backend.neighbors("a", user_id="u1", predicate="links", direction="out", depth=2)
        assert [(h, e.id) for h, e in d2_out] == [
            (1, "e2"),
            (1, "e1"),
            (2, "e5"),
            (2, "e4"),
            (2, "e3"),
        ]

        # Depth 2 both on diamond + cross-edge e5 (reachable from b and c at depth 2):
        # No duplicate edges; MIN(depth) wins; e6 is depth 1 via 'in'.
        d2_both = backend.neighbors("a", user_id="u1", predicate="links", direction="both", depth=2)
        ids_both = [e.id for _, e in d2_both]
        assert len(ids_both) == len(set(ids_both))
        assert set(ids_both) == {"e1", "e2", "e3", "e4", "e5", "e6"}
        assert dict((e.id, h) for h, e in d2_both) == {
            "e1": 1, "e2": 1, "e6": 1,
            "e3": 2, "e4": 2, "e5": 2,
        }

        # Limit enforcement
        limited = backend.neighbors("a", user_id="u1", predicate="links", direction="out", depth=2, limit=3)
        assert [(h, e.id) for h, e in limited] == [(1, "e2"), (1, "e1"), (2, "e5")]
        assert backend.neighbors("a", user_id="u1", limit=0) == []

        # as_of enforced on BOTH hop 1 and hop 2:
        # Make a path x1 -> x2 -> x3 where hop 1 closes at day 5 and hop 2 starts at day 2.
        add_e("h1", "x1", "p", "x2", 1, t_invalid=_dt(5))
        add_e("h2", "x2", "p", "x3", 2)
        add_e("h3", "x1", "p", "x4", 1)
        add_e("h4", "x4", "p", "x5", 6)

        # At day 3: h1 valid, h2 valid, h3 valid, h4 NOT valid yet.
        at_day3 = backend.neighbors("x1", user_id="u1", direction="out", depth=2, as_of=_dt(3))
        assert {e.id for _, e in at_day3} == {"h1", "h2", "h3"}

        # At day 7 (or current as_of=None): h1 is closed so h2 is unreachable! h3 and h4 are reachable.
        at_day7 = backend.neighbors("x1", user_id="u1", direction="out", depth=2, as_of=_dt(7))
        assert {e.id for _, e in at_day7} == {"h3", "h4"}
        at_now = backend.neighbors("x1", user_id="u1", direction="out", depth=2)
        assert {e.id for _, e in at_now} == {"h3", "h4"}
    finally:
        backend.close()


def test_mentions_lifecycle_tombstone_delete_and_missing():
    backend = SqliteBackend()
    try:
        m1 = backend.add_memory(content="Alice met Bob.", user_id="u1", t_valid=_dt(1))
        m2 = backend.add_memory(content="Bob went home.", user_id="u1", t_valid=_dt(2))
        m3 = backend.add_memory(content="Charlie arrived.", user_id="u1", t_valid=_dt(3))
        m_other = backend.add_memory(content="Other user.", user_id="u2", t_valid=_dt(1))

        assert [m.id for m in backend.memories_without_mentions(user_id="u1")] == [m1.id, m2.id, m3.id]

        with pytest.raises(ValueError):
            backend.add_mentions([Mention(memory_id=m1.id, entity_id="bob", user_id="u1", method="guess")])

        inserted = backend.add_mentions(
            [
                Mention(memory_id=m1.id, entity_id="alice", user_id="u1", method="alias"),
                Mention(memory_id=m1.id, entity_id="bob", user_id="u1", method="model"),
                Mention(memory_id=m1.id, entity_id="bob", user_id="u1", method="curator"),
                Mention(memory_id=m2.id, entity_id="bob", user_id="u1", method="curator"),
                Mention(memory_id=m_other.id, entity_id="bob", user_id="u2", method="alias"),
            ]
        )
        assert inserted == 4
        # Re-adding is strictly idempotent.
        assert backend.add_mentions([Mention(memory_id=m1.id, entity_id="bob", user_id="u1", method="alias")]) == 0

        assert [m.id for m in backend.memories_without_mentions(user_id="u1")] == [m3.id]
        assert backend.mentions_for_entity("bob", user_id="u1", live_only=True) == [m1.id, m2.id]
        assert [mn.entity_id for mn in backend.mentions_for_memory(m1.id, user_id="u1")] == ["alice", "bob"]
        assert backend.mentions_for_memory(m1.id, user_id="u2") == []

        # Tombstoning hides from live_only=True and graph_stats, but retains audit rows for live_only=False.
        assert backend.transition_state(m2.id, new_state="tombstoned") is True
        assert backend.mentions_for_entity("bob", user_id="u1", live_only=True) == [m1.id]
        assert set(backend.mentions_for_entity("bob", user_id="u1", live_only=False)) == {m1.id, m2.id}

        # Hard delete removes mentions completely.
        assert backend.delete(m1.id) is True
        assert backend.mentions_for_entity("bob", user_id="u1", live_only=True) == []
        assert backend.mentions_for_entity("bob", user_id="u1", live_only=False) == [m2.id]
        assert backend.delete(m2.id) is True
        assert backend.mentions_for_entity("bob", user_id="u1", live_only=False) == []
    finally:
        backend.close()


def test_flags_graph_stats_and_clear():
    backend = SqliteBackend()
    try:
        with pytest.raises(ValueError):
            backend.add_flag(user_id="u1", target_kind="unknown", note="bad")
        with pytest.raises(ValueError):
            backend.add_flag(user_id="", target_kind="entity", note="bad")

        ist = timezone(timedelta(hours=5, minutes=30))
        f1 = backend.add_flag(
            user_id="u1",
            target_kind="entity",
            target_id="ent-1",
            note="Possible duplicate",
            source="curator",
            created_at=datetime(2026, 1, 2, 15, 30, tzinfo=ist),
        )
        f2 = backend.add_flag(
            user_id="u1",
            target_kind="missing",
            note="Missing employer",
            created_at=_dt(3),
        )
        f_other = backend.add_flag(user_id="u2", target_kind="edge", note="u2 flag", created_at=_dt(1))

        assert f1.id != f2.id
        assert f1.created_at == datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
        assert [f.id for f in backend.list_flags(user_id="u1")] == [f1.id, f2.id]

        # Wrong user cannot resolve f1.
        assert backend.resolve_flag(f1.id, user_id="u2", resolution="nope") is False
        assert backend.resolve_flag(f1.id, user_id="u1", resolution="merged", resolved_at=_dt(4)) is True
        # Re-resolving returns False.
        assert backend.resolve_flag(f1.id, user_id="u1", resolution="again") is False

        assert [f.id for f in backend.list_flags(user_id="u1")] == [f2.id]
        all_u1 = backend.list_flags(user_id="u1", include_resolved=True)
        assert [f.id for f in all_u1] == [f1.id, f2.id]
        assert all_u1[0].resolution == "merged"
        assert all_u1[0].resolved_at == _dt(4)

        # Populate entities, edges, mentions and verify graph_stats.
        backend.upsert_entity(Entity(id="e1", user_id="u1", type="person", name="A", t_valid=_dt(1)))
        backend.upsert_entity(
            Entity(id="e2", user_id="u1", type="person", name="B", t_valid=_dt(1), t_invalid=_dt(2))
        )
        backend.upsert_edge(
            Edge(id="ed1", user_id="u1", src_id="e1", predicate="p", dst_id="e2", origin="stated", t_valid=_dt(1), recorded_at=_dt(1))
        )
        backend.upsert_edge(
            Edge(id="ed2", user_id="u1", src_id="e1", predicate="p", dst_id="e2", origin="stated", t_valid=_dt(1), recorded_at=_dt(1), t_invalid=_dt(2))
        )
        mem_live = backend.add_memory(content="live", user_id="u1")
        mem_tomb = backend.add_memory(content="tomb", user_id="u1")
        backend.add_mentions(
            [
                Mention(memory_id=mem_live.id, entity_id="e1", user_id="u1", method="alias"),
                Mention(memory_id=mem_tomb.id, entity_id="e1", user_id="u1", method="alias"),
            ]
        )
        backend.transition_state(mem_tomb.id, new_state="tombstoned")

        assert backend.graph_stats(user_id="u1") == GraphStats(
            entities=1,
            edges=1,
            mentions=1,
            open_flags=1,
        )
        assert backend.graph_stats(user_id="u2") == GraphStats(
            entities=0,
            edges=0,
            mentions=0,
            open_flags=1,
        )

        # clear() wipes both memories and all 5 graph tables.
        backend.clear()
        assert backend.graph_stats(user_id="u1") == GraphStats()
        assert backend.graph_stats(user_id="u2") == GraphStats()
        assert backend.list_entities(user_id="u1", include_closed=True) == []
        assert backend.list_edges(user_id="u1", include_closed=True) == []
        assert backend.list_flags(user_id="u1", include_resolved=True) == []
        assert backend.mentions_for_entity("e1", user_id="u1", live_only=False) == []
        assert backend.list_memories(user_id="u1", include_tombstoned=True) == []
        assert f_other.id is not None
    finally:
        backend.close()
