"""bench.compare and bench.gateway, offline: a fake system, no network."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench import compare, gateway, lifetime
from bench.systems import Turn
from tests.test_bench_locomo import SAMPLE


class FactStore(compare.Adapter):
    """Stands in for a system that stores facts, not turns: a turn that
    is only a reaction yields nothing, and a turn naming a subject already stored is merged
    into that fact. State is one JSON file, so it can be reopened."""

    name = "fake"
    label = "Fake fact store"
    calls = 0

    def open(self) -> None:
        self._path = self.store / "facts.json"
        self._facts = json.loads(self._path.read_text()) if self._path.exists() else {}

    def ingest(self, batch):
        type(self).calls += 1
        changed = []
        for turn_id, turn in batch:
            if not turn.content.startswith("I "):  # nothing worth keeping
                continue
            subject = turn.role
            fact = self._facts.setdefault(subject, {"text": "", "sources": []})
            fact["text"] += " " + turn.content
            fact["sources"].append(turn_id)
            changed.append(compare.Item(subject, tuple(fact["sources"]), fact["text"]))
        self._path.write_text(json.dumps(self._facts))
        return changed

    def search(self, question, k):
        asked = set(question.lower().replace("?", "").replace("'s", "").split())
        ranked = sorted(
            self._facts.items(),
            key=lambda pair: -len(asked & {pair[0].lower(), *pair[1]["text"].lower().split()}),
        )
        return [compare.Item(key, tuple(f["sources"]), f["text"]) for key, f in ranked[:k]]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setitem(compare.ADAPTERS, "fake", FactStore)
    monkeypatch.setattr(FactStore, "calls", 0)
    return FactStore


def _data(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([SAMPLE]))
    return str(path)


def test_an_item_counts_for_every_turn_it_derives_from():
    merged = compare.Item("a", ("D1:1", "D2:1"))
    other = compare.Item("b", ("D1:2",))
    assert compare.score(["D1:1", "D2:1"], [merged])["top5"]
    assert not compare.score(["D1:1", "D1:2"], [merged])["top5"]
    assert compare.score(["D1:1", "D1:2"], [merged, other])["top5"]
    # Depth is counted in items returned, not in turns.
    ranked = [compare.Item(str(n), ()) for n in range(5)] + [other]
    hit = compare.score(["D1:2"], ranked)
    assert not hit["top5"] and hit["top10"] and hit["top20"]


def test_a_fact_written_from_a_context_turn_can_be_credited_to_it(tmp_path, fake, capsys):
    order = ["D1:1", "D1:2", "D2:1"]
    texts = {
        "D1:1": "Ana: I adopted a beagle named Pickle.",
        "D1:2": "Ben: Lovely!",
        "D2:1": "Ben: I started pottery classes last week.",
    }
    restated = "Ana adopted a beagle named Pickle"
    # Stored while writing Ben's reply, but the words are Ana's.
    assert compare.with_context(restated, ["D1:2"], order, texts, 10) == ["D1:2", "D1:1"]
    # Not when the source holds the words itself, nor outside the window.
    assert compare.with_context(restated, ["D1:1"], order, texts, 10) == ["D1:1"]
    assert compare.with_context(restated, ["D2:1"], order, texts, 1) == ["D2:1"]
    assert compare.with_context(restated, ["D1:2"], order, texts, 0) == ["D1:2"]

    # A system with a context window gets both readings in the table.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fake, "context_turns", 10)
    try:
        args = ["--system", "fake", "--data", _data(tmp_path), "--out", str(tmp_path / "o"),
                "--gateway", "", "--seeds", "1"]
        assert compare.main(args) == 0
    finally:
        monkeypatch.undo()
    assert "Also crediting a context turn" in capsys.readouterr().out
    summary = compare.summarize(
        [tmp_path / "o" / "fake" / "conv-1"], [0], {"conv-1": SAMPLE}, window=10)
    assert summary["context_top5"] >= summary["top5"]


def test_a_session_level_item_is_given_the_turn_it_reads_like():
    session = {"D1:1": "Ana: I adopted a beagle named Pickle.", "D1:2": "Ben: Lovely!"}
    assert compare.closest_turns("Ana adopted a beagle", session) == ("D1:1",)
    assert compare.closest_turns("Something about tax returns", session) == ()
    both = {"D1:1": "Ana: the beagle", "D1:2": "Ben: the beagle"}
    assert compare.closest_turns("a beagle", both) == ("D1:1", "D1:2")
    assert "guessed" in compare.ADAPTERS["mem0-session"].label


def test_units_are_turns_or_whole_sessions():
    turns = lifetime.locomo_timeline(SAMPLE)[0][0]
    ids = ["D1:1", "D1:2"]
    assert [len(u) for u in compare.units(turns, ids, "turn")] == [1, 1]
    assert [len(u) for u in compare.units(turns, ids, "session")] == [2]
    assert compare.units(turns, ids, "turn")[1][0][0] == "D1:2"


def test_replay_asks_the_same_schedule_as_the_lifetime_benchmark(tmp_path, fake):
    adapter = fake(tmp_path, "", "", "")
    adapter.open()
    compare.replay(adapter, SAMPLE, [0, 1], tmp_path)
    asked = compare._rows(tmp_path / "asked.jsonl")
    _, ids, questions = lifetime.locomo_timeline(SAMPLE)
    for seed in (0, 1):
        due = lifetime.schedule(ids, questions, seed)
        want = sorted((step, qa["question_id"], qa["lag"]) for step, qs in due.items() for qa in qs)
        got = sorted((r["step"], r["question_id"], r["lag"]) for r in asked if r["seed"] == seed)
        assert got == want
    # The beagle fact derives from Ana's turn; "Lovely!" produced no fact.
    beagle = next(r for r in asked if r["evidence"] == ["D1:1"])
    assert beagle["top5"] and beagle["written"]
    writes = compare._rows(tmp_path / "writes.jsonl")
    assert [w["turns"] for w in writes] == [["D1:1"], ["D1:2"], ["D2:1"]]
    assert writes[1]["items"] == []


def test_evidence_with_no_stored_item_is_reported_not_hidden(tmp_path, fake):
    sample = json.loads(json.dumps(SAMPLE))
    sample["qa"].append(
        {"question": "What did Ben say?", "answer": "Lovely", "evidence": ["D1:2"], "category": 4}
    )
    adapter = fake(tmp_path, "", "", "")
    adapter.open()
    compare.replay(adapter, sample, [0], tmp_path)
    row = next(r for r in compare._rows(tmp_path / "asked.jsonl") if r["evidence"] == ["D1:2"])
    assert not row["written"] and not row["top20"]
    summary = compare.summarize([tmp_path], [0])
    assert summary["questions"] == 3
    assert summary["written"] == pytest.approx(200 / 3)


def test_an_interrupted_run_continues_without_writing_twice(tmp_path, fake, monkeypatch):
    out = tmp_path / "out"
    args = ["--system", "fake", "--data", _data(tmp_path), "--out", str(out), "--gateway", "",
            "--seeds", "2"]
    folder = out / "fake" / "conv-1"

    # No time at all: nothing is written and the exit code says so.
    assert compare.main([*args, "--max-minutes", "1e-9"]) == 3
    assert fake.calls == 0 and not compare.finished(folder, [0, 1], 6)

    # Time for one write, then the deadline passes.
    adapter = fake(folder / "store", "", "", "")
    adapter.open()
    ticks = iter([0.0])
    with monkeypatch.context() as patch:
        patch.setattr(compare.time, "monotonic", lambda: next(ticks, 5.0))
        with pytest.raises(compare.Incomplete):
            compare.replay(adapter, SAMPLE, [0, 1], folder, deadline=1.0)
    assert fake.calls == 1 and not compare.finished(folder, [0, 1], 6)

    assert compare.main(args) == 0
    assert fake.calls == 3  # the two turns that were left, not all three again
    assert compare.finished(folder, [0, 1], 6)
    assert len(compare._rows(folder / "writes.jsonl")) == 3
    before = (folder / "asked.jsonl").read_text()
    assert compare.main(args) == 0 and fake.calls == 3
    assert (folder / "asked.jsonl").read_text() == before


def test_main_prints_ranking_and_cost_tables(tmp_path, fake, capsys):
    out = tmp_path / "out"
    args = ["--system", "fake", "myelin", "--data", _data(tmp_path), "--out", str(out),
            "--gateway", "", "--seeds", "2", "--embedder", ""]
    assert compare.main(args) == 0
    printed = capsys.readouterr().out
    assert "| Fake fact store |" in printed and "| Myelin-AI (static arm) |" in printed
    assert "All evidence in top 5" in printed and "Model calls to write" in printed
    # Myelin-AI under this harness is the lifetime benchmark's static arm.
    mine = compare.summarize([out / "myelin" / "conv-1"], [0, 1])
    theirs = [
        lifetime.summarize([lifetime.replay(SAMPLE, seed, decay=False, feedback=False)])
        for seed in (0, 1)
    ]
    assert mine["questions"] == theirs[0]["questions"] == 2
    assert mine["top5"] == sum(t["top5"] for t in theirs) / 2
    assert mine["written"] == 100.0 and mine["turns"] == 3 and mine["write_calls"] == 0
    # --report reads what is there and writes nothing.
    assert compare.main([*args, "--report"]) == 0
    assert capsys.readouterr().out == printed


def test_every_adapter_says_what_it_is():
    for name, kind in compare.ADAPTERS.items():
        assert kind.name == name and kind.label and kind.__doc__
    # A reduced mode must say so in the label people will read.
    assert "no agent" in compare.ADAPTERS["letta-passages"].label
    assert "infer=True" in compare.ADAPTERS["mem0"].label


def test_gateway_reads_tags_and_usage():
    assert gateway.split_tag("/t/mem0.conv-1/v1/chat/completions") == (
        "mem0.conv-1", "chat/completions")
    assert gateway.split_tag("/v1/embeddings") == ("", "embeddings")
    assert gateway.split_tag("/t/x/usage") == ("x", "usage")
    plain = json.dumps({"usage": {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.5}})
    assert gateway.usage_of(plain.encode()) == {
        "prompt_tokens": 7.0, "completion_tokens": 2.0, "cost": 0.5}
    stream = 'data: {"choices": []}\n\ndata: {"usage": {"prompt_tokens": 3}}\n\ndata: [DONE]\n'
    assert gateway.usage_of(stream.encode())["prompt_tokens"] == 3.0
    assert gateway.usage_of(b"not json")["cost"] == 0.0


def test_gateway_embeds_locally_and_counts_what_it_forwards(monkeypatch):
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers.get("Authorization") is None  # the client's key is dropped
            body = json.dumps({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "cost": 0.25},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.delenv("MYELIN_BENCH_API_KEY", raising=False)
    upstream = HTTPServer(("127.0.0.1", 0), Upstream)
    server = gateway.serve(0, upstream=f"http://127.0.0.1:{upstream.server_address[1]}")
    monkeypatch.setattr(
        server.gateway, "embed", lambda model, texts: [[float(len(t)), 1.0] for t in texts]
    )
    for each in (upstream, server):
        threading.Thread(target=each.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, payload):
        request = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    try:
        assert compare._embed(f"{base}/t/a/v1", "minishlab/any", ["abc", "de"]) == [
            [3.0, 1.0], [2.0, 1.0]]
        packed = post("/t/a/v1/embeddings",
                      {"model": "minishlab/any", "input": "abcd", "encoding_format": "base64"})
        assert isinstance(packed["data"][0]["embedding"], str)
        before = compare._usage(base, "a")
        reply = post("/t/a/v1/chat/completions", {"model": "m", "messages": []})
        assert reply["choices"][0]["message"]["content"] == "ok"
        assert compare._spent(base, "a", before) == {
            "calls": 1.0, "prompt_tokens": 11.0, "completion_tokens": 4.0, "cost": 0.25,
            "embedded_texts": 0.0, "embedding_tokens": 0.0,
        }
        # Embeddings for any other model are bought upstream and counted apart.
        post("/t/b/v1/embeddings", {"model": "other/model", "input": ["abc"]})
        bought = compare._usage(base, "b")
        assert bought["calls"] == 0.0 and bought["embedding_tokens"] == 11.0
        assert bought["cost"] == 0.25
        assert compare._usage(base, "a")["embedded_texts"] == 3.0
        assert compare._usage(base, "someone-else")["calls"] == 0.0
    finally:
        server.shutdown()
        upstream.shutdown()


def test_fixture_turns_are_invented():
    # The tests must not carry benchmark text; the fixture is three made-up turns.
    turns = [t for batch in lifetime.locomo_timeline(SAMPLE)[0] for t in batch]
    assert len(turns) == 3 and all(isinstance(t, Turn) for t in turns)
