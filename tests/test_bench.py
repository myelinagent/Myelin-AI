"""The headless benchmark harness, exercised on a tiny synthetic dataset."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from bench import longmemeval as lme
from bench.llm import ChatClient, LLMError
from bench.systems import SYSTEMS, Turn


def _entry(qid: str, qtype: str, question: str, answer: str, gold: list[str]) -> dict:
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "question_date": "2023/06/01 (Thu) 09:00",
        "answer": answer,
        "answer_session_ids": gold,
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_dates": [
            "2023/05/01 (Mon) 10:00",
            "2023/05/02 (Tue) 10:00",
            "2023/05/03 (Wed) 10:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I adopted a greyhound named Comet."},
                {"role": "assistant", "content": "Greyhounds are gentle dogs."},
            ],
            [
                {"role": "user", "content": "Any tips for baking sourdough bread?"},
                {"role": "assistant", "content": "Use a hot dutch oven."},
            ],
            [
                {"role": "user", "content": "My sister Priya moved to Lisbon."},
                {"role": "assistant", "content": "Lisbon has lovely weather."},
            ],
        ],
    }


@pytest.fixture
def dataset(tmp_path):
    entries = [
        _entry("q1", "single-session-user", "What is my greyhound called?", "Comet", ["s1"]),
        _entry("q2", "multi-session", "Where did Priya move, and what dog do I have?",
               "Lisbon; a greyhound", ["s1", "s3"]),
        _entry("q3_abs", "single-session-user", "What is my cat called?",
               "No cat was mentioned.", ["s1"]),
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(entries))
    return path


class _FakeLLM(BaseHTTPRequestHandler):
    """Answers every question 'Comet' and every judge prompt 'yes'."""

    fail_first = 0
    seen_auth: ClassVar[list[str | None]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_auth.append(self.headers.get("Authorization"))
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.end_headers()
            return
        prompt = body["messages"][0]["content"]
        text = "yes" if "Answer yes or no only." in prompt else "Comet"
        payload = json.dumps(
            {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def llm(monkeypatch):
    _FakeLLM.fail_first = 0
    _FakeLLM.seen_auth = []
    server = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "MYELIN_BENCH_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1"
    )
    monkeypatch.delenv("MYELIN_BENCH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("bench.llm.time.sleep", lambda seconds: None)
    yield _FakeLLM
    server.shutdown()
    server.server_close()


def test_parse_date():
    assert lme.parse_date("2023/05/20 (Sat) 02:21") == datetime(
        2023, 5, 20, 2, 21, tzinfo=UTC
    )


def test_session_recall_distinguishes_any_from_all():
    recall = lme.session_recall(["a", "x", "x", "x", "x", "b"], ["a", "b"])
    assert (recall["any@5"], recall["all@5"]) == (1.0, 0.0)
    assert recall["all@10"] == 1.0


def test_judge_prompt_follows_question_type():
    base = {"question": "Q", "answer": "A"}
    temporal = lme.judge_prompt(
        {**base, "question_id": "x", "question_type": "temporal-reasoning"}, "H"
    )
    assert "off-by-one" in temporal
    abstain = lme.judge_prompt(
        {**base, "question_id": "x_abs", "question_type": "temporal-reasoning"}, "H"
    )
    assert "unanswerable" in abstain
    with pytest.raises(ValueError, match="unknown question_type"):
        lme.judge_prompt({**base, "question_id": "x", "question_type": "nope"}, "H")


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_systems_rank_the_relevant_turn_first(name):
    system = SYSTEMS[name]()
    system.reset()
    when = datetime(2023, 5, 1, tzinfo=UTC)
    system.ingest(
        [
            Turn("s1", when, 0, "user", "I adopted a greyhound named Comet."),
            Turn("s2", when, 0, "user", "Any tips for baking sourdough bread?"),
        ]
    )
    hits = system.retrieve("What is my greyhound called?", 5)
    assert hits[0].session_id == "s1"
    system.close()


def test_myelin_system_stamps_memories_with_the_session_date():
    system = SYSTEMS["myelin"]()
    system.reset()
    when = datetime(2023, 5, 1, 10, 0, tzinfo=UTC)
    system.ingest([Turn("s1", when, 0, "user", "I adopted a greyhound.")])
    (memory,) = system._client._backend.list_memories(user_id="bench")
    assert memory.t_valid == when
    system.close()


def test_retrieval_only_run_needs_no_model(dataset, tmp_path, capsys):
    out = tmp_path / "run"
    code = lme.main(["run", "--system", "myelin", "--data", str(dataset), "--out", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["answered"] == 3
    assert summary["judged"] == 0
    # The abstention question is excluded from retrieval recall.
    assert summary["session_recall"]["questions"] == 2
    assert summary["session_recall"]["any@5"] == 1.0
    lme.main(["report", str(out)])
    table = capsys.readouterr().out
    assert "Recall, all evidence sessions @10" in table
    assert "QA accuracy" not in table


def test_full_run_answers_judges_and_resumes(dataset, tmp_path, llm):
    out = tmp_path / "run"
    argv = ["run", "--system", "bm25", "--data", str(dataset), "--out", str(out),
            "--answerer", "fake-answerer", "--judge", "fake-judge", "--workers", "2"]
    assert lme.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["judged"] == 3
    assert summary["accuracy"]["overall"] == 1.0
    assert summary["accuracy"]["abstention"] == 1.0
    assert summary["cost"]["answerer_prompt_tokens"] == 300
    hypotheses = [json.loads(line) for line in (out / "hypotheses.jsonl").read_text().splitlines()]
    assert {h["question_id"] for h in hypotheses} == {"q1", "q2", "q3_abs"}
    assert all(h["hypothesis"] == "Comet" for h in hypotheses)
    # No key configured: no Authorization header is sent.
    assert set(llm.seen_auth) == {None}

    calls = len(llm.seen_auth)
    assert lme.main(argv) == 0
    assert len(llm.seen_auth) == calls, "a finished run must not call the model again"


def test_run_refuses_to_mix_settings_in_one_directory(dataset, tmp_path):
    out = tmp_path / "run"
    base = ["run", "--data", str(dataset), "--out", str(out)]
    assert lme.main([*base, "--system", "myelin"]) == 0
    assert lme.main([*base, "--system", "bm25"]) == 2


def test_oracle_gives_the_reader_only_evidence_sessions(dataset, tmp_path, llm):
    out = tmp_path / "run"
    assert lme.main(["run", "--system", "oracle", "--data", str(dataset), "--out", str(out),
                     "--answerer", "fake", "--limit", "1"]) == 0
    row = json.loads((out / "answers.jsonl").read_text().splitlines()[0])
    assert row["context_turns"] == 2
    assert "recall" not in row


def test_chat_client_retries_then_succeeds(llm):
    llm.fail_first = 2
    completion = ChatClient("m").complete("What is my greyhound called?", max_tokens=5)
    assert completion.text == "Comet"
    assert len(llm.seen_auth) == 3


def test_chat_client_gives_up_after_retries(llm):
    llm.fail_first = 99
    with pytest.raises(LLMError, match="HTTP 503"):
        ChatClient("m", max_retries=1).complete("hi", max_tokens=5)
    assert len(llm.seen_auth) == 2


def test_chat_client_sends_configured_key(llm, monkeypatch):
    monkeypatch.setenv("MYELIN_BENCH_API_KEY", "token-for-test")
    ChatClient("m").complete("hi", max_tokens=5)
    assert llm.seen_auth == ["Bearer token-for-test"]


def test_token_budget_decides_what_the_answerer_sees(dataset, tmp_path, capsys):
    tight, roomy = tmp_path / "tight", tmp_path / "roomy"
    base = ["run", "--system", "myelin", "--data", str(dataset)]
    assert lme.main([*base, "--out", str(tight), "--token-budget", "1"]) == 0
    assert lme.main([*base, "--out", str(roomy), "--token-budget", "5000"]) == 0
    rows = {
        name: [json.loads(line) for line in (path / "answers.jsonl").read_text().splitlines()]
        for name, path in (("tight", tight), ("roomy", roomy))
    }
    assert all(row["context_turns"] <= 1 for row in rows["tight"])
    assert max(row["context_turns"] for row in rows["roomy"]) > 1
    summary = json.loads((roomy / "summary.json").read_text())
    assert summary["manifest"]["token_budget"] == 5000
    assert summary["session_recall"]["all@context"] == 1.0
    # A different budget is a different run and may not reuse the directory.
    assert lme.main([*base, "--out", str(tight), "--token-budget", "5000"]) == 2
    lme.main(["report", str(tight), str(roomy)])
    assert "All evidence sessions in the answerer's context" in capsys.readouterr().out


def test_evidence_turns_are_tracked_and_long_turns_excerpted(tmp_path):
    entry = _entry("q1", "single-session-user", "What is my greyhound called?", "Comet", ["s1"])
    entry["haystack_sessions"][0][0]["has_answer"] = True
    filler = " ".join(f"Sentence {i} is about the weather." for i in range(200))
    entry["haystack_sessions"][0][1]["content"] = (
        f"{filler} Your greyhound is a gentle dog. {filler}"
    )
    data = tmp_path / "data.json"
    data.write_text(json.dumps([entry]))

    turns = lme.history_turns(entry)
    assert [t.evidence for t in turns[:2]] == [True, False]

    out = tmp_path / "run"
    assert lme.main(["run", "--system", "myelin", "--data", str(data), "--out", str(out),
                     "--token-budget", "400", "--max-turn-tokens", "50"]) == 0
    row = json.loads((out / "answers.jsonl").read_text().splitlines()[0])
    # Both greyhound turns fit only because the long reply was cut down.
    assert row["context_turns"] == 2
    assert row["context_tokens_estimate"] < 150
    assert row["recall"]["evidence_turns_all@context"] == 1.0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["manifest"]["max_turn_tokens"] == 50

    shortened = lme.within_budget(turns[:2], 400, entry["question"], 50)
    assert "greyhound is a gentle dog" in shortened[1].content
    assert len(shortened[1].content) < 400


def test_within_budget_always_keeps_one_turn():
    when = datetime(2023, 5, 1, tzinfo=UTC)
    turns = [Turn("s", when, i, "user", "x" * 400) for i in range(3)]
    assert len(lme.within_budget(turns, 1)) == 1
    assert len(lme.within_budget(turns, 250)) == 2
    assert lme.within_budget([], 100) == []


@pytest.mark.parametrize(
    ("response", "graded"),
    [
        ("Comet", "Comet"),
        ("Notes: adopted in May.\nAnswer: Comet", "Comet"),
        ("notes\n**Answer:** Comet", "Comet"),
        ("Answer: first\nmore notes\nAnswer: second", "second"),
        ("Notes only.\nAnswer:", "Notes only.\nAnswer:"),
    ],
)
def test_only_the_final_answer_is_graded(response, graded):
    assert lme.final_answer(response) == graded


def test_prompt_choice_is_recorded_and_pinned(dataset, tmp_path, llm):
    out = tmp_path / "run"
    base = ["run", "--system", "oracle", "--data", str(dataset), "--out", str(out),
            "--answerer", "fake", "--limit", "1"]
    assert lme.main([*base, "--prompt", "direct"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["prompt_version"] == "myelin-direct-v2"
    row = json.loads((out / "answers.jsonl").read_text().splitlines()[0])
    assert row["hypothesis"] == row["response"] == "Comet"
    # The same directory cannot be continued with another prompt.
    assert lme.main([*base, "--prompt", "v1"]) == 2


def test_extracted_facts_are_indexed_and_optionally_shown(dataset, tmp_path, llm, monkeypatch):
    from bench import extract_facts as ef

    entries = json.loads(dataset.read_text())
    cache = tmp_path / "facts.jsonl"
    # Tag the first session's first turn with a fact that shares no word with it.
    key = ef.session_key(entries[0]["haystack_dates"][0], entries[0]["haystack_sessions"][0])
    cache.write_text(json.dumps({"key": key, "facts": {"0": ["The user owns a sighthound."]}}) + "\n")
    assert ef.load_cache(cache) == {key: {0: ["The user owns a sighthound."]}}

    monkeypatch.setattr(lme, "FACTS", {})
    data = tmp_path / "one.json"
    entries[0]["question"] = "Do I own a sighthound?"
    data.write_text(json.dumps(entries[:1]))

    plain = tmp_path / "plain"
    assert lme.main(["run", "--system", "myelin", "--data", str(data), "--out", str(plain)]) == 0
    with_facts = tmp_path / "facts"
    assert lme.main(["run", "--system", "myelin", "--data", str(data), "--out", str(with_facts),
                     "--facts", str(cache), "--show-facts", "--answerer", "fake"]) == 0
    rows = {
        name: json.loads((path / "answers.jsonl").read_text().splitlines()[0])
        for name, path in (("plain", plain), ("facts", with_facts))
    }
    assert rows["plain"]["context_turns"] == 0          # nothing matched "sighthound"
    assert rows["facts"]["recall"]["any@5"] == 1.0      # found through the fact
    manifest = json.loads((with_facts / "manifest.json").read_text())
    assert manifest["facts"] == "facts.jsonl" and manifest["show_facts"] is True
    turns = lme.history_turns(entries[0])
    assert turns[0].facts == ("The user owns a sighthound.",)
    monkeypatch.setattr(lme, "SHOW_FACTS", True)
    assert "(noted: The user owns a sighthound.)" in lme.render_history(turns[:1])


def test_fact_extraction_script_is_resumable(dataset, tmp_path, monkeypatch):
    from bench import extract_facts as ef

    calls: list[str] = []

    class FakeClient:
        def __init__(self, model):
            pass

        def complete(self, prompt, *, max_tokens):
            from bench.llm import Completion

            calls.append(prompt)
            return Completion('{"0": ["The user said something."]}', 10, 5, 0.0)

    monkeypatch.setattr(ef, "ChatClient", FakeClient)
    out = tmp_path / "facts.jsonl"
    argv = ["--data", str(dataset), "--out", str(out), "--model", "fake", "--workers", "2"]
    assert ef.main(argv) == 0
    # Three questions share the same three sessions: three calls, not nine.
    assert len(calls) == 3
    assert len(ef.load_cache(out)) == 3
    assert ef.main(argv) == 0
    assert len(calls) == 3
