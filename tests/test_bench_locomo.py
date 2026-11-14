"""The LoCoMo runner: parsing, evidence coverage and judging, offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from bench import locomo

SAMPLE = {
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Ana",
        "speaker_b": "Ben",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a beagle named Pickle."},
            {"speaker": "Ben", "dia_id": "D1:2", "text": "Lovely!",
             "blip_caption": "a dog on a beach"},
        ],
        "session_2_date_time": "9:05 am on 2 June, 2023",
        "session_2": [
            {"speaker": "Ben", "dia_id": "D2:1", "text": "I started pottery classes last week."},
        ],
    },
    "qa": [
        {"question": "What breed is Ana's dog?", "answer": "beagle",
         "evidence": ["D1:1"], "category": 4},
        {"question": "When did Ben start pottery?", "answer": "late May 2023",
         "evidence": ["D2:1"], "category": 2},
        {"question": "What is Ana's cat called?", "adversarial_answer": "Pickle",
         "evidence": ["D1:1"], "category": 5},
    ],
}


def test_parse_date():
    assert locomo.parse_date("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)


def test_conversation_turns_carry_dates_ids_and_captions():
    turns = locomo.conversation_turns(SAMPLE)
    assert [t.dia_id for t in turns] == ["D1:1", "D1:2", "D2:1"]
    assert turns[0].role == "Ana"
    assert turns[1].content == "Lovely! [shares a photo: a dog on a beach]"
    assert turns[2].session_date == datetime(2023, 6, 2, 9, 5, tzinfo=UTC)
    assert turns[0].session_id != turns[2].session_id


def test_adversarial_questions_are_skipped():
    asked = locomo.questions(SAMPLE)
    assert [q["category"] for q in asked] == [4, 2]
    assert asked[0]["question_id"] == "conv-1:0"


def test_judged_correct_reads_json_then_text():
    assert locomo.judged_correct('Same topic. {"label": "CORRECT"}')
    assert not locomo.judged_correct('{"label": "WRONG"}')
    assert locomo.judged_correct("The answer is CORRECT")
    assert not locomo.judged_correct("CORRECT? No, WRONG")
    assert not locomo.judged_correct("")


def test_retrieval_only_run_scores_evidence_and_resumes(tmp_path):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([SAMPLE]))
    out = tmp_path / "run"
    args = ["run", "--system", "bm25", "--data", str(data), "--out", str(out), "--workers", "1"]
    assert locomo.main(args) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["answered"] == 2
    assert summary["retrieval"]["evidence_all"] == 1.0
    assert summary["manifest"]["questions"] == 2
    before = (out / "answers.jsonl").read_text()
    assert locomo.main(args) == 0
    assert (out / "answers.jsonl").read_text() == before
    assert locomo.main([*args, "--token-budget", "100"]) == 2
    assert "bm25" in locomo.render_report([summary])


def test_facts_are_extracted_once_then_indexed_and_shown(tmp_path, monkeypatch):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([SAMPLE]))
    cache = tmp_path / "facts.jsonl"
    prompts = []

    class FakeClient:
        def __init__(self, model):
            pass

        def complete(self, prompt, max_tokens=0):
            prompts.append(prompt)
            reply = '{"0": ["Ben began ceramics lessons on 2023-05-26."]}'
            return type("Completion", (), {"text": reply if "pottery" in prompt else "{}"})()

    monkeypatch.setattr(locomo, "ChatClient", FakeClient)
    args = ["facts", "--data", str(data), "--out", str(cache), "--model", "m", "--workers", "1"]
    assert locomo.main(args) == 0
    assert len(prompts) == 2 and "between Ana and Ben" in prompts[0]
    assert locomo.main(args) == 0
    assert len(prompts) == 2  # resumed: nothing sent again

    out = tmp_path / "run"
    run = ["run", "--system", "myelin", "--data", str(data), "--out", str(out),
           "--facts", str(cache), "--show-facts", "--workers", "1"]
    assert locomo.main(run) == 0
    turns = locomo.conversation_turns(SAMPLE)
    assert turns[2].facts == ("Ben began ceramics lessons on 2023-05-26.",)
    assert turns[0].facts == ()
    from bench.longmemeval import render_history, within_budget

    assert "(noted: Ben began ceramics" in render_history(turns)
    assert len(within_budget(turns, 12)) < len(within_budget(turns, 1000))
    monkeypatch.setattr(locomo._lme, "SHOW_FACTS", False)
    locomo.FACTS.clear()


def test_cite_feedback_benchmark_runs_offline(tmp_path, capsys, monkeypatch):
    from bench import cite_feedback

    monkeypatch.delenv("MYELIN_CITE_LEARNING", raising=False)
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([SAMPLE]))
    assert cite_feedback.main(["--data", str(data), "--seeds", "2"]) == 0
    out = capsys.readouterr().out
    for arm in cite_feedback.ARMS:
        assert f"| {arm} |" in out
    scores = cite_feedback.run_split([SAMPLE], seed=0)
    assert set(scores) == set(cite_feedback.ARMS)
    assert all(0.0 <= v <= 1.0 for by_k in scores.values() for v in by_k.values())
