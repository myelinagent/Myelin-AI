"""LongHAUL's static arm for other memory systems.

    python -m bench.gateway --port 8765 &
    python -m bench.compare --system mem0 --data locomo10.json --out bench/runs/compare \\
        --limit 2 --seeds 3 --embedder minishlab/potion-retrieval-32M
    python -m bench.compare --report --system myelin mem0 graphiti letta-passages \\
        --data locomo10.json --out bench/runs/compare --limit 2

The same replay as ``bench.lifetime``'s ``static`` arm: a session is
stored on its date, and each question is asked some sessions after the
last message that answers it (``bench.lifetime.schedule``, so the same
seeds give the same lags). Nothing is forgotten and nothing is fed back.

The other systems do not return the turns they were given. They return
facts, graph edges or passages made from them. So every returned item
carries the ids of the turns it derives from, and an evidence turn
counts as retrieved when an item deriving from it is in the top k. How
each adapter knows where an item came from is in its docstring.

Writing is the expensive part for these systems (a model call or
several per turn), and a static store looks the same whichever schedule
is being asked. So a conversation is written once and every schedule's
questions are asked along the way. Each write and each asking is
appended to a file under ``--out``; an interrupted run continues from
there, and ``--max-minutes`` stops cleanly between writes.

Model calls and embeddings go through ``bench.gateway``, which counts
tokens and cost and serves the static embedder to every system.
Third-party packages are imported only by the adapter that needs them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bench import gateway
from bench.lifetime import KS, locomo_timeline, schedule
from bench.systems import Turn, _terms

SHOWN = max(KS)
USAGE = gateway.FIELDS


@dataclass(frozen=True)
class Item:
    """One returned item and the turns it derives from."""

    id: str
    sources: tuple[str, ...]
    text: str = ""


class Adapter:
    """A memory system that can be written to turn by turn and searched.

    ``store`` is a directory the adapter may keep its state in; an
    adapter with ``persistent = True`` can be closed and opened again on
    the same directory and carry on. ``base_url`` is an OpenAI-compatible
    endpoint for the model and the embedder.
    """

    name = ""
    label = ""
    # "turn": one write per message. "session": one write per session.
    unit = "turn"
    persistent = True
    # Earlier messages the system shows its model while writing one. A
    # fact drawn from one of those is stored under the message being
    # written; see ``with_context``.
    context_turns = 0

    def __init__(self, store: Path, base_url: str, embedder: str, model: str) -> None:
        self.store = store
        self.base_url = base_url
        self.embedder = embedder
        self.model = model

    def open(self) -> None:
        pass

    def set_time(self, when: datetime) -> None:
        """The simulated clock moved; most systems cannot use it."""

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        """Write turns (id, turn); return the items this created or
        changed, each with every turn it now derives from."""
        raise NotImplementedError

    def search(self, question: str, k: int) -> list[Item]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


def _version(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


class MyelinAdapter(Adapter):
    """Myelin-AI exactly as ``bench.lifetime`` runs its static arm: one
    memory per turn, a session per ``add_many``, default search, in
    memory. A memory is its turn, so the mapping is one to one."""

    name = "myelin"
    label = "Myelin-AI (static arm)"
    unit = "session"
    persistent = False

    def open(self) -> None:
        from myelin import MemoryClient

        embedder: Any = None
        if self.embedder.startswith(gateway.LOCAL_PREFIX):
            from myelin.embedders import StaticEmbedder

            embedder = StaticEmbedder(self.embedder)
        elif self.embedder:
            embedder = _RemoteEmbedder(self.base_url, self.embedder)
        self._now: list[datetime | None] = [None]
        self._client = MemoryClient(
            user_id="bench", embedder=embedder, clock=lambda: self._now[0], cite_learning=True
        )
        self._turn_of: dict[str, str] = {}

    def set_time(self, when: datetime) -> None:
        self._now[0] = when

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        stored = self._client.add_many(
            [
                {
                    "content": turn.text,
                    "t_valid": turn.session_date,
                    "metadata": {"session_id": turn.session_id, "turn": turn.index},
                }
                for _, turn in batch
            ]
        )
        items = []
        for (turn_id, _), memory in zip(batch, stored, strict=True):
            self._turn_of[memory.id] = turn_id
            items.append(Item(memory.id, (turn_id,)))
        return items

    def search(self, question: str, k: int) -> list[Item]:
        return [
            Item(hit.memory.id, (self._turn_of[hit.memory.id],), hit.memory.content)
            for hit in self._client.search(question, top_k=k)
        ]

    def describe(self) -> dict[str, Any]:
        return {"package": "myelin-ai " + _version("myelin-ai"), "model calls to write": "none"}

    def close(self) -> None:
        self._client.close()


class Mem0Adapter(Adapter):
    """Mem0 open source, default write mode (``infer=True``).

    One ``Memory.add`` per turn, so a memory comes from exactly one
    call. The turn's id is passed as ``metadata`` on that call; Mem0
    stores it on every memory the call creates and returns it from
    ``search``, which is where ``sources`` is read from. The first
    speaker is sent as ``user`` and the second as ``assistant``, each
    message prefixed with the speaker's name, as in Mem0's own
    multi-speaker prompt example.

    Weaknesses. Mem0 skips a fact it judges already stored: a later turn
    that repeats an earlier one then has no memory of its own, and the
    earlier memory is not credited to it. And Mem0 shows its model the
    last ten messages as context; a memory the model writes from one of
    those is stored under the turn being added, not the turn that said
    it (``context_turns`` and ``with_context`` give the second reading).
    """

    name = "mem0"
    label = "Mem0 OSS (infer=True, one add per turn)"
    user = "conversation"
    context_turns = 10  # SQLiteManager.get_last_messages(limit=10)

    def open(self) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        os.environ.setdefault("MEM0_DIR", str(self.store / "home"))
        from mem0 import Memory

        endpoint = {"openai_base_url": self.base_url, "api_key": "sk-or-dummy"}
        dims = len(_embed(self.base_url, self.embedder, ["x"])[0])
        self._first_speaker: str | None = None
        speaker = self.store / "first_speaker"
        if speaker.exists():
            self._first_speaker = speaker.read_text()
        self._memory = Memory.from_config(
            {
                "llm": {"provider": "openai", "config": {"model": self.model, **endpoint}},
                "embedder": {
                    "provider": "openai",
                    "config": {"model": self.embedder, "embedding_dims": dims, **endpoint},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(self.store / "qdrant"),
                        "collection_name": "mem0",
                        "embedding_model_dims": dims,
                        "on_disk": True,
                    },
                },
                "history_db_path": str(self.store / "history.db"),
            }
        )

    def _message(self, turn: Turn) -> dict[str, str]:
        if self._first_speaker is None:
            self._first_speaker = turn.role
            (self.store / "first_speaker").write_text(turn.role)
        role = "user" if turn.role == self._first_speaker else "assistant"
        return {"role": role, "content": turn.text}

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        items = []
        for turn_id, turn in batch:
            added = self._memory.add(
                [self._message(turn)], user_id=self.user, metadata={"turn_id": turn_id}
            )
            items += [Item(row["id"], (turn_id,), row["memory"]) for row in added["results"]]
        return items

    def _sources(self, row: dict[str, Any]) -> tuple[str, ...]:
        return (row["metadata"]["turn_id"],)

    def search(self, question: str, k: int) -> list[Item]:
        found = self._memory.search(question, filters={"user_id": self.user}, top_k=k)
        return [Item(row["id"], self._sources(row), row["memory"]) for row in found["results"]]

    def describe(self) -> dict[str, Any]:
        return {
            "package": f"mem0ai {_version('mem0ai')}",
            "write": "Memory.add(infer=True), one call per turn",
            "search": "Memory.search(top_k=20), default threshold",
            "store": f"qdrant-client {_version('qdrant-client')} local, "
                     f"BM25 via fastembed {_version('fastembed')}, "
                     f"entities via spacy {_version('spacy')}",
        }

    def close(self) -> None:
        self._memory.close()


def closest_turns(text: str, turns: dict[str, str]) -> tuple[str, ...]:
    """The turn or turns holding most of an item's words; none if no
    turn shares a word with it."""
    wanted = set(_terms(text))
    held = {turn_id: len(wanted & set(_terms(body))) for turn_id, body in turns.items()}
    best = max(held.values(), default=0)
    return tuple(turn_id for turn_id, count in held.items() if count == best) if best else ()


class Mem0SessionAdapter(Mem0Adapter):
    """Mem0 open source, default write mode, a whole session per ``add``.

    Cheaper and closer to adding a finished conversation, but a memory
    then comes from a call that held every message of the session, and
    Mem0 does not say which. The session is recorded exactly (in
    ``metadata``); the turn is a guess: the turn of that session
    sharing the most words with the memory, or all that tie. Read these
    figures as approximate in both directions.
    """

    name = "mem0-session"
    label = "Mem0 OSS (infer=True, one add per session; turn guessed from wording)"
    context_turns = 0  # the guess already looks across the whole session
    unit = "session"

    def open(self) -> None:
        super().open()
        self._turns_path = self.store / "turns.json"
        self._turns: dict[str, dict[str, str]] = (
            json.loads(self._turns_path.read_text()) if self._turns_path.exists() else {}
        )

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        session = batch[0][0].split(":")[0]
        self._turns[session] = {turn_id: turn.text for turn_id, turn in batch}
        self._turns_path.write_text(json.dumps(self._turns))
        added = self._memory.add(
            [self._message(turn) for _, turn in batch],
            user_id=self.user,
            metadata={"session": session},
        )
        return [
            Item(row["id"], closest_turns(row["memory"], self._turns[session]), row["memory"])
            for row in added["results"]
        ]

    def _sources(self, row: dict[str, Any]) -> tuple[str, ...]:
        return closest_turns(row["memory"], self._turns[row["metadata"]["session"]])

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "write": "Memory.add(infer=True), one call per session"}


class GraphitiAdapter(Adapter):
    """Graphiti (Zep's open-source graph), default write mode.

    One ``add_episode`` per turn, named after the turn. ``search``
    returns edges (facts); every edge lists the episodes it was
    extracted from, and Graphiti adds an episode to an existing edge
    when a later message states the same fact. ``sources`` is that list
    mapped back through the episode names, read from the graph, so an
    edge merged from several turns derives from all of them.

    Weakness: an edge's episode list is what Graphiti's extraction and
    de-duplication recorded. A fact drawn from the ten preceding
    messages Graphiti shows the model as context is credited only to
    the message being written (see ``context_turns``).

    Runs on the embedded Kuzu driver. That driver's own index setup is
    not called by ``Graphiti.build_indices_and_constraints`` in this
    version, so it is called here.
    """

    name = "graphiti"
    label = "Graphiti (add_episode per turn, Kuzu)"
    context_turns = 10  # RELEVANT_SCHEMA_LIMIT previous episodes

    def open(self) -> None:
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        import warnings

        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        self._loop = asyncio.new_event_loop()
        path = self.store / "kuzu.db"
        fresh = not path.exists()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._driver = KuzuDriver(db=str(path))
        llm = LLMConfig(
            api_key="sk-or-dummy", model=self.model, small_model=self.model,
            base_url=self.base_url,
        )
        dims = len(_embed(self.base_url, self.embedder, ["x"])[0])
        self._graph = Graphiti(
            graph_driver=self._driver,
            llm_client=OpenAIGenericClient(config=llm),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key="sk-or-dummy", embedding_model=self.embedder,
                    embedding_dim=dims, base_url=self.base_url,
                )
            ),
            # Required by the constructor; the default search does not call it.
            cross_encoder=OpenAIRerankerClient(config=llm),
        )
        if fresh:
            self._run(self._driver.graph_ops.build_indices_and_constraints(self._driver))
        rows, _, _ = self._run(
            self._driver.execute_query("MATCH (e:Episodic) RETURN e.uuid AS uuid, e.name AS name")
        )
        self._turn_of = {row["uuid"]: row["name"] for row in rows}

    def _run(self, coroutine: Any) -> Any:
        return self._loop.run_until_complete(coroutine)

    def _item(self, edge: Any) -> Item:
        sources = tuple(dict.fromkeys(self._turn_of[e] for e in edge.episodes if e in self._turn_of))
        return Item(edge.uuid, sources, edge.fact)

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        from graphiti_core.nodes import EpisodeType

        items = []
        for turn_id, turn in batch:
            added = self._run(
                self._graph.add_episode(
                    name=turn_id,
                    episode_body=turn.text,
                    source=EpisodeType.message,
                    source_description="conversation",
                    reference_time=turn.session_date,
                )
            )
            self._turn_of[added.episode.uuid] = turn_id
            items += [self._item(edge) for edge in added.edges]
        return items

    def search(self, question: str, k: int) -> list[Item]:
        edges = self._run(self._graph.search(question, num_results=k))
        return [self._item(edge) for edge in edges]

    def describe(self) -> dict[str, Any]:
        return {
            "package": f"graphiti-core {_version('graphiti-core')}",
            "write": "Graphiti.add_episode(source=message), one call per turn",
            "search": "Graphiti.search(num_results=20): edges, hybrid with RRF",
            "store": f"kuzu {_version('kuzu')} embedded",
        }

    def close(self) -> None:
        self._run(self._graph.close())
        self._loop.close()


class LettaPassagesAdapter(Adapter):
    """Letta archival passages, inserted directly. Not a Letta agent.

    Each turn is written with ``agents.passages.create`` and tagged with
    its id; ``agents.passages.search`` returns the tags, which is where
    ``sources`` is read from. No model is called: this is Letta's
    archival vector search over the raw turns, and it leaves out what a
    Letta agent does with a message (edit its memory blocks).

    Needs a Letta server (``LETTA_BASE_URL``, default
    ``http://127.0.0.1:8283``). One agent per conversation holds the
    passages; the store directory only records its id.
    """

    name = "letta-passages"
    label = "Letta archival passages (direct insert, no agent, no model)"

    def open(self) -> None:
        from letta_client import Letta

        self._client = Letta(
            base_url=os.environ.get("LETTA_BASE_URL", "http://127.0.0.1:8283"), api_key="none"
        )
        record = self.store / "agent_id"
        if record.exists():
            self._agent = record.read_text()
            return
        dims = len(_embed(self.base_url, self.embedder, ["x"])[0])
        agent = self._client.agents.create(
            name=f"compare-{self.store.parent.name}-{int(time.time())}",
            llm_config={
                "model": self.model, "model_endpoint_type": "openai",
                "model_endpoint": self.base_url, "context_window": 32000,
            },
            embedding_config={
                "embedding_model": self.embedder, "embedding_endpoint_type": "openai",
                "embedding_endpoint": self.base_url, "embedding_dim": dims,
                "embedding_chunk_size": 300,
            },
        )
        self._agent = agent.id
        record.write_text(agent.id)

    def ingest(self, batch: list[tuple[str, Turn]]) -> list[Item]:
        items = []
        for turn_id, turn in batch:
            passages = self._client.agents.passages.create(
                self._agent, text=turn.text, tags=[turn_id], created_at=turn.session_date
            )
            items += [Item(p.id, (turn_id,)) for p in passages]
        return items

    def search(self, question: str, k: int) -> list[Item]:
        found = self._client.agents.passages.search(self._agent, query=question, top_k=k)
        return [Item(r.id, tuple(r.tags or ()), r.content) for r in found.results]

    def describe(self) -> dict[str, Any]:
        return {
            "package": f"letta {_version('letta')}, letta-client {_version('letta-client')}",
            "write": "agents.passages.create, one call per turn, no model",
            "search": "agents.passages.search(top_k=20): embedding similarity",
        }


class _RemoteEmbedder:
    """An embedding model behind the gateway, in the shape Myelin-AI takes."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url, self._model = base_url, model

    def encode(self, text: str) -> list[float]:
        return _embed(self._base_url, self._model, [text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return _embed(self._base_url, self._model, list(texts))


ADAPTERS: dict[str, type[Adapter]] = {
    a.name: a
    for a in (
        MyelinAdapter, Mem0Adapter, Mem0SessionAdapter, GraphitiAdapter, LettaPassagesAdapter
    )
}


def _embed(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    request = urllib.request.Request(
        f"{base_url}/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        rows = json.loads(response.read())["data"]
    return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]


def _usage(gateway: str, tag: str) -> dict[str, float]:
    """Running totals for a tag, or zeros when there is no gateway."""
    if not gateway:
        return dict.fromkeys(USAGE, 0.0)
    with urllib.request.urlopen(f"{gateway}/t/{tag}/usage", timeout=30) as response:
        return json.loads(response.read())


def _spent(gateway: str, tag: str, before: dict[str, float]) -> dict[str, float]:
    after = _usage(gateway, tag)
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in USAGE}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:  # a line cut short by an interrupted run
            continue
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def units(batch: list[Turn], ids: list[str], unit: str) -> list[list[tuple[str, Turn]]]:
    pairs = list(zip(ids, batch, strict=True))
    return [pairs] if unit == "session" else [[pair] for pair in pairs]


def score(evidence: list[str], ranked: list[Item]) -> dict[str, bool]:
    """All of a question's evidence among the turns the top k items
    derive from, for each k."""
    wanted = set(evidence)
    return {
        f"top{k}": wanted <= {s for item in ranked[:k] for s in item.sources} for k in KS
    }


def with_context(
    text: str, sources: list[str], order: list[str], texts: dict[str, str], window: int
) -> list[str]:
    """``sources`` plus the context turns the item's wording points to.

    A system that shows its model the preceding ``window`` messages can
    write a fact from one of them while storing another. For each
    source this adds the one preceding message, within the window,
    that holds more of the item's words than the source does, if there
    is one. It is a guess from wording, so figures that use it are
    reported beside the recorded mapping, not in place of it.
    """
    wanted = set(_terms(text))
    if not wanted or not window:
        return list(sources)

    def held(turn_id: str) -> int:
        return len(wanted & set(_terms(texts[turn_id])))

    out = list(sources)
    for source in sources:
        at = order.index(source)
        before = order[max(0, at - window):at]
        best = max(before, key=held, default=None)
        if best is not None and held(best) > held(source) and best not in out:
            out.append(best)
    return out


class Incomplete(Exception):
    """The time allowed ran out before the conversation was finished."""


def replay(
    adapter: Adapter,
    sample: dict[str, Any],
    seeds: list[int],
    folder: Path,
    *,
    max_lag: int = 6,
    gateway: str = "",
    tag: str = "",
    deadline: float | None = None,
) -> None:
    """Write one conversation into ``adapter`` and ask every schedule's
    questions as their sessions come, recording both under ``folder``."""
    sessions, ids, asked = locomo_timeline(sample)
    due = {seed: schedule(ids, asked, seed, max_lag=max_lag) for seed in seeds}
    writes, asks = folder / "writes.jsonl", folder / "asked.jsonl"
    # A store that cannot be reopened is rebuilt, and its writes logged again.
    earlier = _rows(writes) if adapter.persistent else []
    written = {(row["step"], row["unit"]) for row in earlier}
    answered = {(row["seed"], row["question_id"]) for row in _rows(asks)}
    # Turns that some stored item derives from, as the writes reported.
    covered = {s for row in earlier for _, sources in row["items"] for s in sources}

    for step, batch in enumerate(sessions):
        when = batch[0].session_date
        adapter.set_time(when)
        for number, unit in enumerate(units(batch, ids[step], adapter.unit)):
            if (step, number) in written:
                continue
            if deadline is not None and time.monotonic() > deadline:
                raise Incomplete
            before, started = _usage(gateway, tag), time.monotonic()
            items = adapter.ingest(unit)
            covered |= {s for item in items for s in item.sources}
            _append(writes, {
                "pass": seeds[0], "step": step, "unit": number, "turns": [turn_id for turn_id, _ in unit],
                "items": [[item.id, list(item.sources)] for item in items],
                "seconds": time.monotonic() - started, "usage": _spent(gateway, tag, before),
            })
        # Questions come a little after the session they follow.
        adapter.set_time(when + timedelta(hours=1))
        shown: dict[str, tuple[list[Item], float, dict[str, float]]] = {}
        for seed in seeds:
            for qa in due[seed].get(step, []):
                if (seed, qa["question_id"]) in answered:
                    continue
                if qa["question"] not in shown:  # the store is the same for every schedule
                    before, started = _usage(gateway, tag), time.monotonic()
                    ranked = adapter.search(qa["question"], SHOWN)
                    shown[qa["question"]] = (
                        ranked, time.monotonic() - started, _spent(gateway, tag, before)
                    )
                ranked, seconds, usage = shown[qa["question"]]
                _append(asks, {
                    "seed": seed, "question_id": qa["question_id"], "step": step,
                    "lag": qa["lag"], "evidence": qa["evidence"],
                    "written": all(e in covered for e in qa["evidence"]),
                    "ranked": [[i.id, list(i.sources), i.text] for i in ranked],
                    "seconds": seconds, "usage": usage,
                    **score(qa["evidence"], ranked),
                })
    _append(folder / "done.jsonl", {"seeds": seeds, "max_lag": max_lag})


def finished(folder: Path, seeds: list[int], max_lag: int) -> bool:
    done = {s for row in _rows(folder / "done.jsonl") if row["max_lag"] == max_lag
            for s in row["seeds"]}
    return set(seeds) <= done


def summarize(
    folders: list[Path],
    seeds: list[int],
    samples: dict[str, dict[str, Any]] | None = None,
    window: int = 0,
) -> dict[str, Any]:
    """Figures for one system over finished conversations: ranking as
    mean and spread over schedules, and what writing cost.

    With ``samples`` (by ``sample_id``) and a context ``window``, also
    the ranking when an item is credited to a context turn it restates
    (``with_context``), under ``context_top5`` and so on.
    """
    writes = [row for folder in folders for row in _rows(folder / "writes.jsonl")]
    # A store rebuilt for every schedule logged its writes once per schedule.
    writes = [row for row in writes if row["pass"] == seeds[0]]
    asks = []
    for folder in folders:
        rows = [row for row in _rows(folder / "asked.jsonl") if row["seed"] in seeds]
        if samples and window:
            sessions, ids, _ = locomo_timeline(samples[folder.name])
            order = [i for names in ids for i in names]
            texts = {i: t.text for batch, names in zip(sessions, ids, strict=True)
                     for t, i in zip(batch, names, strict=True)}
            for row in rows:
                ranked = [
                    Item(i, tuple(with_context(text, sources, order, texts, window)))
                    for i, sources, text in row["ranked"]
                ]
                row.update({"context_" + k: v for k, v in score(row["evidence"], ranked).items()})
        asks += rows

    def share(rows: list[dict[str, Any]], key: str) -> float:
        return 100 * sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0

    per_seed = []
    for seed in seeds:
        rows = [row for row in asks if row["seed"] == seed]
        per_seed.append({
            **{f"top{k}": share(rows, f"top{k}") for k in KS},
            "top10_late": share([r for r in rows if r["lag"] >= 3], "top10"),
            "written": share(rows, "written"),
            **{f"context_top{k}": share(rows, f"context_top{k}")
               for k in KS if rows and f"context_top{k}" in rows[0]},
        })
    out: dict[str, Any] = {"questions": len(asks) // max(1, len(seeds))}
    for key in per_seed[0]:
        values = [run[key] for run in per_seed]
        out[key] = statistics.mean(values)
        out[key + "_spread"] = statistics.stdev(values) if len(values) > 1 else None
    turns = sum(len(row["turns"]) for row in writes)
    out["turns"] = turns
    out["items"] = len({item_id for row in writes for item_id, _ in row["items"]})
    out["write_seconds"] = sum(row["seconds"] for row in writes)
    for key in USAGE:
        out["write_" + key] = sum(row["usage"].get(key, 0.0) for row in writes)
    out["search_calls"] = sum(row["usage"]["calls"] for row in asks)
    searches = [row["seconds"] for row in asks]
    out["search_ms"] = 1000 * statistics.mean(searches) if searches else 0.0
    return out


COLUMNS = (
    ("top5", "All evidence in top 5"),
    ("top10", "top 10"),
    ("top20", "top 20"),
    ("top10_late", "top 10, asked 3+ sessions later"),
    ("written", "Evidence all has a stored item"),
)


def render(summaries: dict[str, dict[str, Any]]) -> str:
    """Two tables: ranking, then what writing cost."""
    lines = ["| | " + " | ".join(title for _, title in COLUMNS) + " | Questions |",
             "|---|" + "---:|" * (len(COLUMNS) + 1)]
    for label, s in summaries.items():
        cells = []
        for key, _ in COLUMNS:
            spread = s[key + "_spread"]
            cells.append(f"{s[key]:.1f}" + (f" ± {spread:.1f}" if spread is not None else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {s['questions']} |")
    restated = {label: s for label, s in summaries.items() if "context_top5" in s}
    if restated:
        lines += ["", "| Also crediting a context turn the item restates | "
                  + " | ".join(f"top {k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
        for label, s in restated.items():
            cells = [
                f"{s[f'context_top{k}']:.1f}" + (
                    f" ± {s[f'context_top{k}_spread']:.1f}"
                    if s[f"context_top{k}_spread"] is not None else ""
                )
                for k in KS
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    titles = ("Turns written", "Items stored", "Model calls to write", "Prompt tokens",
              "Completion tokens", "Cost to write", "Writing, wall clock", "per turn",
              "Search, mean")
    lines += ["", "| | " + " | ".join(titles) + " |", "|---|" + "---:|" * len(titles)]
    for label, s in summaries.items():
        turns = max(1, s["turns"])
        lines.append(
            f"| {label} | {s['turns']} | {s['items']} | {s['write_calls']:.0f} | "
            f"{s['write_prompt_tokens']:,.0f} | {s['write_completion_tokens']:,.0f} | "
            f"${s['write_cost']:.2f} | {s['write_seconds'] / 60:.1f} min | "
            f"{s['write_seconds'] / turns:.2f} s | {s['search_ms']:.0f} ms |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.compare", description=__doc__)
    parser.add_argument("--system", nargs="+", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--data", required=True, help="path to locomo10.json")
    parser.add_argument("--out", required=True, help="where stores and results are kept")
    parser.add_argument("--limit", type=int, default=0, help="only the first N conversations")
    parser.add_argument("--skip", type=int, default=0, help="leave out the first N of those")
    parser.add_argument("--seeds", type=int, default=3, help="random question schedules")
    parser.add_argument("--max-lag", type=int, default=6, help="sessions a question may wait")
    parser.add_argument("--embedder", default="minishlab/potion-retrieval-32M")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="extraction model")
    parser.add_argument(
        "--gateway", default="http://127.0.0.1:8765",
        help="bench.gateway address; '' for none (enough for --system myelin)",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=0.0,
        help="stop between writes after this long; run again to continue",
    )
    parser.add_argument(
        "--report", action="store_true", help="write nothing; print the table from --out"
    )
    args = parser.parse_args(argv)
    samples = json.loads(Path(args.data).read_text())
    if args.limit:
        samples = samples[: args.limit]
    seeds = list(range(args.seeds))
    deadline = time.monotonic() + 60 * args.max_minutes if args.max_minutes else None
    summaries: dict[str, dict[str, Any]] = {}
    incomplete = False
    for name in args.system:
        done = []
        for number, sample in enumerate(samples):
            folder = Path(args.out) / name / str(sample["sample_id"])
            if not finished(folder, seeds, args.max_lag) and number >= args.skip:
                if args.report:
                    continue
                (folder / "store").mkdir(parents=True, exist_ok=True)
                tag = f"{name}.{sample['sample_id']}"
                base_url = f"{args.gateway}/t/{tag}/v1" if args.gateway else ""
                kind = ADAPTERS[name]
                if not kind.persistent:
                    for stale in ("writes.jsonl", "asked.jsonl"):
                        (folder / stale).unlink(missing_ok=True)
                try:
                    # A system whose store costs nothing to build gets a
                    # fresh one per schedule, as in bench.lifetime; the
                    # others are written once and asked every schedule.
                    for group in ([seeds] if kind.persistent else [[seed] for seed in seeds]):
                        adapter = kind(folder / "store", base_url, args.embedder, args.model)
                        adapter.open()
                        (folder / "about.json").write_text(
                            json.dumps(adapter.describe(), indent=1)
                        )
                        try:
                            replay(
                                adapter, sample, group, folder, max_lag=args.max_lag,
                                gateway=args.gateway, tag=tag, deadline=deadline,
                            )
                        finally:
                            adapter.close()
                except Incomplete:
                    incomplete = True
                    print(f"{name} {sample['sample_id']}: out of time, run again to continue",
                          file=sys.stderr)
                    break
            if finished(folder, seeds, args.max_lag):
                done.append(folder)
        print(f"{name}: {len(done)} of {len(samples)} conversations finished", file=sys.stderr)
        if done:
            summaries[ADAPTERS[name].label] = summarize(
                done, seeds, {str(s["sample_id"]): s for s in samples},
                ADAPTERS[name].context_turns,
            )
    if summaries:
        print(render(summaries))
    return 3 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
