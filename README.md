# myelin-ai

`pip install myelin-ai` — a drop-in memory layer for AI agents. Event-anchored
bi-temporal facts, store-grounded dream-synthesis consolidation, and an
apoptosis-based forgetting path that quarantines low-health memories before
they can corrupt future answers. Zero hard dependencies in the core
(`sqlite3` only).

```bash
pip install myelin-ai             # core
pip install "myelin-ai[openai]"   # + OpenAI provider for OrchestratedAgent
pip install "myelin-ai[google]"   # + Gemini provider for OrchestratedAgent
```

Requires Python 3.11+. The import name remains `myelin`.

---

## Set up Myelin-AI as memory for your agents

Myelin-AI ships `myelin-mcp`, a small MCP server that gives an AI
assistant memory that lasts between sessions. Most coding agents can
register it with one command. This section is written so that either a
person or an AI agent can follow it.

- [What gets set up](#what-gets-set-up)
- [If you are an AI agent: run the guided setup](#if-you-are-an-ai-agent-run-the-guided-setup)
- [Install](#install)
- [Claude Code](#claude-code) · [Codex](#openai-codex) · [Gemini CLI](#gemini-cli) · [Grok](#grok) · [OpenClaw](#openclaw) · [Hermes](#hermes-agent) · [Cloud agents over HTTP](#cloud-agents-the-http-server) · [OpenAI Dots](#openai-dots)
- [Home-built agents](#home-built-agents)
- [The memory instructions](#the-memory-instructions) · [Check it works](#check-it-works) · [Remove it](#remove-it)

**What has been tested.** The server (see `tests/test_mcp_server.py`
and `tests/test_mcp_http.py`), the Claude Code commands, the `uvx`
form, the Python example, the sidecar calls and the HTTP server (with
the official MCP client and with Claude Code as a remote client) were
run here. No HTTPS proxy or public deployment was tested. The
commands for Codex, Gemini CLI, Grok, OpenClaw and Hermes are taken
from each vendor's documentation as of 2026-11-16 and have not been
run here; each section links its source. If a command has
changed, the tool's own `mcp --help` is the authority.

### What gets set up

| Piece | What it is | Where it lives |
|---|---|---|
| The package | `myelin-ai` from PyPI, which provides the `myelin-mcp` command | your Python environment |
| The memory file | one SQLite file holding every memory as plain text | `~/.myelin/memory.db` by default, or a path you choose |
| The registration | an entry telling your agent how to start `myelin-mcp` | the agent's own config file |
| The instructions | a few lines telling the agent when to recall and remember | the agent's instruction file (`CLAUDE.md`, `AGENTS.md`, ...) |

The assistant gets five tools: `remember`, `recall`, `mark_used` (which
recalled memories its answer relied on, so useful ones rank higher next
time), `forget` and `memory_stats`.

What to know before you start:

- **It is local** unless you choose the HTTP server. The server makes no network calls. With hybrid search
  it downloads a small embedding model once, then runs offline.
- **It is not encrypted.** Anyone who can read the file can read the
  memories. Do not let it store passwords, keys or other secrets.
- **Nothing is deleted or pruned automatically.**
- **One file can be shared.** Several tools pointed at the same
  `--db-path` share one memory. Separate paths keep them apart.

### If you are an AI agent: run the guided setup

A person has asked you to set up Myelin-AI for them. Do it as a short
conversation, not a silent install. The person should finish knowing
what was put on their machine, where, and how to undo it.

**Start by offering the choice.** Say, in your own words:

> Myelin-AI gives me memory that lasts between our sessions. It is a
> small local program and one file on your machine; nothing is sent
> anywhere. I can walk you through five short choices, or you can say
> **"skip to the end"** and I will use the recommended settings, set
> everything up, and tell you what I did.

**Guided path.** Ask one question at a time, give the recommendation,
and wait for the answer.

1. **Which tool am I?** Work out which agent you are running in (Claude
   Code, Codex, Gemini CLI, Grok, OpenClaw, Hermes, something else) and
   tell the person. Use the matching section below. If you cannot tell,
   ask.
2. **One memory for everything, or one per project?** Explain the
   difference: *global* means preferences and decisions follow the
   person into every project; *per project* means each project keeps its
   own memory and nothing crosses over. Recommend global for a single
   person's own machine, per project for client work or anything
   confidential.
3. **Search by meaning, or by words only?** *Hybrid* finds memories that
   mean the same thing in different words; it installs two extra
   libraries and downloads a model of about 125 MB once.
   *Lexical* needs nothing extra and matches on shared words. Recommend
   hybrid.
4. **Where should the memory file live?** Offer the default for the
   scope chosen (`~/.myelin/memory.db`, or `.myelin/memory.db` inside
   the project). Say that it is a plain, unencrypted file. For a
   per-project file, offer to add `.myelin/` to `.gitignore` so
   memories are not committed.
5. **May I add standing instructions?** Show the exact text from
   [The memory instructions](#the-memory-instructions) and name the
   file you would add it to. Without it you will use memory only when
   asked. Recommend yes.

Then install, register, and verify (next sections), and finish with the
summary below.

**"Skip to the end" path.** Use these defaults without asking: global
scope, hybrid search, `~/.myelin/memory.db`, instructions added. Then
give the same summary.

**Rules for both paths.**

- Append to existing instruction and config files; never replace them.
  If a `myelin` entry or a "Memory" block is already there, leave it
  and say so.
- Do not use `sudo` or change system Python. If the install fails, stop
  and report the error.
- Verify before you claim success: run the check in
  [Check it works](#check-it-works) and report what you saw.
- Never store a secret, and say so if the person asks you to.

**Finish with a summary** that lists: what was installed and its
version; the path of the memory file; which config file you changed and
the entry you added; which instruction file you changed; whether the
check passed; that a new session is needed before the tools appear; and
the commands to remove it all.

### Install

```bash
pip install -U "myelin-ai[static]"    # hybrid search (recommended)
pip install -U myelin-ai              # lexical search only
```

Requires Python 3.11 or newer. `myelin-mcp` must be on the `PATH` that
your agent uses; `which myelin-mcp` (or `where myelin-mcp` on Windows)
should print a path. If it does not, either install with
`pipx install "myelin-ai[static]"`, or skip installing and let `uvx`
run it: wherever a command below says `myelin-mcp`, write
`uvx --from "myelin-ai[static]" myelin-mcp` instead.

Options: `--db-path FILE` (default `~/.myelin/memory.db`), `--user-id
NAME` (separate ids in one file do not see each other), `--no-embedder`
(lexical search even if the extra is installed).

### Claude Code

**Global** (every project shares one memory):

```bash
claude mcp add --scope user myelin -- myelin-mcp
```

**One project** (run inside the project; memory stays in the project):

```bash
claude mcp add myelin -- myelin-mcp --db-path .myelin/memory.db
```

The default scope is `local`: the entry applies to this project on your
machine only. To share the registration with teammates through a
committed `.mcp.json`, use `--scope project`; each person still gets
their own memory file, so add `.myelin/` to `.gitignore`.

Instructions go in `~/.claude/CLAUDE.md` (global) or `CLAUDE.md` in the
project. Check with `claude mcp list` or `/mcp` in a session.

### OpenAI Codex

```bash
codex mcp add myelin -- myelin-mcp
```

That writes the user config, `~/.codex/config.toml`, which the Codex
CLI and IDE extension share. For one project, add this to
`.codex/config.toml` in the project (read for trusted projects only):

```toml
[mcp_servers.myelin]
command = "myelin-mcp"
args = ["--db-path", ".myelin/memory.db"]
```

Instructions go in `~/.codex/AGENTS.md` (global) or `AGENTS.md` in the
project. Check with `codex mcp list` or `/mcp` in the TUI.
Source: [Codex MCP docs](https://developers.openai.com/codex/mcp).

### Gemini CLI

`gemini mcp add` defaults to **project** scope, so pass `-s user` for a
global memory:

```bash
gemini mcp add -s user myelin myelin-mcp
```

```bash
gemini mcp add myelin myelin-mcp --db-path .myelin/memory.db
```

Or edit `~/.gemini/settings.json` (global) or `.gemini/settings.json`
(project):

```json
{ "mcpServers": { "myelin": { "command": "myelin-mcp", "args": [] } } }
```

Instructions go in `~/.gemini/GEMINI.md` (global) or `GEMINI.md` in the
project. Check with `gemini mcp list` or `/mcp`. A local server in a
folder Gemini does not trust shows as disconnected; run `gemini trust`.
Source: [Gemini CLI MCP docs](https://www.geminicli.com/docs/tools/mcp-server).
That page notes that for some account tiers Gemini CLI has been replaced
by Antigravity CLI; if you use that, check its own MCP documentation.

### Grok

**Grok Build** (xAI's coding CLI):

```bash
grok mcp add myelin -- myelin-mcp
```

```bash
grok mcp add --scope project myelin -- myelin-mcp --db-path .myelin/memory.db
```

These write `~/.grok/config.toml` and `.grok/config.toml`:

```toml
[mcp_servers.myelin]
command = "myelin-mcp"
args = []
```

Grok Build also reads MCP servers from `~/.claude.json` and a project
`.mcp.json`, so a Claude Code registration may already be picked up.
Check with `grok mcp list`, `grok mcp doctor myelin` or `/mcps`. The
docs read for this guide do not name Grok Build's instruction file; put
the memory instructions wherever your Grok setup keeps project rules.
Source: [Grok Build MCP docs](https://docs.x.ai/build/features/mcp-servers).

**Grok Bot** (xAI's cloud teammates) runs on a cloud computer, not your
machine, so it needs the [HTTP server](#cloud-agents-the-http-server).
Untested; see [OpenAI Dots](#openai-dots).

### OpenClaw

```bash
openclaw mcp add myelin --command myelin-mcp
```

For a specific file, each argument gets its own `--arg`. The OpenClaw
docs do not show an argument that starts with a dash, so if this form
is rejected use `openclaw mcp set` with the JSON below:

```bash
openclaw mcp add myelin --command myelin-mcp --arg=--db-path --arg="$HOME/.myelin/memory.db"
```

```bash
openclaw mcp set myelin '{"command":"myelin-mcp","args":["--db-path","/home/you/.myelin/memory.db"]}'
```

Check with `openclaw mcp doctor myelin --probe`. A running gateway does
not pick up the change by itself: restart it, or use its reload path.
Instructions go in `AGENTS.md` in the agent workspace
(`~/.openclaw/workspace` by default), which OpenClaw loads at the start
of every session. OpenClaw agents often run unattended, so the
"never store secrets" line matters more here.
Source: [OpenClaw MCP docs](https://docs.openclaw.ai/cli/mcp).

OpenClaw's memory-plugin convention is also supported directly: the HTTP
sidecar (`myelin-serve`, see [Home-built agents](#home-built-agents))
exposes `/hooks/before_agent_start` and `/hooks/agent_end`.

### Hermes Agent

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  myelin:
    command: "myelin-mcp"
    args: []
```

Start a new session or run `/reload-mcp`. Hermes prefixes MCP tools, so
the tools appear as `mcp_myelin_recall`, `mcp_myelin_remember` and so
on; use those names in the instructions. Hermes starts local servers
with a filtered environment, so if `myelin-mcp` is not found, give its
full path as `command`. Instructions go in `~/.hermes/SOUL.md` (always
loaded) or in the project's `AGENTS.md`.
Source: [Hermes MCP docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp).

### Cloud agents: the HTTP server

Agents that run on someone else's computer (OpenAI Dots, Grok Bot, a
hosted assistant with "custom connectors") cannot start a program on
your machine. For them, `myelin-mcp --http` serves the same five tools
over the network, using MCP's Streamable HTTP transport.

**You host it.** Myelin-AI does not run a service for you. You need a
machine that stays on, keeps its disk, and can be reached over HTTPS: a
small VM, a home server behind a tunnel, or a container host with a
persistent volume. Hosts that give each request a fresh filesystem will
lose the memory file.

```bash
pip install -U "myelin-ai[static]"
export MYELIN_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
myelin-mcp --http --host 0.0.0.0 --port 8766 --db-path /var/lib/myelin/memory.db
```

Then put HTTPS in front of it. The server does not encrypt traffic, so
without this step the token and every memory cross the network in the
clear. Any reverse proxy or tunnel that forwards to port 8766 will do.

The agent connects to `https://your-host/mcp` and sends the token as
`Authorization: Bearer <token>`. For a connector form that only takes a
URL, use `https://your-host/mcp/<token>`; the token is then part of the
URL, so treat the whole URL as the secret and expect it to appear in the
connecting product's settings.

| | |
|---|---|
| One person | set `MYELIN_MCP_TOKEN`; memories belong to `--user-id` |
| Several people | `--tokens-file tokens.json` containing `{"<token>": "<user id>"}`; each token sees only its own user's memories |
| Browser clients | refused unless you pass `--allow-origin https://...` |
| No token | the server starts only on `127.0.0.1`, for local testing; it will not start open to the network |
| Health check | `GET /health` (no token, reveals nothing) |

**One memory for all your agents.** Point every agent at the same
server as the same user and they share one memory: what Codex stores,
Claude Code can recall. Each memory records which agent stored it, and
`recall` shows it as `(from codex)`. Tell the server who is calling in
one of three ways:

- a token per agent, all naming you, in the tokens file:
  `{"<token A>": {"user": "scott", "source": "claude-code"},
  "<token B>": {"user": "scott", "source": "codex"}}`. This also lets
  you cut off one agent by removing its token;
- `?source=codex` on the URL, e.g. `https://your-host/mcp?source=codex`;
- an `X-Myelin-Source: codex` header.

The label is a record of where a memory came from, not a filter: every
agent can still recall, mark and forget all of your memories. A source
given on the URL or in a header is whatever the caller says it is; only
a per-agent token makes it trustworthy. Local (stdio) registrations are
tagged with the name the client announces, or with `--source NAME`.

What this does not do: rate limiting, token rotation without a restart,
OAuth, or encryption of the file at rest. Anyone holding a token can
read, add and delete that user's memories. Use a long random token, keep
it out of chat, and rotate it by restarting with a new one.

Local tools can use the same server, which is one way to share a memory
between machines:

```bash
claude mcp add --transport http myelin https://your-host/mcp --header "Authorization: Bearer $MYELIN_MCP_TOKEN"
```

### OpenAI Dots

**Untested.** A dot runs on OpenAI's cloud computer, so it needs the
[HTTP server](#cloud-agents-the-http-server) above, reachable over
HTTPS. This project has not been able to confirm how a dot adds a custom
MCP server, or whether it can send an `Authorization` header; if its
connector form takes only a URL, use the `/mcp/<token>` form. Published
descriptions of Dots also say they keep their own memory, so decide
which one you want it to rely on. The same applies to xAI's Grok Bot.

### Home-built agents

Three ways in, from least to most code.

**1. Your framework speaks MCP.** Start `myelin-mcp` as a stdio server
and add the [memory instructions](#the-memory-instructions) to your
system prompt. The server's `initialize` reply also carries those
instructions in its `instructions` field.

**2. Python, in process.** The loop is: recall, answer, report what was
used, remember what is worth keeping.

```python
from myelin import MemoryClient, build_answer_prompt, format_context, parse_used
from myelin.embedders import StaticEmbedder      # needs the "static" extra

memory = MemoryClient(db_path="memory.db", user_id="scott",
                      embedder=StaticEmbedder(), cite_learning=True)

def answer(question: str) -> str:
    hits = memory.recall(question, token_budget=2000)
    prompt = build_answer_prompt(format_context(hits, numbered=True),
                                 question, ask_used=True)
    reply = your_model(prompt)                   # any provider
    memory.cite_used(hits, parse_used(reply), query=question)
    return reply

memory.add("Scott deploys on Tuesday mornings.")  # when something is worth keeping
```

Drop `embedder=` for lexical search. Deciding *what* is worth keeping is
your agent's job: give your model a `remember` tool, or call
`memory.add_conversation(turns, when=...)` to store a whole exchange.

**3. Any language, over HTTP.** Run the sidecar on the same machine:

```bash
MYELIN_CITE_LEARNING=1 myelin-serve --db-path ~/.myelin/memory.db --port 8765
```

| Call | Body | Returns |
|---|---|---|
| `POST /tools/memory_store` | `{"content": "..."}` | the stored memory |
| `POST /tools/memory_recall` | `{"query": "...", "token_budget": 2000, "numbered": true}` | `hits` and a ready `context` string |
| `POST /tools/memory_cite_used` | `{"shown": [ids], "used": [1, 3], "query": "..."}` | counts |
| `POST /tools/memory_forget` | `{"memory_id": "..."}` | whether it was deleted |

The sidecar has no authentication and binds to loopback; do not expose
it beyond the local machine without putting it behind something that
does.

### The memory instructions

Registering the server tells the agent the tools exist. These lines make
it use them without being asked. Put them in the agent's instruction
file:

```markdown
## Memory
- At the start of each task, call the myelin `recall` tool with what the
  task is about, and again whenever earlier decisions might matter.
- When I state a preference, a decision or a fact worth keeping, call
  `remember` with one self-contained sentence.
- After answering from recalled memories, call `mark_used` with the
  numbers you relied on.
- Never store passwords, keys or other secrets.
```

| Tool | Global file | Project file |
|---|---|---|
| Claude Code | `~/.claude/CLAUDE.md` | `CLAUDE.md` |
| Codex | `~/.codex/AGENTS.md` | `AGENTS.md` |
| Gemini CLI | `~/.gemini/GEMINI.md` | `GEMINI.md` |
| OpenClaw | `AGENTS.md` in the workspace | |
| Hermes | `~/.hermes/SOUL.md` | `AGENTS.md` |
| Home-built | your system prompt | |

### Check it works

1. The command starts and answers. This prints one line of JSON naming
   `myelin-ai` and its version:

   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | myelin-mcp --db-path /tmp/myelin-check.db --no-embedder
   ```

2. The agent sees it: the tool's list command (`claude mcp list`,
   `codex mcp list`, `gemini mcp list`, `grok mcp list`,
   `openclaw mcp doctor myelin --probe`) shows `myelin` as connected.
3. It remembers across sessions: in a new session say "remember that my
   test phrase is blue heron", start another new session, and ask "what
   is my test phrase?".

### Remove it

```bash
claude mcp remove myelin        # or: codex mcp remove myelin, gemini mcp remove myelin
pip uninstall myelin-ai
```

Then delete the "Memory" block from the instruction file, and delete
the memory file (`~/.myelin/memory.db` or the project's `.myelin/`) if
you want the memories gone. Removing the registration alone leaves the
memories on disk.

## What's new in 1.5.0

| Area | 1.4 | 1.5 |
|---|---|---|
| Finding memories | substring-style scan, top k | BM25 with stemming, optional hybrid search (`embedder=`), `recall(token_budget=)` with query-focused excerpts |
| Answering | bring your own prompt | `format_context`, `build_answer_prompt` |
| Loading history | one `add()` at a time | `add_many`, `add_conversation` (optionally extracting dated facts) |
| Forgetting | `prune()` deleted | quarantine first; deletion only on request; outdated values superseded, not removed |
| Feedback | `cite(id, answer_correct=)` per memory | `cite_used(results, used)`: the answering model names what it used |
| Health | age and own use | one grade over age, use and related memories |
| Benchmarks | none in the repo | `bench/`: LongMemEval, LoCoMo, cite feedback, memory over time (LongHAUL), scale |

Measured with the harness in this repository: LongMemEval-S 90.1% with
`gpt-5-mini` (four runs) and 88.6% with `gpt-4o`; LoCoMo 92.6%. Release
1.4.0 scored 60.4% on LongMemEval-S with `gpt-4o-mini`. Details and
caveats: [`bench/results/final.md`](bench/results/final.md).

LoCoMo (Maharana et al., ACL 2024) is licensed CC BY-NC 4.0 and
LongMemEval (Wu et al., ICLR 2025) is MIT. Neither dataset is
redistributed here; scores are reported for research comparison. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Requires Python 3.11+. `prune()` now quarantines instead of deleting;
see the [changelog](CHANGELOG.md) before upgrading.

## What's new in 1.4.0

Three load-bearing mechanisms that were silently no-oping in 1.3 are now
actually wired. One new feature: store-grounded dream synthesis.

| Area | 1.3 behavior | 1.4 behavior |
|---|---|---|
| D-MEM surprise score (`zscore_surprise`) | `z == 0` for every input — broken normalization | Z-scored against a query-independent reference; signed and discriminating |
| Dream grader (`grade_hit`) | Substring-match against the prompt — answerer could echo and score a hit | Answer key held out of the prompt, graded against the elided key |
| Heuristic salience novelty axis | Pinned at 0.5 cold-start neutral (no embedder wired) | Lazy sentence-transformers embedder + per-scope rolling neighbour window |
| Dream cycle grounding | Ungrounded cloze: answerer guesses from query alone | Optional `retriever=` grounds queries against the memory store |
| Salience reward axis | Not pluggable | New `outcome=` kwarg on `add()` |
| Sidecar liveness | `/health` only | Adds `/metrics` for vacuous-PASS guards |

**This is the first PyPI release where the substrate behaves as designed.** If
you were on 1.3, upgrade — see the
[migration notes](https://github.com/myelinagent/Myelin-AI/blob/main/CHANGELOG.md).

---

## Why `myelin-ai` instead of Mem0 / Letta / Zep

- **Forgets on purpose.** Mem0, Letta, MemGPT, and similar agent-memory layers
  all monotonically accumulate. Adversarial inputs (poison, contradiction
  stuffing, role-play overrides) silently corrupt their stores.
  `client.prune()` is a principled forgetting path with an auditable
  quarantine → tombstone state machine.
- **Counterfactual self-test.** `client.consolidate(retriever=...)` lets the
  system probe its own answerer with held-out paraphrased queries during idle
  windows, grounded against the actual memory store.
- **Drop-in, not opinionated.** Zero hard deps in the core. No bundled
  answerer, no torch dep at install. You bring your own LLM. The
  `OrchestratedAgent` and HTTP sidecar are opt-in extras.

---

## Quickstart — the 1.4.0 happy path

The canonical wiring: add facts with outcome signal, run a store-grounded
dream cycle, and let apoptosis prune low-health residue.

```python
from myelin import MemoryClient

client = MemoryClient(
    user_id="scott",
    salience_mode="heuristic",       # 1.4: novelty + reward axes actually wired
)

# 1.4: outcome= threads success/failure into the heuristic gate's reward axis.
client.add(
    "Scott prefers oolong tea.",
    outcome={"task_succeeded": True, "user_thumbs_up": True},
)
client.add(
    "Scott uses Cursor for code review.",
    outcome={"task_succeeded": True},
)
client.add("Alice loves chess.")

# Hot retrieval.
for h in client.search("What does Scott drink?", top_k=3):
    print(f"{h.score:.2f}  {h.memory.content}")
```

### Store-grounded dream synthesis (the 1.4.0 marquee)

```python
# 1.4: retriever= turns the dream cycle from an ungrounded cloze into a
# real recall test. Each held-out query is prepended with retrieved memory
# context so the answerer must recall from the store, not guess.
class EchoLLM:
    def complete(self, prompt: str) -> str:
        # Replace with your real LLM (OpenAI, Anthropic, vLLM, llama.cpp...)
        return prompt

client = MemoryClient(user_id="scott", llm_client=EchoLLM())

report = client.consolidate(
    retriever=lambda q: [r.memory.content for r in client.search(q, top_k=5)],
)
print(f"hit_rate={report.hit_rate:.2%}  "
      f"facts_sampled={report.facts_sampled}  "
      f"llm_calls={report.llm_calls}")
```

Without `retriever=`, the cycle still runs but uses the v1.3-style ungrounded
cloze (kept for backward compat). The grader, however, is fixed in 1.4 either
way — `grade_hit` now scores against a held-out answer key, not a substring
of the prompt.

---

## Answering from memory (1.5)

Three steps turn stored memories into an answer: recall what fits a token
budget, lay it out in date order, and wrap it in a prompt that tells the
model how to use it.

```python
from datetime import UTC, datetime
from myelin import MemoryClient, build_answer_prompt, format_context

client = MemoryClient(user_id="scott")

# Load history at the time it happened (one transaction, one embedding call).
client.add_many([
    {"content": "Scott moved to Berlin.",
     "t_valid": datetime(2025, 3, 2, tzinfo=UTC),
     "metadata": {"session_id": "s1", "turn": 0}},
    {"content": "Scott is back in Paris for good.",
     "t_valid": datetime(2025, 9, 14, tzinfo=UTC),
     "metadata": {"session_id": "s2", "turn": 0}},
])

question = "Where does Scott live now?"
hits = client.recall(question, token_budget=6000)   # as many top hits as fit
prompt = build_answer_prompt(format_context(hits), question)
answer = your_llm(prompt)                           # bring your own model
```

Without an embedder, search is lexical: a question finds memories that
share words with it (here, "Scott"). Add an embedder, below, to match on
meaning as well.

- **`recall`** ranks with BM25 over a SQLite full-text index and returns
  the best results that fit `token_budget`. Long memories count as an
  excerpt of the passages that bear on the question (`result.excerpt`),
  except the three best-ranked, which stay whole. Stored memories are never
  altered.
- **`format_context`** renders results oldest first under date headings.
- **`build_answer_prompt`** adds rules for the cases models get wrong:
  facts that changed (the latest wins), totals and orderings, dates stated
  relative to a conversation, requests for advice, and questions the
  history does not cover. `style="reasoned"` (default) asks for brief notes
  and a final `Answer:` line; `style="direct"` asks for the answer only.

### Hybrid search

Add an embedding channel beside BM25 by passing an embedder. The core
package stays dependency-free; the bundled embedder is a CPU-only static
model behind an extra.

```bash
pip install "myelin-ai[static]"
```

```python
from myelin import MemoryClient
from myelin.embedders import StaticEmbedder

client = MemoryClient(db_path="memory.db", embedder=StaticEmbedder())
client.backfill_embeddings()    # embed memories written before this
```

Any object with `encode(text) -> sequence of floats` (and optionally
`encode_batch(texts)`) works in place of `StaticEmbedder`. If the embedder
raises, writes and searches carry on lexically.

### Facts at write time (opt-in)

`add_conversation` stores a conversation one memory per message. Given a
model, it also makes one call per conversation to extract short dated
facts and indexes them with the messages they came from. Messages are
still stored and returned whole.

```python
def llm(prompt: str) -> str:        # any model, any provider
    ...

client.add_conversation(
    [("Ana", "Pickle came home yesterday!"), ("Ben", "So happy for you.")],
    when=datetime(2023, 5, 20, tzinfo=UTC),
    llm=llm, people=["Ana", "Ben"],     # omit people for a user/assistant chat
)
client.search("when did Ana get her dog")   # finds the message via "Ana adopted a dog on 2023-05-19."
```

On LoCoMo this raises accuracy by about 3 points; on LongMemEval-S it
made no measurable difference (tables below).

### Measured on LongMemEval-S

All 500 questions, judged by `gpt-4o` with the benchmark's own prompts. One
run each unless several scores are listed; identical runs with `gpt-5-mini`
have differed by up to 3.4 points. Reproduce with
`python -m bench.longmemeval`; see [`bench/README.md`](bench/README.md).

| Setup | `gpt-4o-mini` answering | `gpt-4o` answering | `gpt-5-mini` answering |
|---|---:|---:|---:|
| 1.4.0 search, top 10, plain prompt | 60.4 | not run | not run |
| BM25 + budget + excerpts, plain prompt | 64.8 | not run | not run |
| + `build_answer_prompt` | 77.2 | 82.8 | not run |
| + hybrid search (`StaticEmbedder`) | 78.0 | 87.4 | 89.0 (89.2, 88.8) |
| + 100-token excerpts (the `recall()` default since) | not run | not run | 90.3 (90.2, 90.4) |
| + excerpts that fill their budget around the match | not run | not run | 91.3 (93.0, 89.6, 91.4, 91.0) |
| + two-part fusion: the code on `main` | not run | 88.6 | 90.1 (90.2, 91.0, 89.4, 89.8) |
| Perfect retrieval (control) | 79.4 | 93.6 | 93.4 |

These are not comparable with scores published by other systems beyond
orientation: harnesses, prompts and answer models differ. Full tables,
controls and caveats are in [`bench/results/`](bench/results/).

### Measured on LoCoMo

1,540 questions over ten long conversations, hybrid search, the same
6,000-token budget, judged by `gpt-4o-mini` with the prompt published
LoCoMo scores use. One run each. Reproduce with `python -m bench.locomo`.

| | `gpt-4o-mini` answering | `gpt-5-mini` answering |
|---|---:|---:|
| Judged correct (overall) | 78.7 | 89.0 |
| single-hop / multi-hop | 88.6 / 77.7 | 93.2 / 84.8 |
| temporal / open-domain | 59.8 / 58.3 | 86.6 / 71.9 |
| **+ facts extracted at write time** (`myelin.enrich`), overall | 82.4 | **91.9** |
| the same, on `main` after the later retrieval changes | not run | **92.6** |
| + those facts also shown to the answerer, overall | 83.6 | 91.3 |

Fact extraction is opt-in and costs one model call per conversation. It
pays here and made no measurable difference on LongMemEval-S.

The same caution about comparing with published scores applies; see
[`bench/results/locomo.md`](bench/results/locomo.md).

---

## Memory-aware agent (`OrchestratedAgent`, v1.3)

A provider-agnostic multi-step agent. A cheap classifier (`gpt-4o-mini` by
default) routes each question as `factual` (multi-step with `memory_search`
tool calls) or `synthesis` (top-10 prefetch + tool available).

```python
from myelin import MemoryClient
from myelin.agent import OrchestratedAgent

client = MemoryClient(user_id="u1")
client.add("The user prefers green tea over coffee.")
client.add("The user is allergic to peanuts.")

# Works against OpenAI (openai/*) or Gemini (google/*) — pick what you have.
agent = OrchestratedAgent(client=client, model="openai/gpt-4o")
answer = agent.answer(
    question="What does the user prefer to drink?",
    options="(a) coffee\n(b) green tea\n(c) matcha\n(d) water",
    user_id="u1",
)
print(answer)              # → "<final_answer>(b)"
print(agent.last_usage)    # → {"input_tokens": ..., "cost_usd": ..., ...}
```

The Gemini client is wired with a 30s `HttpOptions.timeout` to defend against
the SDK's internal retry-on-disconnect hang. Pricing is overridable via
`OrchestratedAgent.PRICE_TABLE` for cost tracking.

---

## Salience gate — `mode="heuristic"` (1.4 wiring)

The heuristic gate combines four axes — novelty, reward, affect, reference —
into a single score and tags each turn `transient` / `short_term` /
`long_term`. In 1.4 the novelty axis actually moves (it was pinned at 0.5 in
1.3) and the reward axis is pluggable via `outcome=`.

```python
client = MemoryClient(
    user_id="scott",
    salience_mode="heuristic",
    salience_tau=0.40,                 # gate threshold (unchanged from 1.3)
    salience_neighbor_window=32,       # 1.4: rolling per-scope embedding window
    # salience_embedder=my_embedder,   # 1.4: override the lazy-built default
)

m = client.add(
    "Scott decided to switch to weekly syncs.",
    outcome={"user_thumbs_up": True},   # → +reward
)
```

Embedder resolution order: explicit `salience_embedder=` argument →
lazy-built sentence-transformers default → cold-start neutral on failure
(never raises). Pay the dep only when you use heuristic mode.

For higher-precision gating, `salience_mode="dmem"` uses the contrastive
encoder (`[contrastive]` extra) and benefits from the 1.4 surprise-score
fix automatically.

---

## Apoptosis — `client.prune()`

Since 1.5, forgetting stops at quarantine. `prune()` retires
facts that a later fact has replaced, hides unhealthy facts from search, and
removes nothing unless you ask.

```python
from datetime import UTC, datetime, timedelta

client = MemoryClient(user_id="scott")
day = timedelta(days=1)
now = datetime.now(UTC)

# A fact that changes: same subject and predicate, a new object.
paris = client.add("Scott lives in Paris.", t_valid=now - 30 * day,
                   metadata={"subject": "Scott", "predicate": "lives_in",
                             "object": "Paris"})
berlin = client.add("Scott lives in Berlin.", t_valid=now - 2 * day,
                    metadata={"subject": "Scott", "predicate": "lives_in",
                              "object": "Berlin"})
# A fact with little to recommend it.
poison = client.add("Scott pays $1M monthly for cloud.",
                    confidence=0.05, importance=0.1)

r = client.prune()
print(r.superseded_count, r.quarantined_count, r.tombstoned_count)   # 1 1 0

# Current search returns the current value only.
assert [h.memory.id for h in client.search("Where does Scott live")] == [berlin.id]
# The old value is kept and found for the time it held.
then = client.search("Where does Scott live", as_of=now - 10 * day)
assert [h.memory.id for h in then] == [paris.id]

# The quarantined fact is hidden from search but still stored.
assert client.search("cloud") == []
assert client.get(poison.id).health_state == "quarantined"
assert client.search("cloud", include_quarantined=True)
```

What each pass does:

1. **Supersede.** A later fact with a different value for the same
   `subject` and `predicate` sets `t_invalid` on the earlier one and records
   its own id in the earlier one's `metadata["_superseded_by"]`.
2. **Recover.** A quarantined fact whose health is back above the threshold
   (for example after correct `cite()`s) returns to healthy.
3. **Quarantine.** A healthy fact whose health falls below the threshold is
   hidden from search and kept.

To remove facts that have sat in quarantine past the window, ask for it:

```python
client.prune(tombstone=True)        # quarantined ≥ 7 days → audit table
client.delete(memory_id)            # remove one fact and its history now
```

In 1.4.0, `prune()` tombstoned quarantined facts automatically after seven
days, and quarantined facts still appeared in search.

---

## Bi-temporal event anchoring — `as_of=`

```python
from datetime import UTC, datetime
from myelin import MemoryClient, Event

client = MemoryClient()
client.add(
    "Scott decided to switch to weekly syncs.",
    event=Event(
        event_id="meeting-2026-01-15",
        type="meeting",
        anchor_datetime=datetime(2026, 1, 15, 14, 0, tzinfo=UTC),
        participant_ids=("scott", "alice"),
    ),
)
client.add(
    "Scott decided to switch back to daily syncs.",
    event=Event(
        event_id="meeting-2026-03-15",
        type="meeting",
        anchor_datetime=datetime(2026, 3, 15, 14, 0, tzinfo=UTC),
        participant_ids=("scott", "alice"),
    ),
)

# "What was Scott's stance in February?" — only the January meeting is valid.
feb_state = client.search(
    "Scott syncs", as_of=datetime(2026, 2, 1, tzinfo=UTC),
)
```

---

## Session-scoped working memory (1.2)

Opt in with `MYELIN_WORKING_MEMORY=1` (strict-`"1"` parsing — `"true"` /
`"yes"` won't engage).

```python
import os
os.environ["MYELIN_WORKING_MEMORY"] = "1"

client = MemoryClient(user_id="scott")
SID = "convo-2026-06-09"

t1 = client.observe("What did we agree on yesterday?", session_id=SID)
t2 = client.observe("You committed to a weekly review.", session_id=SID)

# observe() is RAM-only. commit() runs the salience gate and persists.
committed = client.commit([t2.turn_id], session_id=SID)

# Tiered search across working → session → long-term.
hits = client.search(
    "weekly review",
    session_id=SID,
    modes=("working", "session", "long_term"),
    top_k=5,
)
```

Observe-without-commit data is **never persisted**. Per-session state is
RAM-only with a TTL sweeper.

---

## Outcome-conditioned health — `cite()` (1.2)

Opt in with `MYELIN_CITE_LEARNING=1`.

```python
import os
os.environ["MYELIN_CITE_LEARNING"] = "1"

client = MemoryClient(user_id="scott")
m_cursor = client.add("Scott uses Cursor for code review.",
                       confidence=0.7, importance=0.7)

# Cite the memory after using it. Correct → lifts apoptosis health.
client.cite(m_cursor.id, answer_correct=True)
client.cite(m_cursor.id, answer_correct=False)   # incorrect → penalty

# When NO stored memory was cite-worthy, record an abstain event.
client.abstain_event(query="What's Scott's middle name?")
```

Contradiction-dominance invariant: a poisoned memory with 100 correct cites
still quarantines on a single contradiction (asserted at `constants.py`
import time).

Since 1.5, cites also nudge ranking: among memories relevant to
a query, those cited mostly in correct answers move up and those cited in
incorrect ones move down. Uncited memories keep their position, feedback
never pulls in a memory the query did not match, and one cite has little
effect (full strength at five). Pass `MemoryClient(cite_ranking=False)` to
use cites for forgetting only.

Measured on held-out LoCoMo questions with no model
([`bench/results/fusion.md`](bench/results/fusion.md)): when a cite says
which memory was useful, all of a question's evidence is in the top 5 for
47.6% of questions against 42.1% without feedback (hybrid search). When
every memory shown for a question gets the same verdict, there is no
gain. Cite the memories an answer actually used, not everything that was
retrieved. `cite_used` does that, and an answering model can supply the
list itself:

```python
hits = client.recall(question, token_budget=6000)
prompt = build_answer_prompt(
    format_context(hits, numbered=True), question, ask_used=True
)
reply = llm(prompt)                       # ... ends with "Used: 2, 5"
client.cite_used(hits, parse_used(reply), query=question)
answer = strip_used(reply)
```

Memories the answer used are cited as useful. The others are counted as
shown but not used: they rank lower the next time they are relevant, and
their health is untouched, so being unneeded for a question never leads
to a memory being forgotten. With `gpt-4o-mini` naming what it used, the
gain matched the upper bound (47.8% against 47.7%).

### One grade: age, use and relationships

`prune()` decides what to hide from a single health grade per memory, in
which three things are weighed together:

| Signal | Effect on health |
|---|---|
| **Age** | falls once a memory is more than a month old |
| **Use** | rises each time the memory is cited as used; recent use counts extra |
| **Relationships** | rises when its neighbours are used: messages close to it in the same conversation (`metadata["session_id"]` and `"turn"`, which `add_conversation` records), and memories with similar meaning when embeddings are stored |

So a message nobody cited outlives an unrelated one of the same age if
it sits beside one that keeps proving useful. Being retrieved and not
used lowers a memory's rank but not its health. Use and related use
together are
capped below the weight of one contradiction, so a contradicted fact
cannot survive on popularity.

Measured by replaying months-long conversations on a simulated clock
(`python -m bench.lifetime`, [`bench/results/lifetime.md`](bench/results/lifetime.md)).
When questions come back to the same facts, forgetting on its own had
hidden a memory that was needed on 14.1% of repeat askings; with cite
feedback keeping used memories healthy, 2.0%, with a tenth of the store
pruned. Feedback also put all of a question's evidence in the top 5 for
53.7% of repeat askings against 44.1%.

---

## HTTP sidecar + `/metrics` (1.3 + 1.4)

Wraps the client as JSON-over-HTTP, OpenClaw-compatible. The 1.4 addition
is `GET /metrics` for vacuous-PASS preflight guards.

```bash
myelin-serve --host 127.0.0.1 --port 8765   # console script (1.3.1+)
# or: python -m myelin.serve --host 127.0.0.1 --port 8765
```

```bash
curl http://127.0.0.1:8765/health    # {"ok": true, "service": "myelin-sidecar"}
curl http://127.0.0.1:8765/metrics   # {"tool_invocations": 0, "reset_count": 0}

curl -X POST http://127.0.0.1:8765/tools/memory_store \
  -H 'content-type: application/json' \
  -d '{"content": "Scott uses Cursor.", "user_id": "scott"}'

curl http://127.0.0.1:8765/metrics   # {"tool_invocations": 1, "reset_count": 0}
```

Loopback only by default. No auth — wrap behind a reverse proxy if exposed.
Bench harnesses can assert `tool_invocations > 0` before treating an
"all-PASS" result as non-vacuous.

Endpoints: `/tools/memory_{search,recall,get,store,store_many,update,forget,cite_used,consolidate,prune}`,
`/hooks/{before_agent_start,agent_end}`, `/admin/reset`, `/health`,
`/metrics` (1.4).

---

## API surface

```python
# Core CRUD
client.add(content, *, user_id=None, metadata=None, event=None,
           confidence=0.5, importance=0.5, outcome=None,
           t_valid=None, index_text=None) -> Memory                       # outcome= 1.4
client.add_many(items, *, user_id=None) -> list[Memory]   # text or dicts of add() arguments
client.add_conversation(turns, *, user_id=None, when=None, llm=None, people=None) -> list[Memory]
client.cite_used(results, used, *, session_id=None, query=None) -> dict  # which shown memories an answer used
client.set_index_text(memory_id, index_text) -> bool      # extra text to be found by
client.search(query, *, user_id=None, top_k=10, as_of=None,
              session_id=None, modes=("working","session","long_term")) -> list[SearchResult]
client.recall(query, *, token_budget=6000, max_results=100,
              max_result_tokens=100, whole_results=3, user_id=None,
              as_of=None, session_id=None,
              modes=("long_term",)) -> list[SearchResult]
format_context(results) -> str        # from myelin import format_context
build_answer_prompt(history, question, *, now=None,
                    style="reasoned") -> str
client.get(memory_id) -> Memory | None
client.update(memory_id, *, content=None, metadata=None) -> Memory
client.delete(memory_id) -> bool

# Consolidation + apoptosis
client.consolidate(*, user_id=None, max_samples=32,
                   retriever=None) -> ConsolidationReport                  # retriever= 1.4
client.prune(*, user_id=None, now=None, caspase_window_days=7) -> ApoptosisReport

# Outcome-conditioned learning (1.2 — opt in via MYELIN_CITE_LEARNING=1)
client.cite(memory_id, *, answer_correct: bool) -> None
client.abstain_event(query, *, at=None) -> None
client.cite_telemetry() -> dict

# Session-scoped working memory (1.2 — opt in via MYELIN_WORKING_MEMORY=1)
client.observe(content, *, session_id, metadata=None, user_id=None) -> ObservedTurn
client.commit(turn_ids, *, session_id) -> list[Memory]
client.attend(memory_id, *, session_id) -> None
client.session(session_id) -> SessionHandle
client.search_telemetry() -> SearchTelemetry

# Constructor (1.4 additions noted)
MemoryClient(
    user_id=None, backend=None, llm_client=None,
    salience_mode="off",                # "off" | "heuristic" | "dmem"
    salience_tau=0.40,
    salience_require_reward=False,
    salience_embedder=None,             # 1.4 — override default embedder
    salience_neighbor_window=32,        # 1.4 — rolling per-scope window
    dmem_gate=None,
    cite_learning=None, working_memory=None,
)
```

```python
# Agent orchestration (1.3)
from myelin.agent import OrchestratedAgent
agent = OrchestratedAgent(client, model="openai/gpt-4o" | "google/gemini-3.5-flash")
agent.classify(question) -> "factual" | "synthesis"
agent.answer(question, *, options=None, user_id=None) -> str
agent.last_usage  # {"input_tokens", "output_tokens", "cost_usd", ...}
```

---

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

All tests are stdlib + pytest only — no internet, no GPU, no LLM calls.
Dream-synthesis tests use trivial mock LLMs. The suite runs on every pull
request (Python 3.11 and 3.13).

---

## Project

- **Source / Issues:** https://github.com/myelinagent/Myelin-AI
- **Changelog:** [`CHANGELOG.md`](https://github.com/myelinagent/Myelin-AI/blob/main/CHANGELOG.md)

## License

MIT. See [`LICENSE`](https://github.com/myelinagent/Myelin-AI/blob/main/LICENSE).
