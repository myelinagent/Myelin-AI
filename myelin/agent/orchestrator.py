"""Classifier-routed multi-step orchestrator.

Implements the v1.6 pattern from the PersonaMem benchmark series. Briefly:

1. A cheap classifier (gpt-4o-mini by default) labels the question as
   ``"factual"`` (recall-oriented) or ``"synthesis"`` (recommend / generalize).
2. **Factual** questions go through a multi-step loop where the LLM
   decides when to call ``memory_search`` as a tool.
3. **Synthesis** questions get a top-K prefetch (default 10) of memories
   prepended to the prompt; the tool stays available for refinement.

Provider-agnostic — pass ``model="openai/gpt-4o"`` or
``model="google/gemini-3.5-flash"``. The instance auto-loads the right SDK
and translates the function-calling shapes internally.

Returns the model's free-form answer text. Token usage is tracked in
``OrchestratedAgent.last_usage``. Cost is computed only if pricing is
provided via ``OrchestratedAgent.PRICE_TABLE`` (caller can override).
"""
from __future__ import annotations

import json
import os
from typing import Any

from myelin.client import MemoryClient

# Provider pricing (USD per 1M tokens). Override on the class for custom
# rates. These are list prices as of 2026-05.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-2.5-flash": (0.30, 1.00),
}


_OPENAI_TIMEOUT_SECONDS = 60.0
_OPENAI_MAX_RETRIES = 2

# Bounds for the model-supplied ``top_k`` of a memory_search tool call.
_TOOL_DEFAULT_TOP_K = 5
_TOOL_MAX_TOP_K = 50


# ---------- Memory tool surface exposed to the LLM ------------------------

_MEMORY_SEARCH_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search the user's memory substrate for facts relevant to the "
            "query. Returns top-K relevant memories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

# System prompts (tested at gpt-4o + gemini-3.5-flash; see PersonaMem
# v1.4/v1.5 commits for the wording validation).
_SYSTEM_PROMPT_MULTISTEP = (
    "You are answering a question about a user. You have access to a "
    "memory_search tool that retrieves facts from the user's conversation "
    "history. Use memory_search 1-3 times to gather relevant context, then "
    "commit to an answer.\n\n"
    "If the question is multiple choice, reply with ONLY the letter in "
    "parentheses — e.g. (b) — after the special token <final_answer>. "
    "Otherwise reply with a concise direct answer after <final_answer>."
)

_SYSTEM_PROMPT_HYBRID = (
    "You are answering a question about a user. You have been given the "
    "most relevant memories from this user's history below. You may "
    "optionally call the memory_search tool to retrieve additional context "
    "if the initial memories are insufficient — but for many questions the "
    "initial set is enough; do not search unnecessarily.\n\n"
    "If the question is multiple choice, reply with ONLY the letter in "
    "parentheses — e.g. (b) — after the special token <final_answer>. "
    "Otherwise reply with a concise direct answer after <final_answer>."
)


_CLASSIFY_PROMPT = (
    "You will see a question about a user. Classify the question's intent "
    "into exactly one of:\n"
    "  factual    — asks the assistant to recall or reason over specific "
    "facts from the user's history.\n"
    "  synthesis  — asks the assistant to recommend, suggest, generalize, "
    "or extrapolate based on the user's history.\n"
    "Reply with ONLY the single word: factual OR synthesis."
)


class OrchestratedAgent:
    """Classifier-routed, memory-aware, multi-step agent.

    Parameters
    ----------
    client : MemoryClient
        Backing memory substrate.
    model : str
        Answerer in ``"<provider>/<model>"`` form. Supports ``openai/*``
        and ``google/*`` providers.
    classifier_model : str, default ``"openai/gpt-4o-mini"``
        Cheap model used for intent routing. Set to the same provider as
        ``model`` to avoid mixing providers (e.g. if only Gemini creds
        available, pass ``classifier_model="google/gemini-3.5-flash"``).
    prefetch_topk : int, default 10
        Top-K for the synthesis-route prefetch.
    max_tool_steps : int, default 3
        Cap on multi-step tool-call iterations.
    """

    PRICE_TABLE = DEFAULT_PRICES

    def __init__(
        self,
        client: MemoryClient,
        *,
        model: str = "openai/gpt-4o",
        classifier_model: str = "openai/gpt-4o-mini",
        prefetch_topk: int = 10,
        max_tool_steps: int = 3,
    ) -> None:
        self._memory = client
        self._model_spec = model
        self._classifier_spec = classifier_model
        self._prefetch_topk = prefetch_topk
        self._max_tool_steps = max_tool_steps
        self.last_usage: dict[str, Any] = {}

        self._provider, _, self._model = model.partition("/")
        self._clf_provider, _, self._clf_model = classifier_model.partition("/")

        self._llm = self._make_client(self._provider)
        if self._clf_provider == self._provider:
            self._clf = self._llm
        else:
            self._clf = self._make_client(self._clf_provider)

    # ---- client factory ----------------------------------------------

    @staticmethod
    def _make_client(provider: str) -> Any:
        if provider == "openai":
            from openai import OpenAI  # type: ignore

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            # Bounded like the Gemini client below: without an explicit
            # timeout the SDK waits 10 minutes per attempt on a hung socket.
            return OpenAI(
                api_key=api_key,
                timeout=_OPENAI_TIMEOUT_SECONDS,
                max_retries=_OPENAI_MAX_RETRIES,
            )
        if provider == "google":
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY missing")
            # 30s per-request timeout — guards against the SDK's internal
            # retry-on-disconnect hang (see feedback_gemini_sdk_timeout_swallowed).
            return genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=30000),
            )
        raise ValueError(f"unsupported provider {provider!r}; expected openai or google")

    # ---- public API --------------------------------------------------

    def classify(self, question: str) -> str:
        """Return ``"factual"`` or ``"synthesis"`` for ``question``."""
        if self._clf_provider == "openai":
            r = self._clf.chat.completions.create(
                model=self._clf_model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nIntent:"},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            text = (r.choices[0].message.content or "").strip().lower()
        else:
            r = self._clf.models.generate_content(
                model=self._clf_model,
                contents=f"{_CLASSIFY_PROMPT}\n\nQuestion: {question}\n\nIntent:",
            )
            text = (getattr(r, "text", None) or "").strip().lower()
        return "synthesis" if "synthesis" in text else "factual"

    def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        options: str | None = None,
    ) -> str:
        """Answer ``question`` with classifier-routed memory orchestration.

        Returns the LLM's free-form reply. ``options`` (a string of
        line-separated choices like ``"(a) ...\\n(b) ..."``) is optional;
        when present it is appended to the user message and the answerer
        is prompted to reply with the letter only.
        """
        intent = self.classify(question)
        prefetch = self._prefetch_topk if intent == "synthesis" else 0
        return self._answer_one(
            question=question,
            options=options,
            user_id=user_id or self._memory._user_id,
            prefetch_topk=prefetch,
        )

    # ---- multi-step inner loop ---------------------------------------

    def _answer_one(
        self,
        question: str,
        options: str | None,
        user_id: str,
        prefetch_topk: int,
    ) -> str:
        prefetched: list[str] = []
        if prefetch_topk > 0:
            hits = self._memory.search(
                query=question, user_id=user_id, top_k=prefetch_topk,
            )
            for h in (hits or [])[:prefetch_topk]:
                mem = getattr(h, "memory", None) or h
                c = getattr(mem, "content", None)
                if c:
                    prefetched.append(f"- {c}")

        if prefetched:
            memories_block = "Initial relevant memories:\n" + "\n".join(prefetched) + "\n\n"
            sys_prompt = _SYSTEM_PROMPT_HYBRID
        else:
            memories_block = ""
            sys_prompt = _SYSTEM_PROMPT_MULTISTEP

        user_msg = f"{memories_block}Question: {question}"
        if options:
            user_msg += f"\n\nOptions:\n{options}"

        if self._provider == "openai":
            return self._loop_openai(sys_prompt, user_msg, user_id)
        return self._loop_gemini(sys_prompt, user_msg, user_id)

    # ---- OpenAI multi-step loop --------------------------------------

    def _loop_openai(self, system_prompt: str, user_msg: str, user_id: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        tools = [_MEMORY_SEARCH_TOOL_OPENAI]
        total_in = 0
        total_out = 0
        n_tool_calls = 0

        for _ in range(self._max_tool_steps + 1):
            r = self._llm.chat.completions.create(
                model=self._model, messages=messages, tools=tools,
                temperature=0.0, max_tokens=400,
            )
            usage = r.usage
            total_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            total_out += int(getattr(usage, "completion_tokens", 0) or 0)
            msg = r.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs:
                self.last_usage = {
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "n_tool_calls": n_tool_calls,
                    "provider": "openai",
                    "model": self._model,
                    "cost_usd": self._cost("openai", self._model, total_in, total_out),
                }
                return msg.content or ""
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.function.name, "arguments": tc.function.arguments,
                    }}
                    for tc in tcs
                ],
            })
            for tc in tcs:
                n_tool_calls += 1
                tool_text = self._execute_memory_tool(tc.function.name,
                                                     tc.function.arguments,
                                                     user_id=user_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_text,
                })

        # Force final answer
        r = self._llm.chat.completions.create(
            model=self._model,
            messages=messages + [{
                "role": "user",
                "content": "Commit to an answer NOW after <final_answer>.",
            }],
            temperature=0.0,
            max_tokens=80,
        )
        usage = r.usage
        total_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        total_out += int(getattr(usage, "completion_tokens", 0) or 0)
        self.last_usage = {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "n_tool_calls": n_tool_calls,
            "provider": "openai",
            "model": self._model,
            "cost_usd": self._cost("openai", self._model, total_in, total_out),
        }
        return r.choices[0].message.content or ""

    # ---- Gemini multi-step loop --------------------------------------

    def _loop_gemini(self, system_prompt: str, user_msg: str, user_id: str) -> str:
        from google.genai import types  # type: ignore

        tools = [types.Tool(function_declarations=[types.FunctionDeclaration(
            name=_MEMORY_SEARCH_TOOL_OPENAI["function"]["name"],
            description=_MEMORY_SEARCH_TOOL_OPENAI["function"]["description"],
            parameters_json_schema=_MEMORY_SEARCH_TOOL_OPENAI["function"]["parameters"],
        )])]
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            max_output_tokens=600,
        )
        try:
            cfg.thinking_config = types.ThinkingConfig(thinking_level="low")
        except Exception:
            pass

        contents: list[Any] = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]),
        ]
        total_in = 0
        total_out = 0
        n_tool_calls = 0

        for _ in range(self._max_tool_steps + 1):
            r = self._llm.models.generate_content(
                model=self._model, contents=contents, config=cfg,
            )
            um = getattr(r, "usage_metadata", None)
            total_in += int(getattr(um, "prompt_token_count", 0) or 0)
            total_out += int(getattr(um, "candidates_token_count", 0) or 0)
            total_out += int(getattr(um, "thoughts_token_count", 0) or 0)

            cand = r.candidates[0] if r.candidates else None
            if cand is None or cand.content is None:
                break
            parts = cand.content.parts or []
            fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if not fcs:
                self.last_usage = {
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "n_tool_calls": n_tool_calls,
                    "provider": "google",
                    "model": self._model,
                    "cost_usd": self._cost("google", self._model, total_in, total_out),
                }
                return "".join(p.text for p in parts if getattr(p, "text", None))

            contents.append(cand.content)
            response_parts = []
            for fc in fcs:
                n_tool_calls += 1
                tool_text = self._execute_memory_tool(
                    fc.name, fc.args or {}, user_id=user_id,
                )
                response_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": tool_text},
                ))
            contents.append(types.Content(role="user", parts=response_parts))

        # Force final answer
        contents.append(types.Content(role="user", parts=[
            types.Part.from_text(text="Commit to an answer NOW after <final_answer>."),
        ]))
        final_cfg = types.GenerateContentConfig(
            system_instruction=system_prompt, max_output_tokens=120,
        )
        r = self._llm.models.generate_content(
            model=self._model, contents=contents, config=final_cfg,
        )
        um = getattr(r, "usage_metadata", None)
        total_in += int(getattr(um, "prompt_token_count", 0) or 0)
        total_out += int(getattr(um, "candidates_token_count", 0) or 0)
        total_out += int(getattr(um, "thoughts_token_count", 0) or 0)
        text = (getattr(r, "text", None) or "").strip()
        self.last_usage = {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "n_tool_calls": n_tool_calls,
            "provider": "google",
            "model": self._model,
            "cost_usd": self._cost("google", self._model, total_in, total_out),
        }
        return text

    # ---- shared helpers ----------------------------------------------

    def _execute_memory_tool(
        self,
        tool_name: str,
        arguments: Any,
        user_id: str,
    ) -> str:
        """Execute a memory tool call from the LLM and return rendered text."""
        # Arguments come from the model, so anything malformed is
        # reported back to it as a tool error instead of raising.
        args = arguments
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        try:
            args = dict(args or {})
        except (TypeError, ValueError):
            args = {}
        if tool_name != "memory_search":
            return f"unknown tool {tool_name}"
        query = args.get("query")
        if not isinstance(query, str) or not query:
            return "memory_search error: missing query"
        try:
            top_k = int(args.get("top_k", _TOOL_DEFAULT_TOP_K))
        except (TypeError, ValueError):
            top_k = _TOOL_DEFAULT_TOP_K
        top_k = max(1, min(top_k, _TOOL_MAX_TOP_K))
        try:
            hits = self._memory.search(
                query=query, user_id=user_id, top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            return f"memory_search error: {exc}"
        bullets = []
        for h in (hits or [])[:top_k]:
            mem = getattr(h, "memory", None) or h
            c = getattr(mem, "content", None)
            if c:
                bullets.append(f"- {c}")
        return "Memories:\n" + ("\n".join(bullets) if bullets else "(none)")

    @classmethod
    def _cost(cls, provider: str, model: str, inp: int, out: int) -> float:
        price = cls.PRICE_TABLE.get(model)
        if not price:
            return 0.0
        return (inp * price[0] + out * price[1]) / 1_000_000
