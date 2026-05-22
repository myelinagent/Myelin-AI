"""Memory-aware agent orchestration.

Implements the v1.6 classifier-routed multi-step pattern, validated on
PersonaMem n=589 as the highest-lift orchestration in our benchmark series
(acc 0.7097 with gpt-4o, +2.55 pp over the prior best).

The hypotheses that **didn't** lift (HyDE query expansion, distilled-fact
ingestion, K=3 self-consistency voting) are intentionally not in the
library — see ``data/registry/test-results-registry.jsonl`` for the
falsifying evidence.

Public API::

    from myelin import MemoryClient
    from myelin.agent import OrchestratedAgent

    client = MemoryClient()
    agent = OrchestratedAgent(client=client, model="openai/gpt-4o")
    answer = agent.answer(
        question="What does the user prefer to drink?",
        options="(a) coffee\\n(b) green tea\\n(c) matcha\\n(d) water",
        user_id="u1",
    )
"""
from myelin.agent.orchestrator import OrchestratedAgent

__all__ = ["OrchestratedAgent"]
