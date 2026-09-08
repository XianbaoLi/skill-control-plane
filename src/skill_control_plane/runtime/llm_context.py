from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from skill_control_plane.runtime.evidence import EvidencePool
from skill_control_plane.runtime.stage_retrieval import StageRetrievalContext


TextCompleter = Callable[[str], str]


def build_src_prompt(evidence: EvidencePool) -> str:
    """Build a narrow prompt for semantic Skill-retrieval enhancement."""

    return f"""You are generating retrieval context for an Agent Skill library.

Given runtime evidence, identify the capability areas that may be needed next.

Rules:
- Do not solve the underlying task.
- Do not explain the full root cause.
- Return a short retrieval-oriented phrase or sentence.
- Prefer concrete capability language suitable for searching Skill descriptions.
- Preserve uncertainty when the evidence is ambiguous.

Runtime evidence:
{evidence.direct_retrieval_text()}

Stage Retrieval Context:"""


@dataclass(slots=True)
class LLMStageContextEnhancer:
    """Provider-neutral adapter around a text completion function.

    The control plane owns only the narrow SRC prompt. Authentication, provider
    choice, retries, token accounting, and model configuration stay outside this
    class so runtime routing is not coupled to one LLM vendor.
    """

    complete: TextCompleter

    def build_context(self, evidence: EvidencePool) -> StageRetrievalContext:
        text = self.complete(build_src_prompt(evidence)).strip()
        return StageRetrievalContext(text=text, source="llm")
