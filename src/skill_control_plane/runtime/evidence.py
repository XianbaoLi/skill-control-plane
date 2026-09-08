from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidencePool:
    """Runtime evidence available for the next capability-retrieval pass.

    Main-Agent interpretation is intentionally kept beside raw evidence so the
    control plane can reuse reasoning that already happened during execution.
    """

    raw_runtime: tuple[str, ...] = ()
    structured_signals: tuple[str, ...] = ()
    agent_interpretation: tuple[str, ...] = ()

    def direct_retrieval_text(self) -> str:
        parts = [
            *self.raw_runtime,
            *self.structured_signals,
            *self.agent_interpretation,
        ]
        return "\n".join(part.strip() for part in parts if part.strip())

    @property
    def is_empty(self) -> bool:
        return not (
            self.raw_runtime
            or self.structured_signals
            or self.agent_interpretation
        )
