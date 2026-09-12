from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from skill_control_plane.evals.legacy.stage_retrieval import (
    RetrievalPhase,
    StageRetrievalResult,
)


def summarize_retrieval_escalation(
    results: Sequence[StageRetrievalResult],
) -> dict[str, Any]:
    """Summarize how often each semantic-cost tier resolved a transition."""

    total = len(results)
    counts = {phase.value: 0 for phase in RetrievalPhase}
    successful_counts = {phase.value: 0 for phase in RetrievalPhase}

    for result in results:
        counts[result.phase.value] += 1
        if result.traces and result.traces[-1].sufficient:
            successful_counts[result.phase.value] += 1

    denominator = total or 1
    return {
        "transition_count": total,
        "raw_resolution_rate": counts[RetrievalPhase.RAW.value] / denominator,
        "agent_reuse_rate": counts[RetrievalPhase.AGENT_AUGMENTED.value] / denominator,
        "src_escalation_rate": counts[RetrievalPhase.SRC_ENHANCED.value] / denominator,
        "phase_counts": counts,
        "sufficient_phase_counts": successful_counts,
    }
