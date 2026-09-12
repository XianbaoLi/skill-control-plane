from skill_control_plane.evals.escalation import summarize_retrieval_escalation
from skill_control_plane.evals.legacy.stage_retrieval import (
    RetrievalPhase,
    RetrievalTrace,
    StageRetrievalResult,
)


def _result(phase: RetrievalPhase, sufficient: bool) -> StageRetrievalResult:
    return StageRetrievalResult(
        candidates=(),
        phase=phase,
        traces=(
            RetrievalTrace(
                phase=phase,
                query="q",
                candidates=(),
                sufficient=sufficient,
            ),
        ),
    )


def test_summarize_retrieval_escalation_separates_cost_tiers() -> None:
    report = summarize_retrieval_escalation(
        [
            _result(RetrievalPhase.RAW, True),
            _result(RetrievalPhase.RAW, True),
            _result(RetrievalPhase.AGENT_AUGMENTED, True),
            _result(RetrievalPhase.SRC_ENHANCED, True),
        ]
    )

    assert report["raw_resolution_rate"] == 0.5
    assert report["agent_reuse_rate"] == 0.25
    assert report["src_escalation_rate"] == 0.25
    assert report["phase_counts"]["raw"] == 2
