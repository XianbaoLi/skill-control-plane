from __future__ import annotations

from skill_control_plane.evals import (
    MultiSkillGoldCase,
    StageGold,
    StageTransitionGoldCase,
    evaluate_control_plane,
    evaluate_stage_reroute,
)
from skill_control_plane.models import RetrievalCandidate


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)
            for rank, skill_id in enumerate(self.results.get(query, [])[:k], start=1)
        ]


def _stage_case() -> StageTransitionGoldCase:
    return StageTransitionGoldCase(
        case_id="ST",
        initial_task="start task",
        stages=(
            StageGold(
                stage_id="S1",
                runtime_evidence=(),
                required_now=("skill-a",),
                new_required=("skill-a",),
                useful=(),
                hard_negative=(),
            ),
            StageGold(
                stage_id="S2",
                runtime_evidence=("raw failure output",),
                required_now=("skill-a", "skill-b"),
                new_required=("skill-b",),
                useful=(),
                hard_negative=(),
            ),
            StageGold(
                stage_id="S3",
                runtime_evidence=("user asks for final review",),
                required_now=("skill-c",),
                new_required=("skill-c",),
                useful=(),
                hard_negative=(),
            ),
        ),
        rationale="",
        snapshot_id="snapshot",
    )


def test_stage_reroute_uses_raw_evidence_and_measures_incremental_recovery() -> None:
    stage = [_stage_case()]

    bm25 = FakeRetriever(
        {
            "start task": ["skill-a", "skill-c"],
            "start task\nraw failure output": ["skill-a", "noise-a"],
            "start task\nuser asks for final review": ["skill-c", "noise-a"],
        }
    )
    dense = FakeRetriever(
        {
            "start task": ["noise-b"],
            "start task\nraw failure output": ["skill-b", "noise-b"],
            "start task\nuser asks for final review": ["noise-b"],
        }
    )

    report = evaluate_stage_reroute(stage, bm25=bm25, dense=dense, k=2)

    assert report["one_shot_stage_full_coverage"] == 2 / 3
    assert report["reroute_stage_full_coverage"] == 1.0
    assert report["transition_new_skill_recall"] == 1.0

    # skill-b was absent from one-shot and genuinely recovered at S2.
    # skill-c was already present in one-shot, so it is not an incremental target.
    assert report["incremental_recovery"] == 1.0
    assert report["incremental_recovery_target_count"] == 1
    assert report["incremental_recovery_hits"] == 1
    assert report["already_present_transition_skill_count"] == 1

    # S1 is exactly the one-shot query and candidate set.
    s1 = report["stages"][0]
    assert s1["query"] == "start task"
    assert s1["one_shot_candidate_ids"] == s1["reroute_candidate_ids"]


def test_control_plane_still_supports_multi_skill_report() -> None:
    multi = [
        MultiSkillGoldCase(
            case_id="MS",
            initial_task="multi task",
            required=("skill-a", "skill-b"),
            useful=(),
            hard_negative=(),
            rationale="",
            snapshot_id="snapshot",
        )
    ]
    stage = [_stage_case()]

    bm25 = FakeRetriever(
        {
            "multi task": ["skill-a", "noise-a"],
            "start task": ["skill-a", "skill-c"],
            "start task\nraw failure output": ["skill-a", "noise-a"],
            "start task\nuser asks for final review": ["skill-c", "noise-a"],
        }
    )
    dense = FakeRetriever(
        {
            "multi task": ["skill-b", "noise-b"],
            "start task": ["noise-b"],
            "start task\nraw failure output": ["skill-b", "noise-b"],
            "start task\nuser asks for final review": ["noise-b"],
        }
    )

    report = evaluate_control_plane(
        multi,
        stage,
        bm25=bm25,
        dense=dense,
        k=2,
    )

    assert report["multi_skill"]["required_skill_recall"] == 1.0
    assert report["stage_transition"]["reroute_stage_full_coverage"] == 1.0
    assert report["activation_metrics_available"] is False
