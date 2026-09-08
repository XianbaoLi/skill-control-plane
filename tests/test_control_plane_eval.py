from __future__ import annotations

from skill_control_plane.evals import (
    MultiSkillGoldCase,
    StageGold,
    StageTransitionGoldCase,
    evaluate_control_plane,
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


def test_control_plane_measures_multi_skill_and_reroute_gain() -> None:
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
    stage = [
        StageTransitionGoldCase(
            case_id="ST",
            initial_task="start task",
            stages=(
                StageGold(
                    stage_id="S1",
                    observed_state="initial state",
                    transition_trigger=None,
                    required_now=("skill-a",),
                    new_required=("skill-a",),
                    useful=(),
                    hard_negative=(),
                ),
                StageGold(
                    stage_id="S2",
                    observed_state="skill-b is now needed",
                    transition_trigger="new failure appears",
                    required_now=("skill-a", "skill-b"),
                    new_required=("skill-b",),
                    useful=(),
                    hard_negative=(),
                ),
            ),
            rationale="",
            snapshot_id="snapshot",
        )
    ]

    bm25 = FakeRetriever(
        {
            "multi task": ["skill-a", "noise-a"],
            "start task": ["skill-a", "noise-a"],
            "start task\ninitial state": ["skill-a", "noise-a"],
            "start task\nnew failure appears\nskill-b is now needed": [
                "skill-a",
                "noise-a",
            ],
        }
    )
    dense = FakeRetriever(
        {
            "multi task": ["skill-b", "noise-b"],
            "start task": ["noise-b"],
            "start task\ninitial state": ["noise-b"],
            "start task\nnew failure appears\nskill-b is now needed": [
                "skill-b",
                "noise-b",
            ],
        }
    )

    report = evaluate_control_plane(
        multi,
        stage,
        bm25=bm25,
        dense=dense,
        k=2,
    )

    multi_report = report["multi_skill"]
    assert multi_report["required_skill_recall"] == 1.0
    assert multi_report["full_required_set_coverage"] == 1.0

    stage_report = report["stage_transition"]
    assert stage_report["one_shot_stage_full_coverage"] == 0.5
    assert stage_report["reroute_stage_full_coverage"] == 1.0
    assert stage_report["reroute_gain"] == 0.5
    assert stage_report["new_skill_recovery"] == 1.0
    assert stage_report["new_required_skill_hits"] == 1
    assert stage_report["stages"][0]["recovery_target"] == []
    assert stage_report["stages"][1]["recovery_target"] == ["skill-b"]

    assert report["activation_metrics_available"] is False
