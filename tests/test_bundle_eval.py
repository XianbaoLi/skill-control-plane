from __future__ import annotations

from skill_control_plane.evals import StageGold
from skill_control_plane.evals.bundles import evaluate_bundle_trajectory
from skill_control_plane.models import RetrievalCandidate, SkillRecord


def _record(skill_id: str, category: str) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        description="",
        body="",
        source_path=f"/skills/{category}/{skill_id}/SKILL.md",
        category=category,
    )


def _candidate(skill_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)


def test_bundle_trajectory_measures_shelf_reuse_and_new_bundle_rate() -> None:
    stages = (
        StageGold(
            stage_id="S1",
            runtime_evidence=(),
            required_now=("pytest",),
            new_required=("pytest",),
            useful=(),
            hard_negative=(),
        ),
        StageGold(
            stage_id="S2",
            runtime_evidence=("runtime error",),
            required_now=("pytest", "cuda"),
            new_required=("cuda",),
            useful=(),
            hard_negative=(),
        ),
        StageGold(
            stage_id="S3",
            runtime_evidence=("oauth failure",),
            required_now=("oauth",),
            new_required=("oauth",),
            useful=(),
            hard_negative=(),
        ),
    )
    records = {
        "pytest": _record("pytest", "testing"),
        "dependency": _record("dependency", "runtime"),
        "cuda": _record("cuda", "runtime"),
        "oauth": _record("oauth", "api-access"),
    }
    candidates = {
        "S1": [_candidate("pytest", 1), _candidate("dependency", 2)],
        "S2": [_candidate("cuda", 1)],
        "S3": [_candidate("oauth", 1)],
    }

    report = evaluate_bundle_trajectory(stages, candidates, records)

    assert report["transition_skill_count"] == 2
    assert report["shelf_reuse_rate"] == 0.5
    assert report["new_bundle_rate"] == 0.5
    assert report["final_registered_skill_count"] == 4
    assert report["stages"][1]["active_bundle_ids"] == ["testing", "runtime"]
    assert report["stages"][2]["active_bundle_ids"] == [
        "testing",
        "runtime",
        "api-access",
    ]
