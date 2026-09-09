from pathlib import Path

from skill_control_plane.evals import load_stage_transition_gold


def test_systematic_debugging_is_useful_methodology_not_required_capability():
    cases = load_stage_transition_gold(
        Path("evals/gold/stage-transition-v0.2.jsonl")
    )
    case = next(case for case in cases if case.case_id == "ST-01")
    stages = {stage.stage_id: stage for stage in case.stages}

    s2 = stages["S2"]
    assert "systematic-debugging" not in s2.required_now
    assert "systematic-debugging" not in s2.new_required
    assert "systematic-debugging" in s2.useful

    s3 = stages["S3"]
    assert "systematic-debugging" not in s3.required_now
    assert "systematic-debugging" in s3.useful
    assert s3.new_required == ("python-debugpy",)
    assert "python-debugpy" in s3.required_now
