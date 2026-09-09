from pathlib import Path

from skill_control_plane.evals import load_stage_transition_gold


def test_stage_transition_v03_has_expected_audited_coverage():
    cases = load_stage_transition_gold(
        Path("evals/gold/stage-transition-v0.3.jsonl")
    )
    assert len(cases) == 9

    post_s1_targets = [
        skill_id
        for case in cases
        for stage in case.stages[1:]
        for skill_id in stage.new_required
    ]
    assert len(post_s1_targets) == 13

    added_targets = {
        "document-to-action-items",
        "notion",
        "ocr-and-documents",
        "himalaya",
        "email-inbox-triage",
        "obsidian",
        "llm-wiki",
        "apple-notes",
        "systematic-debugging",
        "github-issue-to-pr",
    }
    existing_targets = {
        "google-workspace",
        "python-debugpy",
        "github-code-review",
    }

    assert set(post_s1_targets) == added_targets | existing_targets
    assert len(added_targets) == 10
