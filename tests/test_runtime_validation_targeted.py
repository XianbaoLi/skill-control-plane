"""Focused contract tests for the final runtime-validation harness."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_failure_is_an_outer_runner_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = load_script("benchmark_run_failure", "benchmark_run.py")
    monkeypatch.setattr(runner, "run_one", lambda **_: {"task_success": False})
    monkeypatch.setattr(sys, "argv", ["benchmark_run.py", "one", "--task", "RT-T01", "--arm", "control-plane",
                                       "--corpus", "S128", "--output", str(tmp_path)])
    assert runner.main() == 1


def test_verifier_success_is_an_outer_runner_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = load_script("benchmark_run_success", "benchmark_run.py")
    monkeypatch.setattr(runner, "run_one", lambda **_: {"task_success": True})
    monkeypatch.setattr(sys, "argv", ["benchmark_run.py", "one", "--task", "RT-T01", "--arm", "control-plane",
                                       "--corpus", "S128", "--output", str(tmp_path)])
    assert runner.main() == 0


def test_source_marker_is_not_runtime_evidence() -> None:
    targeted = load_script("targeted_source", "run_runtime_validation_targeted.py")
    summary = targeted.phase6_summary({"task_id": "RT-T01"}, {"events": [
        {"event_seq": 1, "type": "evidence_emitted", "event_id": "E1", "source": "source_read"},
    ]}, {"reroute_events": []})
    assert summary["evidence_observed"] is False


def test_gated_verifier_marker_is_runtime_evidence() -> None:
    targeted = load_script("targeted_runtime", "run_runtime_validation_targeted.py")
    marker = {"type": "evidence_emitted", "event_id": "E1", "source": "gated_verifier_execution"}
    assert targeted.runtime_e1([marker]) == marker
    assert targeted.phase6_summary({"task_id": "RT-T01"}, {"events": [
        {"event_seq": 1, **marker},
    ]}, {"reroute_events": []})["evidence_observed"] is True


def test_checkpoint_request_is_reported_when_completion_is_missing() -> None:
    targeted = load_script("targeted_request", "run_runtime_validation_targeted.py")
    summary = targeted.phase6_summary(
        {"task_id": "RT-T01", "task_success": False},
        {"events": [
            {"event_seq": 1, "type": "evidence_emitted", "event_id": "E1", "source": "gated_verifier_execution"},
            {"event_seq": 2, "type": "capability_gap_check_requested"},
        ]}, {"reroute_events": []})
    assert summary["checkpoint_request_count"] == 1
    assert summary["checkpoint_completed_count"] == 0
    assert summary["gap_check_requested_after_e1"] is True
    assert summary["gap_check_completed_after_e1"] is False


def test_completed_checkpoint_is_reported_separately() -> None:
    targeted = load_script("targeted_complete", "run_runtime_validation_targeted.py")
    summary = targeted.phase6_summary(
        {"task_id": "RT-T01", "task_success": True},
        {"events": [
            {"event_seq": 1, "type": "evidence_emitted", "event_id": "E1", "source": "gated_verifier_execution"},
            {"event_seq": 2, "type": "capability_gap_check_requested"},
            {"event_seq": 3, "type": "capability_gap_check", "needs_capability": True,
             "generated_need": "monitoring"},
        ]}, {"reroute_events": []})
    assert summary["checkpoint_request_count"] == 1
    assert summary["checkpoint_completed_count"] == 1
    assert summary["needs_capability"] == [True]
    assert summary["generated_need"] == ["monitoring"]


def test_final_targeted_matrix_has_six_600_second_cells() -> None:
    targeted = load_script("targeted_identity", "run_runtime_validation_targeted.py")
    assert len(targeted.cells()) == 6
    assert targeted.TIMEOUT_SECONDS == 600
    assert targeted.MAX_TURNS == 20
    assert [(cell["task_id"], cell["restore"]) for cell in targeted.cells()[:2]] == [
        ("RT-R01", "on"), ("RT-R01", "off")]
