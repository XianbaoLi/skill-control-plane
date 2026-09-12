from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from skill_control_plane.benchmarks.models import VERSION, load_tasks, validate_offline_cost, validate_result
from skill_control_plane.benchmarks.reporter import paired_report
from skill_control_plane.benchmarks.scorer import repeated_discovery_count, required_skill_recall
from scripts.benchmark_validate import validate

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evals/benchmarks/v0.1"


def _result(**updates):
    row = {
        "benchmark_version": VERSION, "task_id": "SC-01", "suite": "scaling",
        "arm": "native", "corpus": "S32", "model": "fixture", "run_id": "r1",
        "task_success": True, "success_details": [], "required_skills": ["a"],
        "activated_skills": ["a"], "required_skill_recall": 1.0,
        "wrong_skill_count": 0, "discovery_count": 1, "repeated_discovery_count": 0,
        "skill_body_load_count": 1, "stage_transition_success": None,
        "bundle_create_count": 0, "bundle_extend_count": 0, "bundle_reuse_count": 0,
        "session_restore_success": None, "llm_input_tokens": 10, "llm_output_tokens": 2,
        "llm_total_tokens": 12, "cached_tokens": None, "query_embedding_calls": 0,
        "wall_time_ms": 20, "total_tool_calls": 2,
        "control_plane_telemetry": {"retrieval_calls": 0, "retrieval_events": None},
    }
    row.update(updates)
    return row


def test_manifests_and_leakage_validate():
    report = validate()
    assert report["scaling_tasks"] == 16
    assert report["runtime_scenarios"] == 20
    assert report["runtime_counts"] == {"single-skill": 6, "multi-skill": 5,
                                         "stage-transition": 5, "reuse-session": 4}
    assert all(row["status"] == "pass" for row in report["leakage_checks"])


def test_scaling_targets_are_nested_s32_tasks():
    manifest = json.loads((ROOT / "evals/corpora/benchmark-corpus-v0.1.json").read_text())
    subsets = {row["skill_id"]: set(row["subsets"]) for row in manifest["skills"]}
    tasks = load_tasks(BENCH / "scaling.jsonl")
    assert len(tasks) == 16
    assert all({"S32", "S64", "S128"} <= subsets[t.raw["target_skill"]] for t in tasks)
    assert all("corpus" not in t.raw["prompt"].casefold() for t in tasks)


def test_runtime_shape_and_required_sets():
    tasks = load_tasks(BENCH / "runtime.jsonl")
    assert Counter(t.raw["scenario_type"] for t in tasks) == {
        "single-skill": 6, "multi-skill": 5, "stage-transition": 5, "reuse-session": 4}
    assert all(2 <= len(t.required_skills) <= 3 for t in tasks if t.raw["scenario_type"] == "multi-skill")
    assert all(all(stage["required_skills"] for stage in t.raw["stages"])
               for t in tasks if t.raw["scenario_type"] == "stage-transition")
    assert all(len(t.raw["turns"]) >= 2 for t in tasks if t.raw["scenario_type"] == "reuse-session")


def test_mechanism_metrics():
    assert required_skill_recall(["a", "b"], ["b", "x"]) == 0.5
    assert repeated_discovery_count(["api", "db", "api", "api", "db"]) == 3


def test_result_schema_rejects_negative_tokens_and_false_total():
    validate_result(_result())
    with pytest.raises(ValueError, match="llm_input_tokens"):
        validate_result(_result(llm_input_tokens=-1, llm_total_tokens=1))
    with pytest.raises(ValueError, match="must equal"):
        validate_result(_result(llm_total_tokens=99))


def test_offline_unavailable_is_explicit_not_fabricated():
    rows = json.loads((BENCH / "offline-cost.json").read_text())
    for row in rows:
        validate_offline_cost(row)
        assert row["telemetry_status"] == "unavailable"
    fake = dict(rows[0], total_tokens=100)
    with pytest.raises(ValueError, match="must not invent"):
        validate_offline_cost(fake)


def test_paired_aggregation_is_control_plane_minus_native():
    native = _result()
    control = _result(arm="control-plane", run_id="r2", task_success=False,
                      required_skill_recall=0.5, wrong_skill_count=2,
                      llm_input_tokens=15, llm_output_tokens=3, llm_total_tokens=18,
                      wall_time_ms=30,
                      control_plane_telemetry={"retrieval_calls": 1, "retrieval_events": []})
    report = paired_report([native, control], "scaling")
    delta = report["paired_delta_control_plane_minus_native"][0]
    assert delta["task_success"] == -1
    assert delta["required_skill_recall"] == -0.5
    assert delta["llm_total_tokens"] == 6
    assert delta["wall_time_ms"] == 10
