from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from skill_control_plane.benchmarks.models import VERSION, load_tasks, validate_offline_cost, validate_result
from skill_control_plane.benchmarks.reporter import paired_report
from skill_control_plane.benchmarks.runner import (assert_gold_isolated,
    assert_identity_matches_run, gated_verifier_server, materialize_fixture,
    materialize_host_verifier)
from skill_control_plane.benchmarks.scorer import _check, repeated_discovery_count, required_skill_recall, score_run
from scripts.benchmark_validate import validate

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evals/benchmarks/v0.1"


def _result(**updates):
    row = {
        "benchmark_version": VERSION, "task_id": "SC-01", "suite": "scaling",
        "arm": "native", "corpus": "S32", "run_id": "r1", "task_success": True,
        "success_details": [], "required_skills": ["a"], "activated_skills": ["a"],
        "required_skill_recall": 1.0, "wrong_skill_count": 0, "discovery_count": 1,
        "repeated_discovery_count": 0, "skill_body_load_count": 1,
        "reroute_success": None, "new_required_skill_recall": None,
        "premature_activation_count": 0, "bundle_create_count": None,
        "bundle_extend_count": None, "bundle_reuse_count": None,
        "session_restore_success": None, "llm_input_tokens": 10, "llm_output_tokens": 2,
        "llm_total_tokens": 12, "cached_tokens": None,
        "query_embedding_calls_startup": 0, "query_embedding_calls_runtime": 0,
        "query_embedding_calls_total": 0, "wall_time_ms": 20, "total_tool_calls": 2,
        "control_plane_telemetry": {"retrieval_calls": None, "retrieval_events": None},
        "pi_package": "@earendil-works/pi-coding-agent", "pi_version": "0.84.1",
        "provider": "fixture", "model": "fixture", "reasoning_config": None,
        "temperature": None, "max_turns": 20, "timeout_seconds": 300,
        "corpus_version": "benchmark-corpus-v0.1", "corpus_subset": "S32",
        "retrieval_card_identity": "cards-sha", "dense_index_identity": "dense-sha",
        "adapter_commit_sha": "abc123",
    }
    row.update(updates)
    return row


def test_manifest_counts_and_no_stage_runtime_concept():
    report = validate()
    assert report["scaling_tasks"] == 16 and report["runtime_scenarios"] == 20
    assert report["runtime_counts"] == {"single-skill": 6, "multi-skill": 5,
                                         "dynamic-reroute": 5, "reuse-session": 4}
    text = "\n".join((BENCH / name).read_text() for name in
                     ("runtime.jsonl", "schema.json", "README.md"))
    assert "stage_transition_success" not in text
    assert "activated_skills_by_stage" not in text
    assert '"stage_id"' not in text and '"stages"' not in text


def test_scaling_targets_are_nested_and_all_tasks_are_body_sufficient():
    manifest = json.loads((ROOT / "evals/corpora/benchmark-corpus-v0.1.json").read_text())
    subsets = {row["skill_id"]: set(row["subsets"]) for row in manifest["skills"]}
    tasks = load_tasks(BENCH / "scaling.jsonl") + load_tasks(BENCH / "runtime.jsonl")
    assert len(tasks) == 36 and all(task.raw["body_sufficient"] is True for task in tasks)
    assert all({"S32", "S64", "S128"} <= subsets[t.raw["target_skill"]]
               for t in tasks if t.suite == "scaling")


def test_dynamic_reroute_event_schema_and_union():
    tasks = load_tasks(BENCH / "runtime.jsonl")
    reroutes = [task.raw for task in tasks if task.raw["scenario_type"] == "dynamic-reroute"]
    assert len(reroutes) == 5
    for task in reroutes:
        union = set(task["initial_required_skills"])
        for event in task["reroute_events"]:
            assert set(event) == {"event_id", "trigger_evidence", "new_required_skills"}
            union.update(event["new_required_skills"])
        assert union == set(task["required_skills"])


def test_reroute_scoring_obeys_event_order(tmp_path):
    task = next(t.raw for t in load_tasks(BENCH / "runtime.jsonl") if t.task_id == "RT-T01")
    trace = {"arm": "control-plane", "activated_skills": ["fastify", "monitoring"],
             "events": [{"event_seq": 1, "type": "capability_activation", "skill_ids": ["fastify"]},
                        {"event_seq": 4, "type": "evidence_emitted", "event_id": "E1"},
                        {"event_seq": 6, "type": "capability_activation", "skill_ids": ["monitoring"]}]}
    task = dict(task, automatic_success_criteria=[])
    scored = score_run(task, trace, tmp_path)
    assert scored["reroute_success"] is True
    assert scored["new_required_skill_recall"] == 1 and scored["premature_activation_count"] == 0
    trace["events"][2]["event_seq"] = 2
    scored = score_run(task, trace, tmp_path)
    assert scored["reroute_success"] is False and scored["premature_activation_count"] == 1


def test_gated_evidence_is_hidden_until_initial_fix(tmp_path):
    fixtures = json.loads((BENCH / "fixtures.json").read_text())
    task = next(t.raw for t in load_tasks(BENCH / "runtime.jsonl") if t.task_id == "RT-T01")
    spec = fixtures[task["fixture"]]
    work, host = tmp_path / "work", tmp_path / "host"
    materialize_fixture(spec, work)
    verifier = materialize_host_verifier(spec, host)
    with gated_verifier_server(verifier, tmp_path / "verifier.sock", work) as socket_path:
        verifier_env = dict(os.environ, BENCHMARK_VERIFIER_CHANNEL=str(socket_path))
        initial = subprocess.run(["python3", "tests/check_all.py"], cwd=work, text=True,
                                 capture_output=True, env=verifier_env)
        assert initial.returncode != 0 and "BENCHMARK_EVIDENCE:E1" not in initial.stdout + initial.stderr
        (work / "src/server.js").write_text("// validated name returns 400 or 201\n")
        advanced = subprocess.run(["python3", "tests/check_all.py"], cwd=work, text=True,
                                  capture_output=True, env=verifier_env)
    assert advanced.returncode != 0 and "BENCHMARK_EVIDENCE:E1" in advanced.stdout + advanced.stderr


def test_all_runtime_fixtures_fail_before_agent_changes(tmp_path):
    fixtures = json.loads((BENCH / "fixtures.json").read_text())
    for task in load_tasks(BENCH / "runtime.jsonl"):
        work = tmp_path / task.task_id / "work"
        verifier = materialize_host_verifier(fixtures[task.raw["fixture"]],
                                             tmp_path / task.task_id / "host")
        materialize_fixture(fixtures[task.raw["fixture"]], work)
        results = [_check(check, work, verifier) for check in task.raw["automatic_success_criteria"]]
        assert not all(ok for ok, _ in results), f"{task.task_id} is vacuous"


def test_json_verifier_rejects_empty_and_fixture_falsehoods(tmp_path):
    check = {"type": "json_contract", "path": "requirements.json", "fields": {
        "functional_requirements": {"type": "array", "nonempty": True, "contains": ["guest"]},
        "assumptions": {"type": "array", "nonempty": True, "contains": ["legal"]}}}
    (tmp_path / "requirements.json").write_text('{"functional_requirements":[],"assumptions":[]}')
    assert _check(check, tmp_path)[0] is False
    (tmp_path / "requirements.json").write_text(
        '{"functional_requirements":["admin login"],"assumptions":["none"]}')
    assert _check(check, tmp_path)[0] is False


def test_gold_isolation_covers_all_agent_visible_projections():
    task = next(iter(load_tasks(BENCH / "runtime.jsonl"))).raw
    fixture = json.loads((BENCH / "fixtures.json").read_text())[task["fixture"]]
    assert_gold_isolated(task, fixture, {"system_context_projection": {}, "tool_descriptions": {}})
    with pytest.raises(ValueError, match="gold exposed"):
        assert_gold_isolated(task, fixture, {"tool_descriptions": {"required_skills": ["x"]}})
    runner_source = (ROOT / "src/skill_control_plane/benchmarks/runner.py").read_text()
    assert '"BENCHMARK_VERIFIER":' not in runner_source


def test_production_bridge_has_tool_parity_and_real_resume_path():
    source = (ROOT / "scripts/pi_benchmark_production_bridge.mjs").read_text()
    for tool in ("'read'", "'write'", "'edit'", "'bash'"):
        assert source.count(tool) >= 2
    assert "SessionManager.open(sessionFile" in source
    assert "session_shutdown" in source and "turnMetrics.push" in source
    assert "faux" not in source.casefold() and "target_skill" not in source


def test_reuse_scoring_requires_both_turn_checks_and_restore(tmp_path):
    task = next(t.raw for t in load_tasks(BENCH / "runtime.jsonl") if t.task_id == "RT-R01")
    task = dict(task, automatic_success_criteria=[])
    base = {"arm": "control-plane", "activated_skills": ["api-design"],
            "session_restore_success": True,
            "turn_metrics": [{"turn_id": "T1", "success_checks": [{"success": True}]},
                             {"turn_id": "T2", "success_checks": [{"success": True}]}]}
    assert score_run(task, base, tmp_path)["task_success"] is True
    base["turn_metrics"][0]["success_checks"][0]["success"] = False
    assert score_run(task, base, tmp_path)["task_success"] is False
    base["turn_metrics"][0]["success_checks"][0]["success"] = True
    base["session_restore_success"] = False
    assert score_run(task, base, tmp_path)["task_success"] is False


def test_embedding_counts_are_separate_and_identity_is_validated():
    validate_result(_result())
    with pytest.raises(ValueError, match="embedding total"):
        validate_result(_result(query_embedding_calls_runtime=1, query_embedding_calls_total=0))
    with pytest.raises(ValueError, match="pi_version"):
        validate_result(_result(pi_version=""))
    identity = {"model": "m", "provider": "p", "corpus_subset": "S128",
                "corpus_version": "benchmark-corpus-v0.1", "timeout_seconds": 300}
    assert_identity_matches_run(identity, model="m", provider="p", corpus="S128",
                                timeout_seconds=300)
    with pytest.raises(ValueError, match="identity drift"):
        assert_identity_matches_run(identity, model="other", provider="p", corpus="S128",
                                    timeout_seconds=300)


def test_result_schema_rejects_negative_tokens_and_false_total():
    with pytest.raises(ValueError, match="llm_input_tokens"):
        validate_result(_result(llm_input_tokens=-1, llm_total_tokens=1))
    with pytest.raises(ValueError, match="must equal"):
        validate_result(_result(llm_total_tokens=99))


def test_offline_unavailable_is_explicit_not_fabricated():
    rows = json.loads((BENCH / "offline-cost.json").read_text())
    for row in rows:
        validate_offline_cost(row)
    with pytest.raises(ValueError, match="must not invent"):
        validate_offline_cost(dict(rows[0], total_tokens=100))


def test_paired_metrics_exclude_control_plane_only_diagnostics():
    native = _result()
    control = _result(arm="control-plane", run_id="r2", task_success=False,
                      bundle_create_count=1, bundle_extend_count=0, bundle_reuse_count=0,
                      required_skill_recall=.5, wrong_skill_count=2, llm_input_tokens=15,
                      llm_output_tokens=3, llm_total_tokens=18, wall_time_ms=30,
                      control_plane_telemetry={"retrieval_calls": 1, "retrieval_events": []})
    delta = paired_report([native, control], "scaling")["paired_delta_control_plane_minus_native"][0]
    assert delta["task_success"] == -1 and delta["llm_total_tokens"] == 6
    assert not ({"bundle_create_count", "bundle_extend_count", "bundle_reuse_count",
                 "retrieval_calls", "query_embedding_calls_total"} & set(delta))


def test_mechanism_helpers():
    assert required_skill_recall(["a", "b"], ["b", "x"]) == .5
    assert repeated_discovery_count(["api", "db", "api", "api", "db"]) == 3
