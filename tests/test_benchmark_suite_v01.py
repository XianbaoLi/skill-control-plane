from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from skill_control_plane.benchmarks.models import VERSION, load_tasks, validate_offline_cost, validate_result
from skill_control_plane.benchmarks.reporter import paired_report
from skill_control_plane.benchmarks.runner import (assert_gold_isolated,
    assert_identity_matches_run, gated_verifier_server, materialize_fixture,
    materialize_host_verifier, run_one)
import scripts.benchmark_run as benchmark_run
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
        "repeated_discovery_count": 0, "redundant_discovery_count": 0,
        "duplicate_skill_activation_count": 0, "skill_body_load_count": 1,
        "reroute_success": None, "new_required_skill_recall": None,
        "premature_activation_count": 0, "bundle_create_count": None,
        "bundle_extend_count": None, "bundle_reuse_count": None,
        "session_restore_success": None, "llm_input_tokens": 10, "llm_output_tokens": 2,
        "llm_total_tokens": 12, "cached_tokens": None,
        "query_embedding_calls_startup": 0, "query_embedding_calls_runtime": 0,
        "query_embedding_calls_total": 0, "wall_time_ms": 20, "total_tool_calls": 2,
        "control_plane_telemetry": {"retrieval_calls": None, "retrieval_events": None},
        "pi_package": "@earendil-works/pi-coding-agent", "pi_version": "0.84.1",
        "provider": "fixture", "model": "fixture", "base_url_host": "api.example.test",
        "reasoning_config": None,
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


def test_production_model_registration_is_benchmark_only():
    source = (ROOT / "scripts/pi_benchmark_production_bridge.mjs").read_text()
    assert "process.env.BENCHMARK_PROVIDER || 'benchmark-bigmodel'" in source
    for name in ("BIGMODEL_CHAT_BASE_URL", "BIGMODEL_CHAT_API_KEY", "BIGMODEL_CHAT_MODEL"):
        assert f"required('{name}')" in source
    assert "modelsPath: null" in source
    assert "allowModelNetwork: false" in source
    assert "modelRuntime.registerProvider(providerId" in source
    assert "await modelRuntime.setRuntimeApiKey(providerId, chatApiKey)" in source
    assert "path.join(agentDir, 'auth.json')" not in source
    assert "models-store.json" not in source
    assert "base_url_host: endpointHost" in source
    assert ".split(chatApiKey).join('[REDACTED]')" in source
    embedding = source[source.index("const embeddingIdentity"):
                       source.index("const trace = {", source.index("const embeddingIdentity"))]
    assert "BIGMODEL_EMBEDDING_MODEL" in embedding
    assert "BIGMODEL_EMBEDDING_BASE_URL" in embedding
    assert "BIGMODEL_CHAT_" not in embedding


def test_production_fails_closed_when_chat_model_config_is_missing(tmp_path, monkeypatch):
    pi_root = Path.home() / ".npm-global/lib/node_modules/@earendil-works/pi-coding-agent"
    if not (pi_root / "dist/bundle/index.js").exists():
        pytest.skip("Pi 0.85 global installation is not present")
    base = {"PATH": os.environ["PATH"], "PI_CORE_ROOT": str(pi_root),
            "BENCHMARK_ARM": "native", "BENCHMARK_CORPUS": "S32",
            "BENCHMARK_PROMPT": "ping", "BENCHMARK_WORKSPACE": str(tmp_path / "work"),
            "BENCHMARK_TRACE": str(tmp_path / "trace.json")}
    for name, required_name in (
        ("BIGMODEL_CHAT_BASE_URL", "BIGMODEL_CHAT_API_KEY"),
        ("BIGMODEL_CHAT_API_KEY", "BIGMODEL_CHAT_MODEL"),
        ("BIGMODEL_CHAT_MODEL", "BIGMODEL_CHAT_BASE_URL"),
    ):
        env = dict(base)
        env[required_name] = "config-present"
        completed = subprocess.run(
            ["node", str(ROOT / "scripts/pi_benchmark_production_bridge.mjs")],
            cwd=tmp_path, env=env, text=True, capture_output=True,
        )
        assert completed.returncode != 0
        assert "missing BIGMODEL_CHAT_" in completed.stderr


def test_production_default_uses_chat_model_env(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_run, "ROOT", tmp_path)
    monkeypatch.setenv("BIGMODEL_CHAT_MODEL", "env-chat-model")
    assert benchmark_run._configured_chat_model() == "env-chat-model"
    monkeypatch.delenv("BIGMODEL_CHAT_MODEL")
    (tmp_path / ".env").write_text("OTHER_KEY=ignored\nBIGMODEL_CHAT_MODEL=file-chat-model\n")
    assert benchmark_run._configured_chat_model() == "file-chat-model"


def test_native_and_control_plane_share_production_model_identity(tmp_path):
    task = next(t.raw for t in load_tasks(BENCH / "scaling.jsonl") if t.task_id == "SC-01")
    fixture = json.loads((BENCH / "fixtures.json").read_text())[task["fixture"]]
    bridge = tmp_path / "fake_bridge.py"
    bridge.write_text(
        "import json, os\n"
        "identity = {'pi_package': 'pi', 'pi_version': 'test',\n"
        "            'provider': os.environ['BENCHMARK_PROVIDER'],\n"
        "            'model': os.environ['BENCHMARK_MODEL'],\n"
        "            'base_url_host': 'api.example.test',\n"
        "            'corpus_subset': 'S32', 'timeout_seconds': 300,\n"
        "            'corpus_version': 'benchmark-corpus-v0.1',\n"
        "            'retrieval_card_identity': 'cards',\n"
        "            'dense_index_identity': 'dense',\n"
        "            'adapter_commit_sha': 'abc'}\n"
        "with open(os.environ['BENCHMARK_TRACE'], 'w') as handle:\n"
        "    json.dump({'events': [], 'usage': {'input': 1, 'output': 1},\n"
        "               'experiment_identity': identity}, handle)\n"
    )
    rows = []
    for arm in ("native", "control-plane"):
        rows.append(run_one(task=task, arm=arm, corpus="S32", model="chat-model",
                            fixture_spec=fixture, output=tmp_path / arm,
                            command=[sys.executable, str(bridge)], provider="benchmark-bigmodel"))
    assert [(row["provider"], row["model"], row["base_url_host"]) for row in rows] == \
        [("benchmark-bigmodel", "chat-model", "api.example.test")] * 2


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
                      redundant_discovery_count=1, duplicate_skill_activation_count=1,
                      required_skill_recall=.5, wrong_skill_count=2, llm_input_tokens=15,
                      llm_output_tokens=3, llm_total_tokens=18, wall_time_ms=30,
                      control_plane_telemetry={"retrieval_calls": 1, "retrieval_events": []})
    delta = paired_report([native, control], "scaling")["paired_delta_control_plane_minus_native"][0]
    assert delta["task_success"] == -1 and delta["llm_total_tokens"] == 6
    assert not ({"bundle_create_count", "bundle_extend_count", "bundle_reuse_count",
                 "redundant_discovery_count", "duplicate_skill_activation_count",
                 "retrieval_calls", "query_embedding_calls_total"} & set(delta))


def test_mechanism_helpers():
    assert required_skill_recall(["a", "b"], ["b", "x"]) == .5
    assert repeated_discovery_count(["api", "db", "api", "api", "db"]) == 3


def test_score_run_reports_capability_aware_redundancy(tmp_path):
    task = next(t.raw for t in load_tasks(BENCH / "scaling.jsonl") if t.task_id == "SC-06")
    trace = {
        "arm": "control-plane",
        "activated_skills": ["api-design"],
        "discoveries": ["first phrasing", "second phrasing", "third phrasing"],
        "skill_body_loads": ["api-design"],
        "bundle_actions": ["CREATE"],
        "bundle_action_requests": ["CREATE", "CREATE", "CREATE"],
        "events": [
            {"event_seq": 1, "type": "capability_search_result", "query": "first",
             "skill_ids": ["api-design"], "active_skill_ids": []},
            {"event_seq": 2, "type": "capability_apply_result",
             "skill_ids": ["api-design"], "committed_skill_ids": ["api-design"],
             "deduplicated_skill_ids": []},
            {"event_seq": 3, "type": "capability_search_result", "query": "second",
             "skill_ids": ["api-design"], "active_skill_ids": ["api-design"]},
            {"event_seq": 4, "type": "capability_apply_result",
             "skill_ids": ["api-design"], "committed_skill_ids": [],
             "deduplicated_skill_ids": ["api-design"]},
            {"event_seq": 5, "type": "capability_search_result", "query": "third",
             "skill_ids": ["api-design"], "active_skill_ids": ["api-design"]},
            {"event_seq": 6, "type": "capability_apply_result",
             "skill_ids": ["api-design"], "committed_skill_ids": [],
             "deduplicated_skill_ids": ["api-design"]},
        ],
    }
    scored = score_run(dict(task, automatic_success_criteria=[]), trace, tmp_path)
    assert scored["redundant_discovery_count"] == 2
    assert scored["duplicate_skill_activation_count"] == 2
    assert scored["bundle_create_count"] == 1
    assert scored["skill_body_load_count"] == 1
