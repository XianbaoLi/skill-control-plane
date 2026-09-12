"""Strict, dependency-free models for benchmark v0.1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "benchmark-v0.1"
ARMS = {"native", "control-plane"}
CORPORA = {"S32", "S64", "S128"}
SUITES = {"scaling", "runtime"}
RUNTIME_KINDS = {"single-skill", "multi-skill", "dynamic-reroute", "reuse-session"}
NONNEGATIVE_FIELDS = (
    "wrong_skill_count", "discovery_count", "repeated_discovery_count",
    "skill_body_load_count", "llm_input_tokens", "llm_output_tokens",
    "llm_total_tokens", "query_embedding_calls_startup",
    "query_embedding_calls_runtime", "query_embedding_calls_total", "wall_time_ms",
    "total_tool_calls",
)

IDENTITY_FIELDS = (
    "pi_package", "pi_version", "provider", "model", "base_url_host",
    "reasoning_config",
    "temperature", "max_turns", "timeout_seconds", "corpus_version",
    "corpus_subset", "retrieval_card_identity", "dense_index_identity",
    "adapter_commit_sha",
)


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    suite: str
    category: str
    required_skills: tuple[str, ...]
    prompt: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkResult:
    data: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    tasks = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for key in ("task_id", "suite", "category", "required_skills", "prompt",
                    "fixture", "automatic_success_criteria", "mechanism_expectation",
                    "selection_reason", "distractor_skills", "leakage_check"):
            _require(key in raw, f"line {number}: missing {key}")
        _require(raw["suite"] in SUITES, f"line {number}: invalid suite")
        _require(isinstance(raw["required_skills"], list) and raw["required_skills"],
                 f"line {number}: required_skills must be non-empty")
        _require(raw.get("body_sufficient") is True,
                 f"line {number}: body_sufficient must be true for v0.1")
        scenario = raw.get("scenario_type")
        if scenario is not None:
            _require(scenario in RUNTIME_KINDS, f"line {number}: invalid runtime scenario_type")
        if scenario == "dynamic-reroute":
            initial = raw.get("initial_required_skills")
            events = raw.get("reroute_events")
            _require(isinstance(initial, list) and initial,
                     f"line {number}: initial_required_skills must be non-empty")
            _require(isinstance(events, list) and events,
                     f"line {number}: reroute_events must be non-empty")
            union = set(initial)
            for event in events:
                _require(set(event) == {"event_id", "trigger_evidence", "new_required_skills"},
                         f"line {number}: invalid reroute event schema")
                _require(event["event_id"] and event["trigger_evidence"],
                         f"line {number}: reroute event strings must be non-empty")
                _require(isinstance(event["new_required_skills"], list) and
                         event["new_required_skills"],
                         f"line {number}: new_required_skills must be non-empty")
                union.update(event["new_required_skills"])
            _require(union == set(raw["required_skills"]),
                     f"line {number}: required_skills must equal reroute Skill union")
        tasks.append(BenchmarkTask(raw["task_id"], raw["suite"], raw["category"],
                                   tuple(raw["required_skills"]), raw["prompt"], raw))
    _require(len({task.task_id for task in tasks}) == len(tasks), "duplicate task_id")
    return tasks


def validate_result(row: dict[str, Any]) -> BenchmarkResult:
    required = (
        "benchmark_version", "task_id", "suite", "arm", "corpus", "run_id",
        "task_success", "success_details", "required_skills", "activated_skills",
        "required_skill_recall", "wrong_skill_count", "discovery_count",
        "repeated_discovery_count", "skill_body_load_count", "reroute_success",
        "new_required_skill_recall", "premature_activation_count",
        "bundle_create_count", "bundle_extend_count", "bundle_reuse_count",
        "session_restore_success", "llm_input_tokens", "llm_output_tokens",
        "llm_total_tokens", "cached_tokens", "query_embedding_calls_startup",
        "query_embedding_calls_runtime", "query_embedding_calls_total", "wall_time_ms",
        "total_tool_calls", "control_plane_telemetry",
    ) + IDENTITY_FIELDS
    missing = [key for key in required if key not in row]
    _require(not missing, f"result missing fields: {', '.join(missing)}")
    _require(row["benchmark_version"] == VERSION, "unsupported benchmark_version")
    _require(row["suite"] in SUITES, "invalid suite")
    _require(row["arm"] in ARMS, "invalid arm")
    _require(row["corpus"] in CORPORA, "invalid corpus")
    _require(isinstance(row["task_success"], bool), "task_success must be bool")
    _require(0 <= row["required_skill_recall"] <= 1, "required_skill_recall out of range")
    _require(row["new_required_skill_recall"] is None or
             0 <= row["new_required_skill_recall"] <= 1,
             "new_required_skill_recall out of range")
    _require(row["reroute_success"] is None or isinstance(row["reroute_success"], bool),
             "reroute_success must be bool or null")
    _require(isinstance(row["premature_activation_count"], int) and
             row["premature_activation_count"] >= 0,
             "premature_activation_count must be non-negative")
    for key in NONNEGATIVE_FIELDS:
        value = row[key]
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"{key} must be a non-negative integer")
    _require(row["llm_total_tokens"] == row["llm_input_tokens"] + row["llm_output_tokens"],
             "llm_total_tokens must equal input + output")
    _require(row["query_embedding_calls_total"] ==
             row["query_embedding_calls_startup"] + row["query_embedding_calls_runtime"],
             "query embedding total must equal startup + runtime")
    _require(row["cached_tokens"] is None or (
        isinstance(row["cached_tokens"], int) and row["cached_tokens"] >= 0),
        "cached_tokens must be null or non-negative")
    cp = row["control_plane_telemetry"]
    _require(isinstance(cp, dict), "control_plane_telemetry must be an object")
    if row["arm"] == "native":
        _require(cp.get("retrieval_calls") is None, "native retrieval_calls must be null")
        for key in ("bundle_create_count", "bundle_extend_count", "bundle_reuse_count"):
            _require(row[key] is None, f"native {key} must be null")
    else:
        for key in ("bundle_create_count", "bundle_extend_count", "bundle_reuse_count"):
            _require(isinstance(row[key], int) and row[key] >= 0,
                     f"control-plane {key} must be non-negative")
    for key in ("pi_package", "pi_version", "provider", "model", "base_url_host",
                "corpus_version", "corpus_subset", "retrieval_card_identity",
                "dense_index_identity", "adapter_commit_sha"):
        _require(isinstance(row[key], str) and bool(row[key]), f"{key} must be non-empty")
    for key in ("max_turns", "timeout_seconds"):
        _require(isinstance(row[key], int) and row[key] > 0, f"{key} must be positive")
    return BenchmarkResult(dict(row))


def validate_offline_cost(row: dict[str, Any]) -> None:
    _require(row.get("telemetry_status") in {"available", "unavailable"},
             "offline telemetry_status must be available/unavailable")
    kind = row.get("artifact_kind")
    _require(kind in {"retrieval-card", "dense-index"}, "invalid artifact_kind")
    if row["telemetry_status"] == "unavailable":
        for key in ("input_tokens", "output_tokens", "total_tokens", "request_count",
                    "retry_count", "wall_time_ms", "embedding_calls", "embedding_input"):
            if key in row:
                _require(row[key] is None, f"unavailable telemetry must not invent {key}")
