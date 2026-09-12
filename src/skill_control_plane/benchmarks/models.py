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
RUNTIME_KINDS = {"single-skill", "multi-skill", "stage-transition", "reuse-session"}
NONNEGATIVE_FIELDS = (
    "wrong_skill_count", "discovery_count", "repeated_discovery_count",
    "skill_body_load_count", "bundle_create_count", "bundle_extend_count",
    "bundle_reuse_count", "llm_input_tokens", "llm_output_tokens",
    "llm_total_tokens", "query_embedding_calls", "wall_time_ms",
    "total_tool_calls",
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
        tasks.append(BenchmarkTask(raw["task_id"], raw["suite"], raw["category"],
                                   tuple(raw["required_skills"]), raw["prompt"], raw))
    _require(len({task.task_id for task in tasks}) == len(tasks), "duplicate task_id")
    return tasks


def validate_result(row: dict[str, Any]) -> BenchmarkResult:
    required = (
        "benchmark_version", "task_id", "suite", "arm", "corpus", "model", "run_id",
        "task_success", "success_details", "required_skills", "activated_skills",
        "required_skill_recall", "wrong_skill_count", "discovery_count",
        "repeated_discovery_count", "skill_body_load_count", "stage_transition_success",
        "bundle_create_count", "bundle_extend_count", "bundle_reuse_count",
        "session_restore_success", "llm_input_tokens", "llm_output_tokens",
        "llm_total_tokens", "cached_tokens", "query_embedding_calls", "wall_time_ms",
        "total_tool_calls", "control_plane_telemetry",
    )
    missing = [key for key in required if key not in row]
    _require(not missing, f"result missing fields: {', '.join(missing)}")
    _require(row["benchmark_version"] == VERSION, "unsupported benchmark_version")
    _require(row["suite"] in SUITES, "invalid suite")
    _require(row["arm"] in ARMS, "invalid arm")
    _require(row["corpus"] in CORPORA, "invalid corpus")
    _require(isinstance(row["task_success"], bool), "task_success must be bool")
    _require(0 <= row["required_skill_recall"] <= 1, "required_skill_recall out of range")
    for key in NONNEGATIVE_FIELDS:
        value = row[key]
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"{key} must be a non-negative integer")
    _require(row["llm_total_tokens"] == row["llm_input_tokens"] + row["llm_output_tokens"],
             "llm_total_tokens must equal input + output")
    _require(row["cached_tokens"] is None or (
        isinstance(row["cached_tokens"], int) and row["cached_tokens"] >= 0),
        "cached_tokens must be null or non-negative")
    cp = row["control_plane_telemetry"]
    _require(isinstance(cp, dict), "control_plane_telemetry must be an object")
    if row["arm"] == "native":
        _require(cp.get("retrieval_calls") in (None, 0), "native retrieval_calls must be null/0")
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
