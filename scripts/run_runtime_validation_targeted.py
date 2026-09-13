#!/usr/bin/env python3
"""Run the frozen Phase 5/6 runtime-validation cells from a local WSL terminal.

This is orchestration only: it invokes the existing one-cell production runner
unchanged, then summarizes its existing result and trace contracts.  It must
not be run in a short-lived sandbox; use tmux on the local host instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORPUS = "S128"
ARM = "control-plane"
TIMEOUT_SECONDS = 600
MAX_TURNS = 20
PHASE5 = ("RT-R01",)
PHASE6 = ("RT-T01", "RT-T02", "RT-T04", "RT-T05")


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_dotenv(path: Path) -> dict[str, str]:
    """Load only BigModel variables; never print their values."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.strip().partition("=")
        if not separator or key.startswith("#") or not key.startswith("BIGMODEL_"):
            continue
        values[key] = value.strip().strip("\"'")
    required = ("BIGMODEL_CHAT_BASE_URL", "BIGMODEL_CHAT_API_KEY", "BIGMODEL_CHAT_MODEL",
                "BIGMODEL_EMBEDDING_BASE_URL", "BIGMODEL_EMBEDDING_API_KEY",
                "BIGMODEL_EMBEDDING_MODEL")
    missing = [key for key in required if not values.get(key) and not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"missing required BigModel configuration: {', '.join(missing)}")
    return values


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_tasks() -> dict[str, dict[str, Any]]:
    rows = (ROOT / "evals/benchmarks/v0.1/runtime.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["task_id"]: row for row in map(json.loads, rows) if row["task_id"] in {*PHASE5, *PHASE6}}


def cell_id(phase: str, task_id: str, restore: str | None = None) -> str:
    return f"{phase}:{task_id}:{restore}" if restore else f"{phase}:{task_id}"


def cells() -> list[dict[str, str | None]]:
    return ([{"phase": "phase5", "task_id": task, "restore": restore}
             for task in PHASE5 for restore in ("on", "off")] +
            [{"phase": "phase6", "task_id": task, "restore": None} for task in PHASE6])


def valid_row(row: dict[str, Any], cell: dict[str, str | None], head: str) -> bool:
    return (row.get("task_id") == cell["task_id"] and row.get("arm") == ARM and
            row.get("corpus") == CORPUS and row.get("adapter_commit_sha") == head and
            row.get("timeout_seconds") == TIMEOUT_SECONDS and row.get("max_turns") == MAX_TURNS)


def result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def trace_for(output: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    run_id = row.get("run_id")
    candidate = output / "traces" / f"{run_id}.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def latest_runtime_policy(output: Path, run_id: str) -> str | None:
    session_files = list((output / "work" / run_id / "sessions").glob("*.jsonl"))
    policies: list[str] = []
    for session_file in session_files:
        for line in session_file.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") != "custom_message" or item.get("customType") != "skill-control-plane/runtime-policy":
                continue
            for content in item.get("content", []):
                if content.get("type") == "text":
                    policies.append(content.get("text", ""))
    return policies[-1] if policies else None


def event_after(events: list[dict[str, Any]], event_type: str, sequence: int) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == event_type and event.get("event_seq", -1) > sequence]


def runtime_e1(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Accept E1 only from the gated verifier, never source-reading output."""
    return next((event for event in events
                 if event.get("type") == "evidence_emitted" and event.get("event_id") == "E1"
                 and event.get("source") == "gated_verifier_execution"), None)


def phase5_summary(row: dict[str, Any], trace: dict[str, Any] | None, restore: str) -> dict[str, Any]:
    trace = trace or {}
    turns = {entry.get("turn_id"): entry for entry in trace.get("turn_metrics", [])}
    t1, t2 = turns.get("T1", {}), turns.get("T2", {})
    events = trace.get("events", [])
    required = set(row.get("required_skills", []))
    t1_active = {skill for event in events if event.get("turn_id") == "T1" and
                 event.get("type") == "capability_activation" for skill in event.get("skill_ids", [])}
    policy = latest_runtime_policy(Path(trace.get("_output", "")), row.get("run_id", "")) if trace else None
    # The bridge records Pi session restoration directly.  Runtime policy is the
    # observable post-resume CapabilityMemory boundary: restore=off must present
    # no Bundle Cards, while the persisted Pi session/history is still resumed.
    return {
        "task_id": row["task_id"], "capability_memory_restore": restore,
        "task_success": row.get("task_success"),
        "session_restore_success": row.get("session_restore_success"),
        "pi_session_history_restored": trace.get("session_restore_success"),
        "t1_required_skill_active": required <= t1_active,
        "t1_active_skills": sorted(t1_active),
        "t2_discovery_count": t2.get("discovery_count"),
        "t2_skill_activation_count": t2.get("skill_activation_count"),
        "t2_body_load_count": t2.get("skill_body_load_count"),
        "rediscovery_avoided": row.get("rediscovery_avoided"),
        "reactivation_avoided": row.get("reactivation_avoided"),
        "behavioral_effective_reuse": row.get("behavioral_effective_reuse"),
        "llm_input_tokens": row.get("llm_input_tokens"),
        "llm_output_tokens": row.get("llm_output_tokens"),
        "llm_total_tokens": row.get("llm_total_tokens"),
        "total_tool_calls": row.get("total_tool_calls"), "wall_time_ms": row.get("wall_time_ms"),
        "capability_memory_runtime_policy": policy,
        "restore_off_no_bundle_runtime_state": ("Bundle Cards: none" in policy) if restore == "off" and policy else None,
    }


def phase6_summary(row: dict[str, Any], trace: dict[str, Any] | None, task: dict[str, Any]) -> dict[str, Any]:
    trace = trace or {}
    events = sorted(trace.get("events", []), key=lambda event: event.get("event_seq", 0))
    e1 = runtime_e1(events)
    e1_seq = e1.get("event_seq", -1) if e1 else -1
    targets = sorted({skill for item in task.get("reroute_events", []) for skill in item.get("new_required_skills", [])})
    checkpoint_requests = event_after(events, "capability_gap_check_requested", e1_seq) if e1 else []
    checkpoint_completions = event_after(events, "capability_gap_check", e1_seq) if e1 else []
    searches = event_after(events, "capability_search_result", e1_seq) if e1 else []
    selected = event_after(events, "capability_apply_result", e1_seq) if e1 else []
    loads = event_after(events, "skill_body_load", e1_seq) if e1 else []
    candidates: dict[str, int | None] = {}
    for target in targets:
        rank = next((list(item.get("skill_ids", [])).index(target) + 1 for item in searches
                     if target in item.get("skill_ids", [])), None)
        candidates[target] = rank
    def has_selected(target: str) -> bool:
        return any(target in (item.get("selected_skill_ids") or item.get("skill_ids") or []) for item in selected)
    def has_committed(target: str) -> bool:
        return any(target in item.get("committed_skill_ids", []) for item in selected)
    def has_loaded(target: str) -> bool:
        return any(item.get("skill_id") == target for item in loads)
    return {
        "task_id": row["task_id"], "task_success": row.get("task_success"),
        "evidence_observed": e1 is not None, "e1_event_seq": e1_seq if e1 else None,
        "checkpoint_request_count": len(checkpoint_requests),
        "checkpoint_completed_count": len(checkpoint_completions),
        "gap_check_requested_after_e1": bool(checkpoint_requests),
        "gap_check_completed_after_e1": bool(checkpoint_completions),
        # Kept for consumers of the first targeted report; it means completed,
        # not merely requested.
        "checkpoint_count": len(checkpoint_completions),
        "gap_check_triggered_after_e1": bool(checkpoint_completions),
        "checkpoints": [{"trigger_reason": item.get("trigger_reason"),
                           "needs_capability": item.get("needs_capability"),
                           "generated_need": item.get("generated_need"),
                           "evidence_fingerprint": item.get("evidence_fingerprint")}
                          for item in checkpoint_completions],
        "needs_capability": [item.get("needs_capability") for item in checkpoint_completions],
        "generated_need": [item.get("generated_need") for item in checkpoint_completions],
        "post_evidence_search": len(event_after(events, "post_evidence_search", e1_seq)) if e1 else 0,
        "targets": {target: {"target_in_candidates": candidates[target] is not None,
                              "target_rank": candidates[target], "target_selected": has_selected(target),
                              "target_committed": has_committed(target), "body_loaded": has_loaded(target)}
                    for target in targets},
        "reroute_success": row.get("reroute_success"),
        "adapter_commit_sha": row.get("adapter_commit_sha"),
    }


def write_reports(output: Path, head: str, tasks: dict[str, dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> None:
    phase5, phase6 = [], []
    for status in statuses.values():
        row = status.get("result")
        if not row:
            continue
        trace = trace_for(output, row)
        if trace is not None:
            trace["_output"] = str(output)
        if status["phase"] == "phase5":
            phase5.append(phase5_summary(row, trace, status["restore"]))
        else:
            phase6.append(phase6_summary(row, trace, tasks[row["task_id"]]))
    phase5.sort(key=lambda item: (item["task_id"], item["capability_memory_restore"]))
    phase6.sort(key=lambda item: item["task_id"])
    json_dump(output / "reuse-ablation-report.json", {"head": head, "cells": phase5})
    json_dump(output / "reroute-validation-report.json", {"head": head, "cells": phase6})
    (output / "reuse-ablation-report.md").write_text("# Phase 5 reuse ablation\n\n```json\n" +
        json.dumps(phase5, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    (output / "reroute-validation-report.md").write_text("# Phase 6 reroute validation\n\n```json\n" +
        json.dumps(phase6, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")


def write_status(output: Path, head: str, statuses: dict[str, dict[str, Any]]) -> None:
    fields = ["cell", "phase", "task_id", "restore", "status", "run_id", "detail"]
    with (output / "cell-status.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for status in statuses.values():
            writer.writerow({field: status.get(field, "") for field in fields})
    completed = sum(item["status"] in {"SUCCESS", "TASK_FAILED", "SKIPPED"} for item in statuses.values())
    total = len(statuses)
    json_dump(output / "progress.json", {"head": head, "total_cells": total, "completed_cells": completed,
        "pending_cells": total - completed, "cells": list(statuses.values()),
        "updated_at": datetime.now(timezone.utc).isoformat()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="existing/new runtime-validation output directory")
    args = parser.parse_args()
    head = git_head()
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        raise SystemExit("refusing to run: .env is required")
    credentials = load_dotenv(dotenv)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or ROOT / "local_artifacts/benchmark-runs" / f"runtime-validation-final-{timestamp}"
    output = output.resolve()
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(parents=True, exist_ok=True)
    (output / "pi-agent").mkdir(parents=True, exist_ok=True)
    (output / "git-head.txt").write_text(head + "\n", encoding="utf-8")
    json_dump(output / "experiment-identity.json", {"head": head, "arm": ARM, "corpus": CORPUS,
        "provider": "benchmark-bigmodel", "chat_model": credentials.get("BIGMODEL_CHAT_MODEL") or os.environ.get("BIGMODEL_CHAT_MODEL"),
        "phases": {"phase5": list(PHASE5), "phase6": list(PHASE6)}, "cell_count": 6,
        "runner": "scripts/benchmark_run.py one --production", "sequential": True,
        "timeout_seconds": TIMEOUT_SECONDS, "max_turns": MAX_TURNS,
        "capability_memory_restore": {"phase5": ["on", "off"], "phase6": ["on"]}})
    tasks = load_tasks()
    prior_statuses: dict[str, dict[str, Any]] = {}
    progress_path = output / "progress.json"
    if progress_path.exists():
        try:
            prior_statuses = {item["cell"]: item for item in
                              json.loads(progress_path.read_text(encoding="utf-8")).get("cells", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            prior_statuses = {}
    statuses: dict[str, dict[str, Any]] = {}
    for cell in cells():
        identifier = cell_id(cell["phase"], cell["task_id"], cell["restore"])
        prior = prior_statuses.get(identifier, {})
        # results.jsonl does not contain the restore toggle.  Only the persisted
        # cell status can associate a result with its on/off invocation.
        previous = prior.get("result") if isinstance(prior.get("result"), dict) else None
        if previous is not None and not valid_row(previous, cell, head):
            previous = None
        prior_error = prior.get("status") == "ERROR"
        statuses[identifier] = {"cell": identifier, **cell,
                                "status": "SKIPPED" if previous else ("ERROR" if prior_error else "PENDING"),
                                "run_id": previous.get("run_id", "") if previous else prior.get("run_id", ""),
                                "detail": "valid result exists" if previous else prior.get("detail", ""),
                                "result": previous}
    write_status(output, head, statuses)
    write_reports(output, head, tasks, statuses)
    for cell in cells():
        identifier = cell_id(cell["phase"], cell["task_id"], cell["restore"])
        status = statuses[identifier]
        if status["status"] in {"SKIPPED", "ERROR"}:
            continue
        status["status"] = "RUNNING"
        write_status(output, head, statuses)
        env = os.environ.copy(); env.update(credentials)
        env["PI_AGENT_DIR"] = str(output / "pi-agent" / identifier.replace(":", "-"))
        env["BENCHMARK_TIMEOUT_SECONDS"] = str(TIMEOUT_SECONDS)
        env["BENCHMARK_MAX_TURNS"] = str(MAX_TURNS)
        command = [str(ROOT / ".venv/bin/python"), "scripts/benchmark_run.py", "one", "--task", str(cell["task_id"]),
                   "--arm", ARM, "--corpus", CORPUS, "--production", "--timeout-seconds", str(TIMEOUT_SECONDS),
                   "--output", str(output)]
        if cell["restore"] is not None:
            command += ["--capability-memory-restore", str(cell["restore"])]
        started = time.monotonic()
        before_rows = result_rows(output / "results.jsonl")
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        log = output / "logs" / f"{identifier.replace(':', '-')}.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        updated = result_rows(output / "results.jsonl")
        new_row = next((row for row in reversed(updated[len(before_rows):]) if valid_row(row, cell, head)), None)
        status["detail"] = f"exit={completed.returncode}; elapsed_seconds={time.monotonic() - started:.1f}"
        if new_row is not None:
            status["result"] = new_row; status["run_id"] = new_row.get("run_id", "")
            status["status"] = "SUCCESS" if new_row.get("task_success") else "TASK_FAILED"
        else:
            status["status"] = "ERROR"
            status["detail"] += "; no valid result written (provider/runner/timeout infrastructure error)"
        write_status(output, head, statuses)
        write_reports(output, head, tasks, statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
