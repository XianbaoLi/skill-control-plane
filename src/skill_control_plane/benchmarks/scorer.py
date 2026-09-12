"""Deterministic mechanism and task outcome scoring."""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def required_skill_recall(required: list[str], activated: list[str]) -> float:
    wanted = set(required)
    return len(wanted & set(activated)) / len(wanted) if wanted else 1.0


def repeated_discovery_count(discoveries: list[str]) -> int:
    counts = Counter(discoveries)
    return sum(count - 1 for count in counts.values() if count > 1)


def _check(check: dict[str, Any], workspace: Path) -> tuple[bool, str]:
    kind = check["type"]
    path = workspace / check.get("path", "")
    if kind == "file_exists":
        ok = path.is_file()
        return ok, f"{path}: {'exists' if ok else 'missing'}"
    if kind == "file_contains":
        ok = path.is_file() and check["text"] in path.read_text(encoding="utf-8")
        return ok, f"{path}: expected text {'found' if ok else 'missing'}"
    if kind == "json_fields":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            ok = all(field in value for field in check["fields"])
        except (OSError, json.JSONDecodeError, TypeError):
            ok = False
        return ok, f"{path}: required JSON fields {'present' if ok else 'missing'}"
    if kind == "command":
        completed = subprocess.run(check["argv"], cwd=workspace, capture_output=True,
                                   text=True, timeout=check.get("timeout_seconds", 30))
        return completed.returncode == 0, (completed.stdout + completed.stderr)[-1000:]
    raise ValueError(f"unknown success check: {kind}")


def score_run(task: dict[str, Any], trace: dict[str, Any], workspace: str | Path) -> dict[str, Any]:
    details = [_check(check, Path(workspace)) for check in task["automatic_success_criteria"]]
    activated = list(dict.fromkeys(trace.get("activated_skills", [])))
    discoveries = trace.get("discoveries", [])
    required = task["required_skills"]
    stage_ok = None
    if task.get("scenario_type") == "stage-transition":
        by_stage = trace.get("activated_skills_by_stage", {})
        stage_ok = all(set(stage["required_skills"]) <= set(by_stage.get(stage["stage_id"], []))
                       for stage in task["stages"])
    return {
        "task_success": all(ok for ok, _ in details),
        "success_details": [{"success": ok, "detail": detail} for ok, detail in details],
        "required_skills": required,
        "activated_skills": activated,
        "required_skill_recall": required_skill_recall(required, activated),
        "wrong_skill_count": len(set(activated) - set(required)),
        "discovery_count": len(discoveries),
        "repeated_discovery_count": repeated_discovery_count(discoveries),
        "skill_body_load_count": len(trace.get("skill_body_loads", [])),
        "stage_transition_success": stage_ok,
        "bundle_create_count": sum(a == "CREATE" for a in trace.get("bundle_actions", [])),
        "bundle_extend_count": sum(a == "EXTEND" for a in trace.get("bundle_actions", [])),
        "bundle_reuse_count": int(trace.get("bundle_reuse_count", 0)),
        "session_restore_success": trace.get("session_restore_success"),
    }
