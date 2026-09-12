"""Deterministic mechanism and task outcome scoring."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def required_skill_recall(required: list[str], activated: list[str]) -> float:
    wanted = set(required)
    return len(wanted & set(activated)) / len(wanted) if wanted else 1.0


def repeated_discovery_count(discoveries: list[str]) -> int:
    """Return the legacy query-text metric; prefer capability-aware metrics."""

    counts = Counter(discoveries)
    return sum(count - 1 for count in counts.values() if count > 1)


def duplicate_skill_activation_count(events: list[dict[str, Any]]) -> int:
    """Count activation attempts after a Skill's first legal activation."""

    ordered = sorted(events, key=lambda event: event.get("event_seq", 0))
    apply_events = [event for event in ordered
                    if event.get("type") == "capability_apply_result" and
                    (event.get("skill_ids") or event.get("committed_skill_ids"))]
    if not apply_events:
        apply_events = [event for event in ordered
                        if event.get("type") == "capability_activation" and
                        event.get("action") != "native-read"]
    seen: set[str] = set()
    duplicates = 0
    for event in apply_events:
        for skill_id in event.get("skill_ids", ()):
            if skill_id in seen:
                duplicates += 1
            seen.add(skill_id)
    return duplicates


def redundant_discovery_count(events: list[dict[str, Any]]) -> int:
    """Count searches that later apply a Skill already active at search time."""

    ordered = sorted(events, key=lambda event: event.get("event_seq", 0))
    search_states: list[tuple[int, set[str]]] = []
    apply_events: list[tuple[int, set[str]]] = []
    active: set[str] = set()
    for event in ordered:
        event_type = event.get("type")
        if event_type == "capability_search_result":
            search_states.append((event.get("event_seq", 0),
                                  set(event.get("active_skill_ids", active))))
        elif event_type == "capability_apply_result":
            requested = event.get("skill_ids") or event.get("committed_skill_ids") or ()
            apply_events.append((event.get("event_seq", 0), set(requested)))
            active.update(event.get("committed_skill_ids", ()))

    redundant = 0
    for search_seq, active_at_search in search_states:
        if any(search_seq < apply_seq and active_at_search & requested
               for apply_seq, requested in apply_events):
            redundant += 1
    return redundant


def _check(check: dict[str, Any], workspace: Path,
           verifier: Path | None = None) -> tuple[bool, str]:
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
            ok = isinstance(value, dict) and all(field in value and bool(value[field])
                                                 for field in check["fields"])
        except (OSError, json.JSONDecodeError, TypeError):
            ok = False
        return ok, f"{path}: required JSON fields {'present' if ok else 'missing'}"
    if kind == "json_contract":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            ok = isinstance(value, dict)
            for field, rule in check["fields"].items():
                item = value.get(field)
                expected_type = rule.get("type")
                ok = ok and ((expected_type == "array" and isinstance(item, list)) or
                             (expected_type == "string" and isinstance(item, str)) or
                             (expected_type == "object" and isinstance(item, dict)))
                if rule.get("nonempty"):
                    ok = ok and bool(item)
                for expected in rule.get("contains", []):
                    rendered = json.dumps(item, ensure_ascii=False).casefold()
                    ok = ok and expected.casefold() in rendered
            for expected in check.get("document_contains", []):
                ok = ok and expected.casefold() in json.dumps(value, ensure_ascii=False).casefold()
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            ok = False
        return ok, f"{path}: semantic JSON contract {'satisfied' if ok else 'failed'}"
    if kind == "command":
        argv = ([sys.executable, str(verifier)] if verifier is not None
                else list(check["argv"]))
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        try:
            completed = subprocess.run(argv, cwd=workspace, capture_output=True,
                                       text=True, timeout=check.get("timeout_seconds", 30))
        except OSError as exc:
            return False, f"unable to execute verifier: {exc}"
        return completed.returncode == 0, (completed.stdout + completed.stderr)[-1000:]
    raise ValueError(f"unknown success check: {kind}")


def score_run(task: dict[str, Any], trace: dict[str, Any], workspace: str | Path,
              verifier: Path | None = None) -> dict[str, Any]:
    details = [_check(check, Path(workspace), verifier) for check in task["automatic_success_criteria"]]
    if task.get("scenario_type") == "reuse-session":
        turn_metrics = trace.get("turn_metrics", [])
        for index, turn in enumerate(turn_metrics):
            checks = turn.get("success_checks", [])
            expected_count = (len(task["turns"][index].get("success_criteria", []))
                              if index < len(task.get("turns", [])) else 0)
            if len(checks) != expected_count:
                details.append((False, f"{turn.get('turn_id')}: missing success checks"))
            details.extend((bool(check.get("success")),
                            f"{turn.get('turn_id')}: {check.get('detail', '')}")
                           for check in checks)
        if len(turn_metrics) != len(task.get("turns", [])):
            details.append((False, "not all reuse turns executed"))
        details.append((trace.get("session_restore_success") is True,
                        "Pi session restore succeeded"))
    activated = list(dict.fromkeys(trace.get("activated_skills", [])))
    discoveries = trace.get("discoveries", [])
    events = trace.get("events", [])
    required = task["required_skills"]
    reroute_ok = None
    new_recall = None
    premature = 0
    if task.get("scenario_type") == "dynamic-reroute":
        ordered = sorted(trace.get("events", []), key=lambda event: event.get("event_seq", 0))
        activation_seq = {}
        evidence_seq = {}
        for event in ordered:
            if event.get("type") == "capability_activation":
                for skill in event.get("skill_ids", []):
                    activation_seq.setdefault(skill, event["event_seq"])
            if event.get("type") == "evidence_emitted":
                evidence_seq.setdefault(event.get("event_id"), event["event_seq"])
        expected = []
        successes = []
        for event in task["reroute_events"]:
            trigger = evidence_seq.get(event["event_id"])
            skills = event["new_required_skills"]
            expected.extend(skills)
            after = [skill for skill in skills if trigger is not None and
                     activation_seq.get(skill, -1) > trigger]
            premature += sum(activation_seq.get(skill, 10**18) < (trigger or -1)
                             for skill in skills)
            successes.append(trigger is not None and len(after) == len(skills))
        discovered_after = sum(any(
            evidence_seq.get(event["event_id"]) is not None and
            activation_seq.get(skill, -1) > evidence_seq[event["event_id"]]
            for event in task["reroute_events"] if skill in event["new_required_skills"])
            for skill in set(expected))
        new_recall = discovered_after / len(set(expected)) if expected else 1.0
        reroute_ok = all(successes) and premature == 0
    return {
        "task_success": all(ok for ok, _ in details),
        "success_details": [{"success": ok, "detail": detail} for ok, detail in details],
        "required_skills": required,
        "activated_skills": activated,
        "required_skill_recall": required_skill_recall(required, activated),
        "wrong_skill_count": len(set(activated) - set(required)),
        "discovery_count": len(discoveries),
        "repeated_discovery_count": repeated_discovery_count(discoveries),
        "redundant_discovery_count": redundant_discovery_count(events),
        "duplicate_skill_activation_count": duplicate_skill_activation_count(events),
        "skill_body_load_count": len(trace.get("skill_body_loads", [])),
        "reroute_success": reroute_ok,
        "new_required_skill_recall": new_recall,
        "premature_activation_count": premature,
        "bundle_create_count": (sum(a == "CREATE" for a in trace.get("bundle_actions", []))
                                if trace.get("arm") == "control-plane" else None),
        "bundle_extend_count": (sum(a == "EXTEND" for a in trace.get("bundle_actions", []))
                                if trace.get("arm") == "control-plane" else None),
        "bundle_reuse_count": (int(trace.get("bundle_reuse_count", 0))
                               if trace.get("arm") == "control-plane" else None),
        "session_restore_success": trace.get("session_restore_success"),
        "turn_metrics": trace.get("turn_metrics", []),
    }
