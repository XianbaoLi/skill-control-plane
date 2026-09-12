#!/usr/bin/env python3
"""Validate benchmark v0.1 task manifests without network access."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from skill_control_plane.benchmarks.models import load_tasks, validate_offline_cost

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evals/benchmarks/v0.1"
CORPUS = ROOT / "evals/corpora/benchmark-corpus-v0.1.json"


def _leaks(prompt: str, term: str) -> bool:
    words = [re.escape(piece) for piece in re.split(r"[-_\s]+", term.casefold()) if piece]
    return bool(words and re.search(r"(?<![a-z0-9])" + r"[-_\s]+".join(words) +
                                    r"(?![a-z0-9])", prompt.casefold()))


def validate() -> dict:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    memberships = {row["skill_id"]: set(row["subsets"]) for row in corpus["skills"]}
    scaling = load_tasks(BENCH / "scaling.jsonl")
    runtime = load_tasks(BENCH / "runtime.jsonl")
    fixtures = json.loads((BENCH / "fixtures.json").read_text(encoding="utf-8"))
    assert len(scaling) == 16, "Scaling must contain exactly 16 tasks"
    assert len(runtime) == 20, "Runtime must contain exactly 20 scenarios"
    counts = Counter(task.raw.get("scenario_type") for task in runtime)
    assert counts == {"single-skill": 6, "multi-skill": 5,
                      "dynamic-reroute": 5, "reuse-session": 4}
    leakage = []
    for task in scaling + runtime:
        assert task.raw["fixture"] in fixtures, f"unknown fixture: {task.raw['fixture']}"
        assert task.raw.get("body_sufficient") is True
        for skill in task.required_skills:
            assert "S128" in memberships.get(skill, set()), f"{task.task_id}: invalid required Skill {skill}"
        if task.suite == "scaling":
            assert "S32" in memberships[task.raw["target_skill"]]
            assert {"S32", "S64", "S128"} <= memberships[task.raw["target_skill"]]
        terms = set(task.required_skills)
        if task.raw.get("target_skill"):
            terms.add(task.raw["target_skill"])
        found = sorted(term for term in terms if _leaks(task.prompt, term))
        assert not found, f"{task.task_id}: prompt leaks {found}"
        leakage.append({"task_id": task.task_id, "status": "pass", "checked_terms": sorted(terms)})
        if task.raw.get("scenario_type") == "multi-skill":
            assert 2 <= len(task.required_skills) <= 3
        assert "stages" not in task.raw and "stage_id" not in task.raw
        if task.raw.get("scenario_type") == "dynamic-reroute":
            assert task.raw["initial_required_skills"]
            assert task.raw["reroute_events"]
            assert all(set(event) == {"event_id", "trigger_evidence", "new_required_skills"}
                       for event in task.raw["reroute_events"])
            visible = json.dumps(fixtures[task.raw["fixture"]].get("files", {})).casefold()
            assert "benchmark_evidence" not in visible
            for event in task.raw["reroute_events"]:
                assert event["trigger_evidence"].casefold() not in visible
        if task.raw.get("scenario_type") == "reuse-session":
            assert len(task.raw.get("turns", [])) >= 2
            assert all(turn.get("success_criteria") for turn in task.raw["turns"])
    offline = json.loads((BENCH / "offline-cost.json").read_text(encoding="utf-8"))
    for row in offline:
        validate_offline_cost(row)
    return {"scaling_tasks": len(scaling), "runtime_scenarios": len(runtime),
            "runtime_counts": counts, "leakage_checks": leakage,
            "offline_cost_records": len(offline)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict) if args.json else
          f"valid: scaling={report['scaling_tasks']} runtime={report['runtime_scenarios']} leakage=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
