#!/usr/bin/env python3
"""Run one benchmark cell or summarize existing results.

There is deliberately no command that expands the complete matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skill_control_plane.benchmarks.models import load_tasks, validate_result
from skill_control_plane.benchmarks.reporter import paired_report
from skill_control_plane.benchmarks.runner import resolve_bridge, run_one

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evals/benchmarks/v0.1"


def _task(task_id: str) -> dict:
    tasks = load_tasks(BENCH / ("scaling.jsonl" if task_id.startswith("SC-") else "runtime.jsonl"))
    try:
        return next(task.raw for task in tasks if task.task_id == task_id)
    except StopIteration as exc:
        raise SystemExit(f"unknown task: {task_id}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("one", help="run exactly one explicitly selected cell")
    run.add_argument("--task", required=True)
    run.add_argument("--arm", required=True, choices=("native", "control-plane"))
    run.add_argument("--corpus", required=True, choices=("S32", "S64", "S128"))
    run.add_argument("--model", default="benchmark-faux-1")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bridge-command")
    report = sub.add_parser("report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--suite", choices=("scaling", "runtime"), required=True)
    args = parser.parse_args()
    if args.command == "report":
        rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            validate_result(row)
        print(json.dumps(paired_report(rows, args.suite), ensure_ascii=False, indent=2))
        return 0
    task = _task(args.task)
    fixtures = json.loads((BENCH / "fixtures.json").read_text(encoding="utf-8"))
    row = run_one(task=task, arm=args.arm, corpus=args.corpus, model=args.model,
                  fixture_spec=fixtures[task["fixture"]], output=args.output,
                  command=resolve_bridge(args.bridge_command))
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
