"""Guarded benchmark runner primitives; no implicit full-matrix execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

from .models import VERSION, validate_result
from .scorer import score_run


def materialize_fixture(spec: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative, content in spec.get("files", {}).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_one(*, task: dict[str, Any], arm: str, corpus: str, model: str,
            fixture_spec: dict[str, Any], output: Path, command: list[str],
            timeout_seconds: int = 300) -> dict[str, Any]:
    """Run one explicitly selected cell via a Pi JSON bridge command."""
    run_id = str(uuid.uuid4())
    work = output / "work" / run_id
    materialize_fixture(fixture_spec, work)
    trace_path = output / "traces" / f"{run_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"BENCHMARK_ARM": arm, "BENCHMARK_CORPUS": corpus,
                "BENCHMARK_PROMPT": task["prompt"], "BENCHMARK_WORKSPACE": str(work),
                "BENCHMARK_TRACE": str(trace_path), "BENCHMARK_MODEL": model})
    started = perf_counter()
    completed = subprocess.run(command, env=env, capture_output=True, text=True,
                               timeout=timeout_seconds)
    wall_time_ms = int((perf_counter() - started) * 1000)
    if not trace_path.exists():
        raise RuntimeError(f"Pi bridge did not write trace (exit {completed.returncode}): {completed.stderr[-1000:]}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    scored = score_run(task, trace, work)
    usage = trace.get("usage", {})
    row = {"benchmark_version": VERSION, "task_id": task["task_id"], "suite": task["suite"],
           "arm": arm, "corpus": corpus, "model": model, "run_id": run_id, **scored,
           "llm_input_tokens": int(usage.get("input", 0)),
           "llm_output_tokens": int(usage.get("output", 0)),
           "llm_total_tokens": int(usage.get("input", 0)) + int(usage.get("output", 0)),
           "cached_tokens": usage.get("cacheRead"),
           "query_embedding_calls": int(trace.get("query_embedding_calls", 0)),
           "wall_time_ms": wall_time_ms, "total_tool_calls": int(trace.get("total_tool_calls", 0)),
           "control_plane_telemetry": trace.get("control_plane_telemetry", {
               "retrieval_calls": 0 if arm == "native" else None,
               "retrieval_events": [] if arm == "control-plane" else None})}
    validate_result(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def resolve_bridge(command: str | None) -> list[str]:
    if command:
        return command.split()
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for the Pi bridge")
    return [node, "scripts/pi_benchmark_bridge.mjs"]
