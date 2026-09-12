"""Guarded benchmark runner primitives; no implicit full-matrix execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from .models import VERSION, validate_result
from .scorer import score_run

GOLD_KEYS = {"target_skill", "required_skills", "distractor_skills",
             "initial_required_skills", "reroute_events", "mechanism_expectation"}


def agent_projection(task: dict[str, Any]) -> dict[str, Any]:
    """The only task data crossing the host/agent boundary."""
    return {"prompt": task["prompt"]}


def assert_gold_isolated(task: dict[str, Any], fixture_spec: dict[str, Any],
                         trace: dict[str, Any] | None = None) -> None:
    visible = {"user": agent_projection(task),
               "workspace": {"files": fixture_spec.get("files", {})}}
    if trace is not None:
        visible["system_context"] = trace.get("system_context_projection", {})
        visible["tool_descriptions"] = trace.get("tool_descriptions", {})
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = GOLD_KEYS & set(value)
            if leaked:
                raise ValueError(f"benchmark gold exposed to agent: {sorted(leaked)}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(visible)


def assert_identity_matches_run(identity: dict[str, Any], *, model: str, corpus: str,
                                provider: str | None, timeout_seconds: int) -> None:
    expected = {"model": model, "corpus_subset": corpus,
                "timeout_seconds": timeout_seconds, "corpus_version": "benchmark-corpus-v0.1"}
    if provider is not None:
        expected["provider"] = provider
    mismatches = [key for key, value in expected.items() if identity.get(key) != value]
    if mismatches:
        raise ValueError(f"experiment identity drift: {', '.join(mismatches)}")


def materialize_fixture(spec: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative, content in spec.get("files", {}).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def materialize_host_verifier(spec: dict[str, Any], destination: Path) -> Path | None:
    files = spec.get("host_files", {})
    if not files:
        return None
    destination.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return destination / spec.get("host_entrypoint", "verify.py")


@contextmanager
def gated_verifier_server(verifier: Path | None, socket_path: Path, workspace: Path):
    """Expose gated results over FIFOs without exposing verifier source or path."""
    if verifier is None:
        yield None
        return

    request_path = Path(str(socket_path) + ".request")
    response_path = Path(str(socket_path) + ".response")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(request_path)
    os.mkfifo(response_path)

    def serve() -> None:
        while True:
            with request_path.open(encoding="utf-8") as request:
                if request.readline().rstrip("\n") == "STOP":
                    return
            completed = subprocess.run(
                [sys.executable, str(verifier)], cwd=workspace,
                capture_output=True, text=True, timeout=30,
            )
            payload = {"returncode": completed.returncode,
                       "stdout": completed.stdout, "stderr": completed.stderr}
            with response_path.open("w", encoding="utf-8") as response:
                response.write(json.dumps(payload) + "\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield socket_path
    finally:
        with request_path.open("w", encoding="utf-8") as request:
            request.write("STOP\n")
        thread.join(timeout=2)
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def run_one(*, task: dict[str, Any], arm: str, corpus: str, model: str,
            fixture_spec: dict[str, Any], output: Path, command: list[str],
            timeout_seconds: int = 300, provider: str | None = None) -> dict[str, Any]:
    """Run one explicitly selected cell via a Pi JSON bridge command."""
    assert_gold_isolated(task, fixture_spec)
    run_id = str(uuid.uuid4())
    work = output / "work" / run_id
    materialize_fixture(fixture_spec, work)
    verifier = materialize_host_verifier(fixture_spec, output / "verifiers" / run_id)
    trace_path = output / "traces" / f"{run_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"BENCHMARK_ARM": arm, "BENCHMARK_CORPUS": corpus,
                "BENCHMARK_PROMPT": task["prompt"], "BENCHMARK_WORKSPACE": str(work),
                "BENCHMARK_TRACE": str(trace_path), "BENCHMARK_MODEL": model,
                "BENCHMARK_PROVIDER": provider or "benchmark-faux",
                "BENCHMARK_TIMEOUT_SECONDS": str(timeout_seconds),
                "BENCHMARK_TURNS": json.dumps(task.get("turns", [])),
                "BENCHMARK_TURN_CHECKS": json.dumps([
                    turn.get("success_criteria", []) for turn in task.get("turns", [])]),
                "BENCHMARK_VERIFIER_CHANNEL": ""})
    started = perf_counter()
    with gated_verifier_server(verifier, output / "sockets" / f"{run_id}.sock", work) as socket_path:
        env["BENCHMARK_VERIFIER_CHANNEL"] = str(socket_path) if socket_path else ""
        completed = subprocess.run(command, env=env, capture_output=True, text=True,
                                   timeout=timeout_seconds)
    wall_time_ms = int((perf_counter() - started) * 1000)
    if not trace_path.exists():
        raise RuntimeError(f"Pi bridge did not write trace (exit {completed.returncode}): {completed.stderr[-1000:]}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert_gold_isolated(task, fixture_spec, trace)
    trace["arm"] = arm
    scored = score_run(task, trace, work, verifier)
    usage = trace.get("usage", {})
    row = {"benchmark_version": VERSION, "task_id": task["task_id"], "suite": task["suite"],
           "arm": arm, "corpus": corpus, "model": model, "run_id": run_id, **scored,
           "llm_input_tokens": int(usage.get("input", 0)),
           "llm_output_tokens": int(usage.get("output", 0)),
           "llm_total_tokens": int(usage.get("input", 0)) + int(usage.get("output", 0)),
           "cached_tokens": usage.get("cacheRead"),
           "query_embedding_calls_startup": int(trace.get("query_embedding_calls_startup", 0)),
           "query_embedding_calls_runtime": int(trace.get("query_embedding_calls_runtime", 0)),
           "query_embedding_calls_total": int(trace.get("query_embedding_calls_startup", 0)) +
                                            int(trace.get("query_embedding_calls_runtime", 0)),
           "wall_time_ms": wall_time_ms, "total_tool_calls": int(trace.get("total_tool_calls", 0)),
           "control_plane_telemetry": trace.get("control_plane_telemetry", {
               "retrieval_calls": None,
               "retrieval_events": [] if arm == "control-plane" else None})}
    identity = trace.get("experiment_identity", {})
    assert_identity_matches_run(identity, model=model, corpus=corpus, provider=provider,
                                timeout_seconds=timeout_seconds)
    row.update({"pi_package": identity.get("pi_package", "@earendil-works/pi-coding-agent"),
                "pi_version": identity.get("pi_version", trace.get("pi_version", "unknown")),
                "provider": identity.get("provider", trace.get("provider", "unknown")),
                "model": identity.get("model", trace.get("model", model)),
                "reasoning_config": identity.get("reasoning_config"),
                "temperature": identity.get("temperature"),
                "max_turns": int(identity.get("max_turns", 20)),
                "timeout_seconds": int(identity.get("timeout_seconds", timeout_seconds)),
                "corpus_version": identity.get("corpus_version", "benchmark-corpus-v0.1"),
                "corpus_subset": identity.get("corpus_subset", corpus),
                "retrieval_card_identity": identity.get("retrieval_card_identity"),
                "dense_index_identity": identity.get("dense_index_identity"),
                "adapter_commit_sha": identity.get("adapter_commit_sha", "unknown")})
    validate_result(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def resolve_bridge(command: str | None, *, production: bool = False) -> list[str]:
    if command:
        return command.split()
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for the Pi bridge")
    return [node, ("scripts/pi_benchmark_production_bridge.mjs" if production
                   else "scripts/pi_benchmark_bridge.mjs")]
