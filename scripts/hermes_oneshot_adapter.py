#!/usr/bin/env python3
"""Thin Hermes --oneshot adapter for the V0.4 capability-facet experiment.

Reads the harness prompt from stdin and writes ONLY Hermes' final response to
stdout.  The adapter deliberately does not import Hermes internals: the
experiment exercises the real Hermes main-agent entrypoint exposed by
hermes.exe -z/--oneshot.

By default, rules/memory/preloaded-skill injection is disabled via
--ignore-rules so the query generator sees only the experiment prompt plus the
normal Hermes system/runtime surface.  --safe-mode is opt-in because it also
ignores user config and may change provider setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _default_hermes_exe() -> str:
    override = os.environ.get("HERMES_EXE")
    if override:
        return override
    if os.name == "nt":
        return r"D:\Hermes\bin\hermes.exe"
    wsl_path = Path("/mnt/d/Hermes/bin/hermes.exe")
    if wsl_path.exists():
        return str(wsl_path)
    return "hermes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes", default=_default_hermes_exe())
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--reasoning")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    prompt = sys.stdin.read()
    if not prompt.strip():
        raise ValueError("Harness prompt on stdin must not be empty")
    prompt_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    command = [args.hermes, "--ignore-rules"]
    if args.safe_mode:
        command.append("--safe-mode")
    if args.model:
        command.extend(["--model", args.model])
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.reasoning:
        command.extend(["--reasoning", args.reasoning])

    with tempfile.TemporaryDirectory(prefix="hermes-facet-") as directory:
        usage_path = Path(directory) / "usage.json"
        command.extend(["--usage-file", str(usage_path), "-z", prompt])

        try:
            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if args.audit_dir:
                args.audit_dir.mkdir(parents=True, exist_ok=True)
                (args.audit_dir / f"{prompt_key}.timeout.json").write_text(
                    json.dumps(
                        {
                            "command": command[:-1] + ["<PROMPT>"],
                            "error": "timeout",
                            "timeout": args.timeout,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            raise

        usage = None
        if usage_path.exists():
            try:
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
            except Exception:
                usage = {"parse_error": True}

        if args.audit_dir:
            args.audit_dir.mkdir(parents=True, exist_ok=True)
            audit = {
                "command": command[:-1] + ["<PROMPT>"],
                "prompt": prompt,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "usage": usage,
            }
            (args.audit_dir / f"{prompt_key}.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if process.returncode:
            raise RuntimeError(
                "Hermes one-shot failed; rerun with --audit-dir to inspect stderr"
            )

        sys.stdout.write(process.stdout.strip())


if __name__ == "__main__":
    main()
