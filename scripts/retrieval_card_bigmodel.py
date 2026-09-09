#!/usr/bin/env python3
"""Direct ZAI/BigModel adapter for RetrievalCard offline extraction.

stdin:  one RetrievalCard extraction prompt
stdout: one canonical JSON object

No Hermes harness, tools, repository context, Gold labels, or retrieval results are
passed to the model. Provider configuration is explicit and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from skill_control_plane.corpus.bigmodel_chat import (
    DEFAULT_BIGMODEL_BASE_URL,
    DEFAULT_BIGMODEL_CHAT_MODEL,
    BigModelChatClient,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("BIGMODEL_CHAT_MODEL", DEFAULT_BIGMODEL_CHAT_MODEL),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BIGMODEL_BASE_URL", DEFAULT_BIGMODEL_BASE_URL),
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default="none",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()

    prompt = sys.stdin.read()
    if not prompt.strip():
        raise ValueError("RetrievalCard prompt on stdin must not be empty")

    prompt_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    client = BigModelChatClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
    )

    try:
        completion = client(prompt)
    except Exception as exc:
        if args.audit_dir:
            args.audit_dir.mkdir(parents=True, exist_ok=True)
            (args.audit_dir / f"{prompt_key}.error.json").write_text(
                json.dumps(
                    {
                        "prompt_sha256": prompt_key,
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "reasoning_effort": args.reasoning_effort,
                        "error_type": type(exc).__name__,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise

    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
        (args.audit_dir / f"{prompt_key}.json").write_text(
            json.dumps(
                {
                    "prompt_sha256": prompt_key,
                    "prompt": prompt,
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "reasoning_effort": args.reasoning_effort,
                    "completion": json.loads(completion),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    sys.stdout.write(completion)


if __name__ == "__main__":
    main()
