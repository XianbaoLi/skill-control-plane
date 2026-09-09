#!/usr/bin/env python3
"""Structured-output BigModel completion adapter for capability extraction.

Reads one prompt from stdin and prints only the assistant JSON text to stdout.
Uses BigModel's JSON response mode so the harness can parse it deterministically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-5.2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("BIGMODEL_CHAT_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BIGMODEL_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    api_key = os.environ.get("BIGMODEL_API_KEY", "")
    if not api_key:
        raise RuntimeError("BIGMODEL_API_KEY is required")

    prompt = sys.stdin.read()
    if not prompt.strip():
        raise ValueError("Prompt on stdin must not be empty")

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"BigModel chat request failed: {type(exc).__name__}"
        ) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("BigModel chat response has no assistant content") from exc
    if not isinstance(content, str):
        raise RuntimeError("BigModel assistant content is not text")

    # Fail here rather than later: stdout is the harness contract.
    parsed = json.loads(content)
    sys.stdout.write(json.dumps(parsed, ensure_ascii=False))


if __name__ == "__main__":
    main()
