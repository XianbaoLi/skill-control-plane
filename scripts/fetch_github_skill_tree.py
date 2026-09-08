from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


API = "https://api.github.com"


def _get_json(url: str, token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "skill-control-plane-stage-experiment",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_blob(
    repo: str,
    sha: str,
    token: str | None,
) -> bytes:
    payload = _get_json(f"{API}/repos/{repo}/git/blobs/{sha}", token)
    if payload.get("encoding") != "base64":
        raise ValueError(f"unexpected blob encoding for {sha}")
    return base64.b64decode(payload["content"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prefix", default="skills")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    commit = _get_json(
        f"{API}/repos/{args.repo}/commits/{args.commit}",
        token,
    )
    tree_sha = commit["commit"]["tree"]["sha"]
    tree = _get_json(
        f"{API}/repos/{args.repo}/git/trees/{tree_sha}?recursive=1",
        token,
    )
    prefix = args.prefix.rstrip("/") + "/"
    entries = [
        item
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").startswith(prefix)
        and item.get("path", "").endswith("/SKILL.md")
    ]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def write_entry(item: dict) -> str:
        path = str(item["path"])
        relative = path[len(prefix):]
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            _fetch_blob(args.repo, str(item["sha"]), token)
        )
        return relative

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        written = list(pool.map(write_entry, entries))

    print(f"commit: {commit['sha']}")
    print(f"skill_md_count: {len(written)}")
    for path in sorted(written):
        print(path)

    if not written:
        raise SystemExit("no SKILL.md files found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
