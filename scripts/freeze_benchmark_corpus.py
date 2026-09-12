#!/usr/bin/env python3
"""Freeze the benchmark Skill corpus from an already cloned source repository."""
from __future__ import annotations

import argparse
from pathlib import Path

from skill_control_plane.corpus import freeze_benchmark_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-repository", default="https://github.com/Emmraan/agent-skills")
    parser.add_argument("--manifest", default=Path("evals/corpora/benchmark-corpus-v0.1.json"), type=Path)
    parser.add_argument("--materialized-root", required=True, type=Path)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    manifest = freeze_benchmark_corpus(
        args.source_root,
        source_commit=args.source_commit,
        source_repository=args.source_repository,
        manifest_path=args.manifest,
        materialized_root=args.materialized_root,
        clean_output=args.clean_output,
    )
    report = manifest["package_report"]
    print(f"source_commit: {manifest['source_commit']}")
    print(f"source_skill_md_count: {manifest['source_skill_md_count']}")
    print(f"source_skill_count: {manifest['source_skill_count']}")
    print(f"eligible_skill_count: {manifest['eligible_skill_count']}")
    print(f"excluded_skill_count: {manifest['excluded_skill_count']}")
    print(f"selected_count: {manifest['selected_count']}")
    print(f"package_file_count: {report['package_file_count']}")
    print(f"package_byte_size: {report['package_byte_size']}")
    print(f"validation: {manifest['validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
