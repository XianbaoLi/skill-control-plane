from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from skill_control_plane.corpus import write_corpus_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skill-control-plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("corpus", help="Corpus utilities")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)

    snapshot = corpus_subparsers.add_parser(
        "snapshot",
        help="Create a content-free manifest for a local Skill tree",
    )
    snapshot.add_argument("root")
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--source", required=True)
    snapshot.add_argument("--harness-commit")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "corpus" and args.corpus_command == "snapshot":
        manifest = write_corpus_manifest(
            args.root,
            args.output,
            source=args.source,
            harness_commit=args.harness_commit,
        )
        print(
            json.dumps(
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "skill_count": manifest["skill_count"],
                    "duplicate_skill_ids": manifest["duplicate_skill_ids"],
                    "output": args.output,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
