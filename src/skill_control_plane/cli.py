from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from skill_control_plane.corpus import write_corpus_manifest
from skill_control_plane.evals import evaluate_union_retrieval, load_runtime_retrieval_gold
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.retrieval import BM25Retriever, DEFAULT_DENSE_MODEL, DenseRetriever


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

    eval_parser = subparsers.add_parser("eval", help="Evaluation utilities")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    retrieval = eval_subparsers.add_parser(
        "retrieval",
        help="Evaluate BM25 Top-K union Dense Top-K against runtime Gold",
    )
    retrieval.add_argument("root", help="Local Skill tree")
    retrieval.add_argument("--gold", required=True, help="Runtime retrieval Gold JSONL")
    retrieval.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    retrieval.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    retrieval.add_argument("--k", type=int, default=5)
    retrieval.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _validate_snapshot(gold_path: str, manifest_path: str) -> None:
    cases = load_runtime_retrieval_gold(gold_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    gold_snapshot = cases[0].snapshot_id
    manifest_snapshot = str(manifest.get("snapshot_id", ""))
    if manifest_snapshot != gold_snapshot:
        raise ValueError(
            "Gold/manifest snapshot mismatch: "
            f"gold={gold_snapshot}, manifest={manifest_snapshot}"
        )


def _print_retrieval_report(report: dict[str, object]) -> None:
    print(
        f"{'CASE':<9} {'BM25':>7} {'DENSE':>7} "
        f"{'UNION':>7} {'SIZE':>6}  MISSING"
    )
    print("-" * 58)

    rows = report["cases"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        required = row["required"]
        bm25_hits = row["bm25_required_hits"]
        dense_hits = row["dense_required_hits"]
        union_hits = row["union_required_hits"]
        missing = row["missing_required"]
        assert isinstance(required, list)
        assert isinstance(bm25_hits, list)
        assert isinstance(dense_hits, list)
        assert isinstance(union_hits, list)
        assert isinstance(missing, list)

        total = len(required)
        print(
            f"{str(row['case_id']):<9} "
            f"{len(bm25_hits):>2}/{total:<2} "
            f"{len(dense_hits):>2}/{total:<2} "
            f"{len(union_hits):>2}/{total:<2} "
            f"{int(row['candidate_set_size']):>6}  "
            f"{', '.join(missing) if missing else '-'}"
        )

    recalled = int(report["recalled_required_skill_count"])
    required_total = int(report["required_skill_count"])
    print()
    print(
        "candidate_recall: "
        f"{float(report['candidate_recall']):.4f} "
        f"({recalled}/{required_total})"
    )
    print(
        "average_candidate_set_size: "
        f"{float(report['average_candidate_set_size']):.2f}"
    )
    print(
        "full_case_coverage: "
        f"{float(report['full_case_coverage']):.4f}"
    )


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

    if args.command == "eval" and args.eval_command == "retrieval":
        _validate_snapshot(args.gold, args.manifest)
        cases = load_runtime_retrieval_gold(args.gold)

        skills = load_skill_tree(args.root)
        metadata_skills = [replace(skill, body="") for skill in skills]

        bm25 = BM25Retriever(metadata_skills)
        dense = DenseRetriever(skills, model_name=args.dense_model)

        report = evaluate_union_retrieval(
            cases,
            bm25=bm25,
            dense=dense,
            k=args.k,
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_retrieval_report(report)
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
