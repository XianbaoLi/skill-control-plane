from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from skill_control_plane.evals import (
    evaluate_stage_bundle_cases,
    load_stage_transition_gold,
)
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.retrieval import (
    BM25Retriever,
    DEFAULT_DENSE_MODEL,
    DenseRetriever,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the direct-evidence Stage Bundle experiment."
    )
    parser.add_argument("root", help="Skill tree root")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-bundles", type=int, default=4)
    parser.add_argument("--max-skills-per-bundle", type=int, default=4)
    parser.add_argument("--allow-snapshot-mismatch", action="store_true")
    parser.add_argument(
        "--corpus-mode",
        default="exact",
        choices=("exact", "public_subset_exploratory"),
    )
    parser.add_argument("--output")
    return parser


def _snapshot_info(
    gold_path: str,
    manifest_path: str,
    *,
    allow_mismatch: bool,
) -> tuple[str, str, bool]:
    cases = load_stage_transition_gold(gold_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = str(manifest.get("snapshot_id", ""))
    gold = cases[0].snapshot_id
    matched = gold == actual
    if not matched and not allow_mismatch:
        raise ValueError(
            "Gold/manifest snapshot mismatch: "
            f"gold={gold}, manifest={actual}"
        )
    return gold, actual, matched


def _print_summary(report: dict[str, object]) -> None:
    print("=== STAGE CAPABILITY BUNDLE EXPERIMENT ===")
    print(f"corpus_mode: {report['corpus_mode']}")
    print(f"gold_snapshot_id: {report['gold_snapshot_id']}")
    print(f"actual_snapshot_id: {report['actual_snapshot_id']}")
    print(f"snapshot_match: {report['snapshot_match']}")
    print(f"skill_count: {report['skill_count']}")
    print(f"cases: {report['case_count']}")
    print(f"stages: {report['stage_count']}")
    print(f"k_per_retriever: {report['k_per_retriever']}")
    print(f"max_bundles: {report['max_bundles']}")
    print(
        "initial_shelf_future_skill_recall: "
        f"{float(report['initial_shelf_future_skill_recall']):.4f} "
        f"({int(report['initial_shelf_future_skill_hits'])}/"
        f"{int(report['future_transition_skill_count'])})"
    )
    print(
        "initial_shelf_future_bundle_recall: "
        f"{float(report['initial_shelf_future_bundle_recall']):.4f} "
        f"({int(report['initial_shelf_future_bundle_hits'])}/"
        f"{int(report['future_transition_bundle_count'])})"
    )
    print(f"shelf_reuse_rate: {float(report['shelf_reuse_rate']):.4f}")
    print(f"new_bundle_rate: {float(report['new_bundle_rate']):.4f}")
    print(
        "mean_active_required_recall: "
        f"{float(report['mean_active_required_recall']):.4f}"
    )
    print(
        "mean_shelf_required_recall: "
        f"{float(report['mean_shelf_required_recall']):.4f}"
    )

    cases = report["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        print()
        print(f"[{case['case_id']}]")
        print(
            "  initial future skill recall: "
            f"{float(case['initial_shelf_future_skill_recall']):.4f} "
            f"hits={case['initial_shelf_future_skill_hits']}"
        )
        print(
            "  initial future bundle recall: "
            f"{float(case['initial_shelf_future_bundle_recall']):.4f} "
            f"hits={case['initial_shelf_future_bundle_hits']}"
        )
        print(
            "  shelf reuse / new bundle: "
            f"{float(case['shelf_reuse_rate']):.4f} / "
            f"{float(case['new_bundle_rate']):.4f}"
        )
        stages = case["stages"]
        assert isinstance(stages, list)
        for stage in stages:
            assert isinstance(stage, dict)
            print(
                f"  {stage['stage_id']}: "
                f"active={stage['active_bundle_ids']} "
                f"shelf={stage['bundle_ids']} "
                f"registered={stage['registered_skill_count']} "
                f"required_recall={float(stage['shelf_required_recall']):.3f}"
            )


def main() -> int:
    args = _parser().parse_args()
    gold_snapshot, actual_snapshot, matched = _snapshot_info(
        args.gold,
        args.manifest,
        allow_mismatch=args.allow_snapshot_mismatch,
    )
    if args.corpus_mode == "exact" and not matched:
        raise ValueError(
            "corpus_mode=exact requires a matching Gold snapshot"
        )
    if args.corpus_mode != "exact" and matched:
        raise ValueError(
            "exploratory corpus mode should only be used for a snapshot mismatch"
        )

    cases = load_stage_transition_gold(args.gold)
    skills = load_skill_tree(args.root)
    records = {skill.skill_id: skill for skill in skills}
    bm25 = BM25Retriever([replace(skill, body="") for skill in skills])
    dense = DenseRetriever(skills, model_name=args.dense_model)

    report = evaluate_stage_bundle_cases(
        cases,
        records,
        bm25=bm25,
        dense=dense,
        k=args.k,
        max_bundles=args.max_bundles,
        max_skills_per_bundle=args.max_skills_per_bundle,
    )
    report["corpus_mode"] = args.corpus_mode
    report["gold_snapshot_id"] = gold_snapshot
    report["actual_snapshot_id"] = actual_snapshot
    report["snapshot_match"] = matched
    report["skill_count"] = len(skills)
    report["dense_model"] = args.dense_model

    _print_summary(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
