from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from skill_control_plane.corpus import build_corpus_manifest, write_corpus_manifest
from skill_control_plane.evals import (
    LLMQueryParaphraser,
    RETRIEVAL_CARD_FIELD_ABLATIONS,
    build_query_variant_cache,
    compare_query_robustness_reports,
    diagnose_stage_retrieval,
    evaluate_capability_facet_retrieval,
    evaluate_control_plane,
    evaluate_frozen_query_retrieval_ablation,
    evaluate_query_variant_retrieval,
    evaluate_stage_bundle_cases,
    evaluate_stage_reroute,
    evaluate_union_retrieval,
    load_multi_skill_gold,
    load_runtime_retrieval_gold,
    load_stage_transition_gold,
    print_capability_facet_report,
    print_field_ablation_summary,
    print_frozen_query_retrieval_ablation,
    print_query_robustness_comparison,
    require_min_target_transitions,
    summarize_field_ablation_reports,
)
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.discovery import (
    BM25Retriever,
    DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS,
    DEFAULT_BIGMODEL_EMBEDDING_MODEL,
    DEFAULT_DENSE_MODEL,
    BigModelDenseRetriever,
    DenseRetriever,
    LLMRetrievalCardExtractor,
    RETRIEVAL_CARD_FIELDS,
    RETRIEVAL_CARD_VERSION,
    apply_retrieval_cards,
    build_dense_index,
    build_retrieval_card_cache,
    load_retrieval_cards,
    write_dense_index,
)
from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient


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

    retrieval_cards = corpus_subparsers.add_parser(
        "retrieval-cards",
        help="Build leakage-resistant RetrievalCard v0.1 JSONL with an offline LLM",
    )
    retrieval_cards.add_argument("root", help="Local Skill tree")
    retrieval_cards.add_argument("--output", required=True, help="RetrievalCard JSONL")
    retrieval_cards.add_argument(
        "--extract-command",
        required=True,
        help="Completion command: prompt on stdin, JSON object on stdout",
    )
    retrieval_cards.add_argument(
        "--force",
        action="store_true",
        help="Ignore matching cached cards and regenerate all Skills",
    )

    dense_index = corpus_subparsers.add_parser(
        "dense-index",
        help="Build DenseIndexV1 corpus embeddings offline",
    )
    dense_index.add_argument("root", help="Local Skill tree")
    dense_index.add_argument("--output", required=True, help="DenseIndexV1 JSON")
    dense_index.add_argument(
        "--retrieval-cards", required=True, help="RetrievalCard v0.1 JSONL"
    )

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

    stage_reroute = eval_subparsers.add_parser(
        "stage-reroute",
        help="Evaluate raw runtime evidence -> retrieval against stage Gold",
    )
    stage_reroute.add_argument("root", help="Local Skill tree")
    stage_reroute.add_argument(
        "--gold",
        required=True,
        help="Stage-transition Gold JSONL with raw runtime_evidence",
    )
    stage_reroute.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    stage_reroute.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    stage_reroute.add_argument("--k", type=int, default=3)
    stage_reroute.add_argument("--json", action="store_true", dest="as_json")

    stage_diagnose = eval_subparsers.add_parser(
        "stage-diagnose",
        help="Diagnose failed stage retrieval with A/B/C query ablations",
    )
    stage_diagnose.add_argument("root", help="Local Skill tree")
    stage_diagnose.add_argument(
        "--gold",
        required=True,
        help="Stage-transition Gold JSONL with raw runtime_evidence",
    )
    stage_diagnose.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    stage_diagnose.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    stage_diagnose.add_argument("--k", type=int, default=3)
    stage_diagnose.add_argument(
        "--all-targets",
        action="store_true",
        help="Show all required targets, not only targets missed by B",
    )
    stage_diagnose.add_argument("--json", action="store_true", dest="as_json")

    stage_bundle = eval_subparsers.add_parser(
        "stage-bundle",
        help="Evaluate Active Bundle + Capability Shelf over stage Gold",
    )
    stage_bundle.add_argument("root", help="Local Skill tree")
    stage_bundle.add_argument(
        "--gold",
        required=True,
        help="Stage-transition Gold JSONL with raw runtime_evidence",
    )
    stage_bundle.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    stage_bundle.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    stage_bundle.add_argument("--k", type=int, default=5)
    stage_bundle.add_argument("--max-bundles", type=int, default=4)
    stage_bundle.add_argument("--max-skills-per-bundle", type=int, default=4)
    bundle_experiment = stage_bundle.add_mutually_exclusive_group()
    bundle_experiment.add_argument("--hierarchical-ab", action="store_true",
                              help="Compare global-only and known-Shelf-first retrieval")
    bundle_experiment.add_argument("--capability-need-ab", action="store_true",
                                   help="Compare hierarchical raw evidence vs capability need")
    bundle_experiment.add_argument("--soft-bundle-ab", action="store_true", help="Compare hard routing and soft priors")
    bundle_experiment.add_argument("--soft-bundle-repair-ab", action="store_true", help="Compare hard, soft and conditional repair")
    stage_bundle.add_argument("--capability-need-command",
                              help="Completion command: prompt on stdin, JSON on stdout (no shell)")
    stage_bundle.add_argument("--json", action="store_true", dest="as_json")

    capability_facets = eval_subparsers.add_parser(
        "capability-facets",
        help="Compare old single rewrite with main-agent capability facets + multi-query RRF",
    )
    capability_facets.add_argument("root", help="Local Skill tree")
    capability_facets.add_argument(
        "--gold",
        required=True,
        help="Stage-transition Gold JSONL with raw runtime_evidence",
    )
    capability_facets.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    capability_facets.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    capability_facets.add_argument(
        "--dense-backend",
        choices=("sentence-transformers", "bigmodel"),
        default="sentence-transformers",
        help="Dense embedding backend; BM25 stays unchanged",
    )
    capability_facets.add_argument(
        "--bigmodel-embedding-model",
        default=os.environ.get(
            "BIGMODEL_EMBEDDING_MODEL", DEFAULT_BIGMODEL_EMBEDDING_MODEL
        ),
    )
    capability_facets.add_argument(
        "--bigmodel-embedding-dimensions",
        type=int,
        choices=(256, 512, 1024, 2048),
        default=int(
            os.environ.get(
                "BIGMODEL_EMBEDDING_DIMENSIONS",
                str(DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS),
            )
        ),
    )
    capability_facets.add_argument(
        "--old-rewrite-command",
        required=True,
        help="Old capability_need completion command: prompt on stdin, JSON on stdout",
    )
    capability_facets.add_argument(
        "--capability-facets-command",
        required=True,
        help="Main-agent facet completion command: prompt on stdin, JSON on stdout",
    )
    capability_facets.add_argument("--per-query-k", type=int, default=10)
    capability_facets.add_argument("--rrf-k", type=int, default=60)
    capability_facets.add_argument("--json", action="store_true", dest="as_json")

    retrieval_ablation = eval_subparsers.add_parser(
        "retrieval-ablation",
        help="Compare Dense, BM25, Union, and RRF using identical frozen old queries",
    )
    retrieval_ablation.add_argument("root", help="Local Skill tree")
    retrieval_ablation.add_argument(
        "--gold",
        required=True,
        help="Stage-transition Gold JSONL with raw runtime_evidence",
    )
    retrieval_ablation.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold set references",
    )
    retrieval_ablation.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    retrieval_ablation.add_argument(
        "--dense-backend",
        choices=("sentence-transformers", "bigmodel"),
        default="sentence-transformers",
        help="Dense embedding backend; BM25 stays unchanged",
    )
    retrieval_ablation.add_argument(
        "--bigmodel-embedding-model",
        default=os.environ.get(
            "BIGMODEL_EMBEDDING_MODEL", DEFAULT_BIGMODEL_EMBEDDING_MODEL
        ),
    )
    retrieval_ablation.add_argument(
        "--bigmodel-embedding-dimensions",
        type=int,
        choices=(256, 512, 1024, 2048),
        default=int(
            os.environ.get(
                "BIGMODEL_EMBEDDING_DIMENSIONS",
                str(DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS),
            )
        ),
    )
    retrieval_ablation.add_argument(
        "--old-rewrite-command",
        required=True,
        help="Frozen old capability_need replay command: prompt on stdin, JSON on stdout",
    )
    retrieval_ablation.add_argument(
        "--skill-representation",
        choices=("metadata", "retrieval-card"),
        default="metadata",
        help="Skill index text: legacy name+description+tags or RetrievalCard v0.1",
    )
    retrieval_ablation.add_argument(
        "--retrieval-cards",
        help="RetrievalCard JSONL; required with --skill-representation retrieval-card",
    )
    retrieval_ablation.add_argument("--per-retriever-k", type=int, default=10)
    retrieval_ablation.add_argument("--rrf-k", type=int, default=60)
    retrieval_ablation.add_argument("--json", action="store_true", dest="as_json")

    field_ablation = eval_subparsers.add_parser(
        "retrieval-field-ablation",
        help="Leave-one-field-out ablation for RetrievalCard v0.1",
    )
    field_ablation.add_argument("root", help="Local Skill tree")
    field_ablation.add_argument("--gold", required=True)
    field_ablation.add_argument("--manifest", required=True)
    field_ablation.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    field_ablation.add_argument(
        "--dense-backend",
        choices=("sentence-transformers", "bigmodel"),
        default="sentence-transformers",
    )
    field_ablation.add_argument(
        "--bigmodel-embedding-model",
        default=os.environ.get(
            "BIGMODEL_EMBEDDING_MODEL", DEFAULT_BIGMODEL_EMBEDDING_MODEL
        ),
    )
    field_ablation.add_argument(
        "--bigmodel-embedding-dimensions",
        type=int,
        choices=(256, 512, 1024, 2048),
        default=int(
            os.environ.get(
                "BIGMODEL_EMBEDDING_DIMENSIONS",
                str(DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS),
            )
        ),
    )
    field_ablation.add_argument(
        "--old-rewrite-command",
        required=True,
        help="Frozen capability_need replay command",
    )
    field_ablation.add_argument(
        "--retrieval-cards",
        required=True,
        help="RetrievalCard v0.1 JSONL",
    )
    field_ablation.add_argument("--per-retriever-k", type=int, default=10)
    field_ablation.add_argument("--rrf-k", type=int, default=60)
    field_ablation.add_argument(
        "--min-target-transitions",
        type=int,
        default=20,
        help="Minimum target Stage transitions required for explanatory results",
    )
    field_ablation.add_argument(
        "--allow-small-sample",
        action="store_true",
        help="Allow fewer target transitions for smoke testing only",
    )
    field_ablation.add_argument("--json", action="store_true", dest="as_json")

    robustness = eval_subparsers.add_parser(
        "retrieval-robustness",
        help="Compare metadata and RetrievalCard under frozen paraphrase queries",
    )
    robustness.add_argument("root", help="Local Skill tree")
    robustness.add_argument("--gold", required=True)
    robustness.add_argument("--manifest", required=True)
    robustness.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    robustness.add_argument(
        "--dense-backend",
        choices=("sentence-transformers", "bigmodel"),
        default="sentence-transformers",
    )
    robustness.add_argument(
        "--bigmodel-embedding-model",
        default=os.environ.get(
            "BIGMODEL_EMBEDDING_MODEL", DEFAULT_BIGMODEL_EMBEDDING_MODEL
        ),
    )
    robustness.add_argument(
        "--bigmodel-embedding-dimensions",
        type=int,
        choices=(256, 512, 1024, 2048),
        default=int(
            os.environ.get(
                "BIGMODEL_EMBEDDING_DIMENSIONS",
                str(DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS),
            )
        ),
    )
    robustness.add_argument(
        "--old-rewrite-command",
        required=True,
        help="Frozen capability_need replay command",
    )
    robustness.add_argument(
        "--paraphrase-command",
        required=True,
        help="Completion command used only to freeze query paraphrases",
    )
    robustness.add_argument(
        "--query-variants",
        required=True,
        help="JSONL cache for original query + paraphrases per target transition",
    )
    robustness.add_argument(
        "--retrieval-cards",
        required=True,
        help="RetrievalCard v0.1 JSONL",
    )
    robustness.add_argument("--paraphrases-per-query", type=int, default=4)
    robustness.add_argument("--force-paraphrases", action="store_true")
    robustness.add_argument("--per-retriever-k", type=int, default=10)
    robustness.add_argument("--rrf-k", type=int, default=60)
    robustness.add_argument(
        "--min-target-transitions",
        type=int,
        default=20,
        help="Minimum target Stage transitions required for explanatory results",
    )
    robustness.add_argument(
        "--allow-small-sample",
        action="store_true",
        help="Allow fewer target transitions for smoke testing only",
    )
    robustness.add_argument("--json", action="store_true", dest="as_json")

    control_plane = eval_subparsers.add_parser(
        "control-plane",
        help="Evaluate multi-Skill coverage and stage rerouting",
    )
    control_plane.add_argument("root", help="Local Skill tree")
    control_plane.add_argument("--multi-skill", required=True, help="Multi-Skill Gold JSONL")
    control_plane.add_argument(
        "--stage-transition",
        required=True,
        help="Stage-transition Gold JSONL",
    )
    control_plane.add_argument(
        "--manifest",
        required=True,
        help="Corpus manifest whose snapshot_id the Gold sets reference",
    )
    control_plane.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    control_plane.add_argument("--k", type=int, default=3)
    control_plane.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _redact_child_stderr(stderr: str | None) -> str:
    detail = (stderr or "").strip()
    if not detail:
        return "(child process produced no stderr)"
    for name, value in os.environ.items():
        upper = name.upper()
        if value and any(token in upper for token in ("API_KEY", "TOKEN", "SECRET")):
            detail = detail.replace(value, "<redacted>")
    return detail[-4000:]


def _manifest_snapshot_id(manifest_path: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return str(manifest.get("snapshot_id", ""))


def _validate_snapshot_id(gold_snapshot: str, manifest_path: str) -> None:
    manifest_snapshot = _manifest_snapshot_id(manifest_path)
    if manifest_snapshot != gold_snapshot:
        raise ValueError(
            "Gold/manifest snapshot mismatch: "
            f"gold={gold_snapshot}, manifest={manifest_snapshot}"
        )


def _validate_root_snapshot(root: str, manifest_path: str) -> None:
    manifest_snapshot = _manifest_snapshot_id(manifest_path)
    actual_snapshot = str(
        build_corpus_manifest(root, source="runtime-validation").get("snapshot_id", "")
    )
    if actual_snapshot != manifest_snapshot:
        raise ValueError(
            "Skill root/manifest snapshot mismatch: "
            f"root={actual_snapshot}, manifest={manifest_snapshot}. "
            "Use the frozen Skill corpus that produced this manifest."
        )


def _validate_snapshot(gold_path: str, manifest_path: str) -> None:
    cases = load_runtime_retrieval_gold(gold_path)
    _validate_snapshot_id(cases[0].snapshot_id, manifest_path)


def _build_retrievers(root: str, dense_model: str) -> tuple[BM25Retriever, DenseRetriever]:
    skills = load_skill_tree(root)
    metadata_skills = [replace(skill, body="") for skill in skills]
    return (
        BM25Retriever(metadata_skills),
        DenseRetriever(skills, model_name=dense_model),
    )


def _build_indexed_retrievers(indexed_skills, args):
    bm25 = BM25Retriever(indexed_skills)
    if args.dense_backend == "bigmodel":
        dense = BigModelDenseRetriever(
            indexed_skills,
            model_name=args.bigmodel_embedding_model,
            dimensions=args.bigmodel_embedding_dimensions,
        )
    else:
        dense = DenseRetriever(indexed_skills, model_name=args.dense_model)
    return bm25, dense


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


def _print_stage_reroute_report(report: dict[str, object]) -> None:
    print("=== STAGE REROUTE: RAW EVIDENCE ONLY ===")
    print(
        f"{'STAGE':<12} {'ONE':>7} {'REROUTE':>9} "
        f"{'NEW':>7} {'INCR':>7} {'SIZE':>6}  REROUTE_MISSING"
    )
    print("-" * 82)

    rows = report["stages"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        required = row["required_now"]
        one_hits = row["one_shot_required_hits"]
        reroute_hits = row["reroute_required_hits"]
        new_required = row["new_required"]
        new_hits = row["transition_new_hits"]
        incremental_target = row["incremental_recovery_target"]
        incremental_hits = row["incremental_recovery_hits"]
        missing = row["reroute_missing_required"]
        assert isinstance(required, list)
        assert isinstance(one_hits, list)
        assert isinstance(reroute_hits, list)
        assert isinstance(new_required, list)
        assert isinstance(new_hits, list)
        assert isinstance(incremental_target, list)
        assert isinstance(incremental_hits, list)
        assert isinstance(missing, list)

        stage_name = f"{row['case_id']}/{row['stage_id']}"
        new_display = "-" if not new_required or row["stage_id"] == "S1" else (
            f"{len(new_hits)}/{len(new_required)}"
        )
        incr_display = "-" if not incremental_target else (
            f"{len(incremental_hits)}/{len(incremental_target)}"
        )
        print(
            f"{stage_name:<12} "
            f"{len(one_hits):>2}/{len(required):<2} "
            f"{len(reroute_hits):>4}/{len(required):<2} "
            f"{new_display:>7} "
            f"{incr_display:>7} "
            f"{int(row['reroute_candidate_set_size']):>6}  "
            f"{', '.join(missing) if missing else '-'}"
        )

    print()
    print(
        "one_shot_stage_full_coverage: "
        f"{float(report['one_shot_stage_full_coverage']):.4f}"
    )
    print(
        "reroute_stage_full_coverage: "
        f"{float(report['reroute_stage_full_coverage']):.4f}"
    )
    print(f"reroute_gain: {float(report['reroute_gain']):+.4f}")
    print(
        "transition_new_skill_recall: "
        f"{float(report['transition_new_skill_recall']):.4f} "
        f"({int(report['transition_new_skill_hits'])}/"
        f"{int(report['transition_new_skill_count'])})"
    )
    print(
        "incremental_recovery: "
        f"{float(report['incremental_recovery']):.4f} "
        f"({int(report['incremental_recovery_hits'])}/"
        f"{int(report['incremental_recovery_target_count'])})"
    )
    print(
        "already_present_transition_skills: "
        f"{int(report['already_present_transition_skill_count'])}"
    )
    print(
        "average_reroute_candidate_set_size: "
        f"{float(report['average_reroute_candidate_set_size']):.2f}"
    )



def _rank_text(value: object) -> str:
    return "-" if value is None else str(value)


def _print_stage_diagnostic_report(report: dict[str, object]) -> None:
    print("=== STAGE RETRIEVAL DIAGNOSTICS: A/B/C ===")
    print("A = initial_task")
    print("B = initial_task + raw runtime evidence")
    print("C = raw runtime evidence only")
    print()
    print(
        f"{'TARGET':<34} "
        f"{'A BM25':>7} {'A DNS':>6} {'A@K':>4}  "
        f"{'B BM25':>7} {'B DNS':>6} {'B@K':>4}  "
        f"{'C BM25':>7} {'C DNS':>6} {'C@K':>4}  "
        "DIAGNOSIS"
    )
    print("-" * 122)

    rows = report["targets"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        a = row["A_initial_task"]
        b = row["B_initial_plus_raw"]
        c = row["C_raw_only"]
        assert isinstance(a, dict)
        assert isinstance(b, dict)
        assert isinstance(c, dict)
        target_name = f"{row['case_id']}/{row['stage_id']}:{row['target_skill']}"
        print(
            f"{target_name:<34} "
            f"{_rank_text(a['bm25_rank']):>7} "
            f"{_rank_text(a['dense_rank']):>6} "
            f"{('Y' if a['union_hit'] else 'N'):>4}  "
            f"{_rank_text(b['bm25_rank']):>7} "
            f"{_rank_text(b['dense_rank']):>6} "
            f"{('Y' if b['union_hit'] else 'N'):>4}  "
            f"{_rank_text(c['bm25_rank']):>7} "
            f"{_rank_text(c['dense_rank']):>6} "
            f"{('Y' if c['union_hit'] else 'N'):>4}  "
            f"{row['diagnosis']}"
        )

    print()
    print(f"diagnosed_target_count: {int(report['diagnosed_target_count'])}")
    counts = report["diagnosis_counts"]
    assert isinstance(counts, dict)
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


def _print_stage_bundle_report(report: dict[str, object]) -> None:
    print("=== STAGE CAPABILITY BUNDLE EXPERIMENT ===")
    print(f"cases: {int(report['case_count'])}")
    print(f"stages: {int(report['stage_count'])}")
    print(f"k_per_retriever: {int(report['k_per_retriever'])}")
    print(f"max_bundles: {int(report['max_bundles'])}")
    print(f"max_skills_per_bundle: {int(report['max_skills_per_bundle'])}")
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

    rows = report["cases"]
    assert isinstance(rows, list)
    for case in rows:
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
                f"registered={int(stage['registered_skill_count'])} "
                f"required_recall={float(stage['shelf_required_recall']):.3f}"
            )


def _print_control_plane_report(report: dict[str, object]) -> None:
    multi = report["multi_skill"]
    stage = report["stage_transition"]
    assert isinstance(multi, dict)
    assert isinstance(stage, dict)

    print("=== MULTI-SKILL ===")
    print(f"{'CASE':<9} {'HIT':>7} {'SIZE':>6}  MISSING")
    print("-" * 46)
    multi_rows = multi["cases"]
    assert isinstance(multi_rows, list)
    for row in multi_rows:
        assert isinstance(row, dict)
        required = row["required"]
        hits = row["required_hits"]
        missing = row["missing_required"]
        assert isinstance(required, list)
        assert isinstance(hits, list)
        assert isinstance(missing, list)
        print(
            f"{str(row['case_id']):<9} "
            f"{len(hits):>2}/{len(required):<2} "
            f"{int(row['candidate_set_size']):>6}  "
            f"{', '.join(missing) if missing else '-'}"
        )

    print()
    print(
        "required_skill_recall: "
        f"{float(multi['required_skill_recall']):.4f} "
        f"({int(multi['required_skill_hits'])}/{int(multi['required_skill_count'])})"
    )
    print(
        "full_required_set_coverage: "
        f"{float(multi['full_required_set_coverage']):.4f}"
    )
    print(
        "average_candidate_set_size: "
        f"{float(multi['average_candidate_set_size']):.2f}"
    )

    print()
    _print_stage_reroute_report(stage)
    print()
    print("activation_metrics: unavailable (runtime judge not implemented)")


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

    if args.command == "corpus" and args.corpus_command == "retrieval-cards":
        from skill_control_plane.evals.legacy.capability_need import command_completer

        skills = load_skill_tree(args.root)
        try:
            stats = build_retrieval_card_cache(
                skills,
                extractor=LLMRetrievalCardExtractor(
                    command_completer(args.extract_command)
                ),
                output=args.output,
                force=args.force,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "RetrievalCard extractor command failed "
                f"(exit={exc.returncode}):\n{_redact_child_stderr(exc.stderr)}"
            ) from exc
        print(
            json.dumps(
                {
                    "version": RETRIEVAL_CARD_VERSION,
                    "output": args.output,
                    **stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "corpus" and args.corpus_command == "dense-index":
        index = build_dense_index(
            skill_root=args.root,
            retrieval_cards=args.retrieval_cards,
            embedding_client=BigModelEmbeddingClient(),
        )
        output = write_dense_index(index, args.output)
        print(json.dumps({
            "version": index.version,
            "embedding_model": index.embedding_model,
            "dimensions": index.dimensions,
            "record_count": len(index.records),
            "output": str(output),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval" and args.eval_command == "retrieval":
        _validate_snapshot(args.gold, args.manifest)
        cases = load_runtime_retrieval_gold(args.gold)
        bm25, dense = _build_retrievers(args.root, args.dense_model)

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

    if args.command == "eval" and args.eval_command == "stage-reroute":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        bm25, dense = _build_retrievers(args.root, args.dense_model)

        report = evaluate_stage_reroute(
            stage_cases,
            bm25=bm25,
            dense=dense,
            k=args.k,
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_stage_reroute_report(report)
        return 0


    if args.command == "eval" and args.eval_command == "stage-diagnose":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        bm25, dense = _build_retrievers(args.root, args.dense_model)

        report = diagnose_stage_retrieval(
            stage_cases,
            bm25=bm25,
            dense=dense,
            k=args.k,
            rank_depth=len(bm25.skills),
            only_missed=not args.all_targets,
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_stage_diagnostic_report(report)
        return 0

    if args.command == "eval" and args.eval_command == "stage-bundle":
        if args.capability_need_ab and not args.capability_need_command:
            raise ValueError("--capability-need-ab requires --capability-need-command")
        if args.soft_bundle_repair_ab and not args.capability_need_command:
            raise ValueError("--soft-bundle-repair-ab requires --capability-need-command")
        if args.capability_need_command and not (args.capability_need_ab or args.soft_bundle_repair_ab):
            raise ValueError("--capability-need-command requires --capability-need-ab")
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        skills = load_skill_tree(args.root)
        records = {skill.skill_id: skill for skill in skills}
        metadata_skills = [replace(skill, body="") for skill in skills]
        bm25 = BM25Retriever(metadata_skills)
        dense = DenseRetriever(skills, model_name=args.dense_model)

        evaluator = evaluate_stage_bundle_cases
        extra = {}
        if args.hierarchical_ab:
            from skill_control_plane.evals.hierarchical import evaluate_hierarchical_cases
            from skill_control_plane.evals.legacy.hierarchical import dense_factory_for
            evaluator = evaluate_hierarchical_cases
            extra = {"dense_factory": dense_factory_for(dense)}
        if args.capability_need_ab:
            from skill_control_plane.evals.hierarchical import evaluate_capability_need_cases
            from skill_control_plane.evals.legacy.hierarchical import dense_factory_for
            from skill_control_plane.evals.legacy.capability_need import (
                LLMCapabilityNeedExtractor, command_completer,
            )
            evaluator = evaluate_capability_need_cases
            extra = {"dense_factory": dense_factory_for(dense),
                     "extractor": LLMCapabilityNeedExtractor(
                         command_completer(args.capability_need_command))}
        if args.soft_bundle_ab or args.soft_bundle_repair_ab:
            from skill_control_plane.evals.soft_bundle import evaluate_soft_bundle_cases
            from skill_control_plane.evals.legacy.hierarchical import dense_factory_for
            evaluator = evaluate_soft_bundle_cases
            extra = {"dense_factory": dense_factory_for(dense)}
            if args.soft_bundle_repair_ab:
                from skill_control_plane.evals.legacy.capability_need import LLMCapabilityNeedExtractor, command_completer
                extra["extractor"] = LLMCapabilityNeedExtractor(command_completer(args.capability_need_command))
        report = evaluator(
            stage_cases,
            records,
            bm25=bm25,
            dense=dense,
            k=args.k,
            max_bundles=args.max_bundles,
            max_skills_per_bundle=args.max_skills_per_bundle,
            **extra,
        )

        if args.as_json or args.hierarchical_ab or args.soft_bundle_ab or args.soft_bundle_repair_ab:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.capability_need_ab:
            from skill_control_plane.evals.hierarchical import print_capability_need_report
            print_capability_need_report(report)
        else:
            _print_stage_bundle_report(report)
        return 0

    if args.command == "eval" and args.eval_command == "capability-facets":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        if args.dense_backend == "bigmodel":
            skills = load_skill_tree(args.root)
            metadata_skills = [replace(skill, body="") for skill in skills]
            bm25 = BM25Retriever(metadata_skills)
            dense = BigModelDenseRetriever(
                skills,
                model_name=args.bigmodel_embedding_model,
                dimensions=args.bigmodel_embedding_dimensions,
            )
        else:
            bm25, dense = _build_retrievers(args.root, args.dense_model)

        from skill_control_plane.evals.legacy.capability_facets import (
            LLMCapabilityFacetExtractor,
        )
        from skill_control_plane.evals.legacy.capability_need import (
            LLMCapabilityNeedExtractor,
            command_completer,
        )

        report = evaluate_capability_facet_retrieval(
            stage_cases,
            bm25=bm25,
            dense=dense,
            old_extractor=LLMCapabilityNeedExtractor(
                command_completer(args.old_rewrite_command)
            ),
            facet_extractor=LLMCapabilityFacetExtractor(
                command_completer(args.capability_facets_command)
            ),
            per_query_k=args.per_query_k,
            rrf_k=args.rrf_k,
            cutoffs=(5, 10),
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_capability_facet_report(report)
        return 0

    if args.command == "eval" and args.eval_command == "retrieval-ablation":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        _validate_root_snapshot(args.root, args.manifest)

        skills = load_skill_tree(args.root)
        if args.skill_representation == "retrieval-card":
            if not args.retrieval_cards:
                raise ValueError(
                    "--skill-representation retrieval-card requires --retrieval-cards"
                )
            cards = load_retrieval_cards(args.retrieval_cards)
            indexed_skills = apply_retrieval_cards(skills, cards)
            skill_representation = RETRIEVAL_CARD_VERSION
        else:
            indexed_skills = [replace(skill, body="") for skill in skills]
            skill_representation = "metadata-v0.1"

        bm25 = BM25Retriever(indexed_skills)
        if args.dense_backend == "bigmodel":
            dense = BigModelDenseRetriever(
                indexed_skills,
                model_name=args.bigmodel_embedding_model,
                dimensions=args.bigmodel_embedding_dimensions,
            )
        else:
            dense = DenseRetriever(indexed_skills, model_name=args.dense_model)

        from skill_control_plane.evals.legacy.capability_need import (
            LLMCapabilityNeedExtractor,
            command_completer,
        )

        report = evaluate_frozen_query_retrieval_ablation(
            stage_cases,
            bm25=bm25,
            dense=dense,
            old_extractor=LLMCapabilityNeedExtractor(
                command_completer(args.old_rewrite_command)
            ),
            per_retriever_k=args.per_retriever_k,
            rrf_k=args.rrf_k,
            cutoffs=(5, 10),
            skill_representation=skill_representation,
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_frozen_query_retrieval_ablation(report)
        return 0

    if args.command == "eval" and args.eval_command == "retrieval-field-ablation":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        _validate_root_snapshot(args.root, args.manifest)
        require_min_target_transitions(
            stage_cases,
            minimum=args.min_target_transitions,
            allow_small_sample=args.allow_small_sample,
        )

        from skill_control_plane.evals.legacy.capability_need import (
            LLMCapabilityNeedExtractor,
            command_completer,
        )

        skills = load_skill_tree(args.root)
        cards = load_retrieval_cards(args.retrieval_cards)
        old_extractor = LLMCapabilityNeedExtractor(
            command_completer(args.old_rewrite_command)
        )

        reports = {}
        metadata_skills = [replace(skill, body="") for skill in skills]
        bm25, dense = _build_indexed_retrievers(metadata_skills, args)
        reports["metadata-v0.1"] = evaluate_frozen_query_retrieval_ablation(
            stage_cases,
            bm25=bm25,
            dense=dense,
            old_extractor=old_extractor,
            per_retriever_k=args.per_retriever_k,
            rrf_k=args.rrf_k,
            cutoffs=(5, 10),
            skill_representation="metadata-v0.1",
        )

        for label, include_fields in RETRIEVAL_CARD_FIELD_ABLATIONS:
            indexed_skills = apply_retrieval_cards(
                skills,
                cards,
                include_fields=include_fields,
            )
            bm25, dense = _build_indexed_retrievers(indexed_skills, args)
            reports[label] = evaluate_frozen_query_retrieval_ablation(
                stage_cases,
                bm25=bm25,
                dense=dense,
                old_extractor=old_extractor,
                per_retriever_k=args.per_retriever_k,
                rrf_k=args.rrf_k,
                cutoffs=(5, 10),
                skill_representation=f"{RETRIEVAL_CARD_VERSION}/{label}",
            )

        summary = summarize_field_ablation_reports(reports)
        if args.as_json:
            print(
                json.dumps(
                    {"summary": summary, "reports": reports},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_field_ablation_summary(summary)
        return 0

    if args.command == "eval" and args.eval_command == "retrieval-robustness":
        stage_cases = load_stage_transition_gold(args.gold)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        _validate_root_snapshot(args.root, args.manifest)
        require_min_target_transitions(
            stage_cases,
            minimum=args.min_target_transitions,
            allow_small_sample=args.allow_small_sample,
        )

        from skill_control_plane.evals.query_robustness import (
            load_query_variant_sets,
        )
        from skill_control_plane.evals.legacy.capability_need import (
            LLMCapabilityNeedExtractor,
            command_completer,
        )

        old_extractor = LLMCapabilityNeedExtractor(
            command_completer(args.old_rewrite_command)
        )
        cache_stats = build_query_variant_cache(
            stage_cases,
            old_extractor=old_extractor,
            paraphraser=LLMQueryParaphraser(
                command_completer(args.paraphrase_command)
            ),
            output=args.query_variants,
            paraphrase_count=args.paraphrases_per_query,
            force=args.force_paraphrases,
        )
        query_sets = load_query_variant_sets(args.query_variants)

        skills = load_skill_tree(args.root)
        cards = load_retrieval_cards(args.retrieval_cards)

        metadata_skills = [replace(skill, body="") for skill in skills]
        metadata_bm25, metadata_dense = _build_indexed_retrievers(
            metadata_skills, args
        )
        metadata_report = evaluate_query_variant_retrieval(
            stage_cases,
            query_sets=query_sets,
            bm25=metadata_bm25,
            dense=metadata_dense,
            per_retriever_k=args.per_retriever_k,
            rrf_k=args.rrf_k,
            cutoffs=(5, 10),
            skill_representation="metadata-v0.1",
        )

        card_skills = apply_retrieval_cards(skills, cards)
        card_bm25, card_dense = _build_indexed_retrievers(card_skills, args)
        card_report = evaluate_query_variant_retrieval(
            stage_cases,
            query_sets=query_sets,
            bm25=card_bm25,
            dense=card_dense,
            per_retriever_k=args.per_retriever_k,
            rrf_k=args.rrf_k,
            cutoffs=(5, 10),
            skill_representation=RETRIEVAL_CARD_VERSION,
        )

        comparison = compare_query_robustness_reports(
            metadata_report,
            card_report,
        )
        if args.as_json:
            print(
                json.dumps(
                    {
                        "query_variant_cache": cache_stats,
                        "comparison": comparison,
                        "metadata_report": metadata_report,
                        "retrieval_card_report": card_report,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_query_robustness_comparison(comparison)
        return 0

    if args.command == "eval" and args.eval_command == "control-plane":
        multi_cases = load_multi_skill_gold(args.multi_skill)
        stage_cases = load_stage_transition_gold(args.stage_transition)
        _validate_snapshot_id(multi_cases[0].snapshot_id, args.manifest)
        _validate_snapshot_id(stage_cases[0].snapshot_id, args.manifest)
        if multi_cases[0].snapshot_id != stage_cases[0].snapshot_id:
            raise ValueError("multi-skill and stage-transition Gold snapshots differ")

        bm25, dense = _build_retrievers(args.root, args.dense_model)
        report = evaluate_control_plane(
            multi_cases,
            stage_cases,
            bm25=bm25,
            dense=dense,
            k=args.k,
        )

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_control_plane_report(report)
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
