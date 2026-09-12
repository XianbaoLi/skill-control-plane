"""V0.4 retrieval-only A/B: single rewrite vs capability facets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.control_plane import StageTransitionGoldCase
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.discovery import reciprocal_rank_fusion
from skill_control_plane.discovery.base import Retriever
from skill_control_plane.evals.legacy.capability_facets import (
    CapabilityFacetExtractor,
    extract_facet_queries,
)
from skill_control_plane.evals.legacy.capability_need import (
    CapabilityNeedExtractor,
    extract_query,
)


def _fuse_queries(
    queries: Sequence[str],
    *,
    bm25: Retriever,
    dense: Retriever,
    per_query_k: int,
    rrf_k: int,
) -> list[RetrievalCandidate]:
    rankings: dict[str, Sequence[RetrievalCandidate]] = {}
    for index, query in enumerate(queries):
        rankings[f"q{index}:bm25"] = bm25.search(query, k=per_query_k)
        rankings[f"q{index}:dense"] = dense.search(query, k=per_query_k)
    return reciprocal_rank_fusion(rankings, k=rrf_k)


def _required_ranks(
    required: Sequence[str],
    ranking: Sequence[RetrievalCandidate],
) -> dict[str, int | None]:
    positions = {candidate.skill_id: candidate.rank for candidate in ranking}
    return {skill_id: positions.get(skill_id) for skill_id in required}


def _recall_at(ranks: Mapping[str, int | None], cutoff: int) -> float:
    if not ranks:
        return 1.0
    return sum(rank is not None and rank <= cutoff for rank in ranks.values()) / len(ranks)


def evaluate_capability_facet_retrieval(
    cases: Sequence[StageTransitionGoldCase],
    *,
    bm25: Retriever,
    dense: Retriever,
    old_extractor: CapabilityNeedExtractor,
    facet_extractor: CapabilityFacetExtractor,
    per_query_k: int = 10,
    rrf_k: int = 60,
    cutoffs: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Compare query formulations only; Bundle/Shelf/Active are deliberately absent."""

    if per_query_k < 1:
        raise ValueError("per_query_k must be >= 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if not cutoffs or any(value < 1 for value in cutoffs):
        raise ValueError("cutoffs must contain positive integers")

    cutoffs = tuple(sorted(set(cutoffs)))
    rows: list[dict[str, Any]] = []
    new_required_occurrences = 0
    required_now_occurrences = 0
    old_hits = {cutoff: 0 for cutoff in cutoffs}
    facet_hits = {cutoff: 0 for cutoff in cutoffs}
    old_required_now_hits = {cutoff: 0 for cutoff in cutoffs}
    facet_required_now_hits = {cutoff: 0 for cutoff in cutoffs}
    facet_ok = 0
    old_ok = 0

    for case in cases:
        for stage in case.stages[1:]:
            required = tuple(stage.new_required)
            required_now = tuple(stage.required_now)
            new_required_occurrences += len(required)
            required_now_occurrences += len(required_now)

            old = extract_query(
                old_extractor,
                case.initial_task,
                stage.runtime_evidence,
            )
            old_queries = (old.query,)
            old_ranking = _fuse_queries(
                old_queries,
                bm25=bm25,
                dense=dense,
                per_query_k=per_query_k,
                rrf_k=rrf_k,
            )
            old_ranks = _required_ranks(required, old_ranking)
            old_required_now_ranks = _required_ranks(required_now, old_ranking)
            old_ok += old.status == "ok"

            facets = extract_facet_queries(
                facet_extractor,
                case.initial_task,
                stage.runtime_evidence,
            )
            facet_ranking = _fuse_queries(
                facets.queries,
                bm25=bm25,
                dense=dense,
                per_query_k=per_query_k,
                rrf_k=rrf_k,
            )
            facet_ranks = _required_ranks(required, facet_ranking)
            facet_required_now_ranks = _required_ranks(required_now, facet_ranking)
            facet_ok += facets.status == "ok"

            old_recall: dict[str, float] = {}
            facet_recall: dict[str, float] = {}
            for cutoff in cutoffs:
                old_value = _recall_at(old_ranks, cutoff)
                facet_value = _recall_at(facet_ranks, cutoff)
                old_recall[str(cutoff)] = old_value
                facet_recall[str(cutoff)] = facet_value
                old_hits[cutoff] += sum(
                    rank is not None and rank <= cutoff
                    for rank in old_ranks.values()
                )
                facet_hits[cutoff] += sum(
                    rank is not None and rank <= cutoff
                    for rank in facet_ranks.values()
                )
                old_required_now_hits[cutoff] += sum(
                    rank is not None and rank <= cutoff
                    for rank in old_required_now_ranks.values()
                )
                facet_required_now_hits[cutoff] += sum(
                    rank is not None and rank <= cutoff
                    for rank in facet_required_now_ranks.values()
                )

            rows.append(
                {
                    "case_id": case.case_id,
                    "stage_id": stage.stage_id,
                    "runtime_evidence": list(stage.runtime_evidence),
                    "new_required": list(required),
                    "required_now": list(required_now),
                    "old": {
                        "query": old.query,
                        "status": old.status,
                        "confidence": old.need.confidence if old.need else None,
                        "new_required_ranks": old_ranks,
                        "required_now_ranks": old_required_now_ranks,
                        "recall_at": old_recall,
                        "top10": [
                            candidate.skill_id for candidate in old_ranking[:10]
                        ],
                    },
                    "facets": {
                        "queries": list(facets.queries),
                        "status": facets.status,
                        "confidence": (
                            facets.facets.confidence if facets.facets else None
                        ),
                        "evidence_basis": (
                            facets.facets.evidence_basis if facets.facets else ""
                        ),
                        "new_required_ranks": facet_ranks,
                        "required_now_ranks": facet_required_now_ranks,
                        "recall_at": facet_recall,
                        "top10": [
                            candidate.skill_id for candidate in facet_ranking[:10]
                        ],
                    },
                }
            )

    transition_count = len(rows)
    denominator = new_required_occurrences or 1
    required_now_denominator = required_now_occurrences or 1
    old_metrics = {
        f"recall_at_{cutoff}": old_hits[cutoff] / denominator
        for cutoff in cutoffs
    }
    facet_metrics = {
        f"recall_at_{cutoff}": facet_hits[cutoff] / denominator
        for cutoff in cutoffs
    }
    old_required_now_metrics = {
        f"recall_at_{cutoff}": old_required_now_hits[cutoff] / required_now_denominator
        for cutoff in cutoffs
    }
    facet_required_now_metrics = {
        f"recall_at_{cutoff}": facet_required_now_hits[cutoff] / required_now_denominator
        for cutoff in cutoffs
    }

    return {
        "comparison": (
            "old single capability_need rewrite vs main-agent 3-5 capability "
            "facets + multi-query RRF"
        ),
        "scope": "retrieval only; runtime transitions only; S1 excluded",
        "ranking": (
            "both arms use identical BM25+Dense reciprocal-rank fusion; "
            "only query formulation differs"
        ),
        "primary_denominator": "new_required Skill occurrences across runtime transitions",
        "secondary_denominator": "required_now Skill occurrences across runtime transitions",
        "transition_count": transition_count,
        "new_required_skill_occurrences": new_required_occurrences,
        "required_now_skill_occurrences": required_now_occurrences,
        "per_query_k": per_query_k,
        "rrf_k": rrf_k,
        "cutoffs": list(cutoffs),
        "old_extraction_ok_rate": old_ok / transition_count if transition_count else 0.0,
        "facet_extraction_ok_rate": facet_ok / transition_count if transition_count else 0.0,
        "old": old_metrics,
        "facets": facet_metrics,
        "required_now_secondary": {
            "old": old_required_now_metrics,
            "facets": facet_required_now_metrics,
            "delta": {
                key: facet_required_now_metrics[key] - old_required_now_metrics[key]
                for key in old_required_now_metrics
            },
        },
        "delta": {
            key: facet_metrics[key] - old_metrics[key]
            for key in old_metrics
        },
        "stages": rows,
    }


def print_capability_facet_report(report: Mapping[str, Any]) -> None:
    print("=== CAPABILITY FACET RETRIEVAL V0.4 ===")
    print(report["comparison"])
    print(f"transitions: {report['transition_count']}")
    print(f"new_required_skill_occurrences: {report['new_required_skill_occurrences']}")
    print(f"required_now_skill_occurrences: {report['required_now_skill_occurrences']}")
    print(f"per_query_k: {report['per_query_k']}")
    print(f"rrf_k: {report['rrf_k']}")
    print(
        "extraction_ok_rate old/facets: "
        f"{float(report['old_extraction_ok_rate']):.4f} / "
        f"{float(report['facet_extraction_ok_rate']):.4f}"
    )

    old = report["old"]
    facets = report["facets"]
    delta = report["delta"]
    assert isinstance(old, Mapping)
    assert isinstance(facets, Mapping)
    assert isinstance(delta, Mapping)

    for cutoff in report["cutoffs"]:
        key = f"recall_at_{cutoff}"
        print(
            f"Recall@{cutoff}: "
            f"old={float(old[key]):.4f} "
            f"facets={float(facets[key]):.4f} "
            f"delta={float(delta[key]):+.4f}"
        )

    for row in report["stages"]:
        print(f"\n{row['case_id']}/{row['stage_id']}")
        print(f"  new_required: {row['new_required']}")
        print(f"  required_now: {row['required_now']}")
        print(f"  old_status: {row['old']['status']}")
        print(f"  old_query: {row['old']['query']}")
        print(f"  old_new_required_ranks: {row['old']['new_required_ranks']}")
        print(f"  facet_status: {row['facets']['status']}")
        print(f"  facet_queries: {row['facets']['queries']}")
        print(f"  facet_new_required_ranks: {row['facets']['new_required_ranks']}")
