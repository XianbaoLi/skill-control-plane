"""Frozen-query retrieval ablation: Dense, BM25, Union, and RRF."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.control_plane import StageTransitionGoldCase
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.discovery import candidate_union, reciprocal_rank_fusion
from skill_control_plane.discovery.base import Retriever
from skill_control_plane.runtime.capability_need import (
    CapabilityNeedExtractor,
    extract_query,
)


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


def _ranked_arm(
    required: Sequence[str],
    ranking: Sequence[RetrievalCandidate],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    ranks = _required_ranks(required, ranking)
    return {
        "new_required_ranks": ranks,
        "recall_at": {
            str(cutoff): _recall_at(ranks, cutoff)
            for cutoff in cutoffs
        },
        "top10": [candidate.skill_id for candidate in ranking[:10]],
    }


def evaluate_frozen_query_retrieval_ablation(
    cases: Sequence[StageTransitionGoldCase],
    *,
    bm25: Retriever,
    dense: Retriever,
    old_extractor: CapabilityNeedExtractor,
    per_retriever_k: int = 10,
    rrf_k: int = 60,
    cutoffs: tuple[int, ...] = (5, 10),
    skill_representation: str = "metadata-v0.1",
) -> dict[str, Any]:
    """Compare retrieval/fusion only while keeping the old query fixed.

    A = Dense only
    B = BM25 only
    C = BM25@K union Dense@K (candidate-generation view; up to 2K candidates)
    D = BM25@K + Dense@K reciprocal-rank fusion (ranked view)

    The query generator is invoked once per runtime transition. When the old
    extractor is backed by an exact replay adapter, all four arms see the
    identical frozen query.
    """

    if per_retriever_k < 1:
        raise ValueError("per_retriever_k must be >= 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if not cutoffs or any(value < 1 for value in cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    if max(cutoffs) > per_retriever_k:
        raise ValueError("cutoffs cannot exceed per_retriever_k")

    cutoffs = tuple(sorted(set(cutoffs)))
    rows: list[dict[str, Any]] = []
    required_occurrences = 0
    extraction_ok = 0
    ranked_hits = {
        arm: {cutoff: 0 for cutoff in cutoffs}
        for arm in ("dense", "bm25", "rrf")
    }
    union_hits = 0
    union_pool_size = 0
    union_full_transition_count = 0
    target_transition_count = 0

    for case in cases:
        for stage in case.stages[1:]:
            required = tuple(stage.new_required)
            required_occurrences += len(required)
            if required:
                target_transition_count += 1

            frozen = extract_query(
                old_extractor,
                case.initial_task,
                stage.runtime_evidence,
            )
            extraction_ok += frozen.status == "ok"
            query = frozen.query

            dense_ranking = dense.search(query, k=per_retriever_k)
            bm25_ranking = bm25.search(query, k=per_retriever_k)
            source_rankings = {
                "bm25": bm25_ranking,
                "dense": dense_ranking,
            }
            union_ranking = candidate_union(source_rankings)
            rrf_ranking = reciprocal_rank_fusion(source_rankings, k=rrf_k)

            dense_arm = _ranked_arm(required, dense_ranking, cutoffs)
            bm25_arm = _ranked_arm(required, bm25_ranking, cutoffs)
            rrf_arm = _ranked_arm(required, rrf_ranking, cutoffs)

            for arm_name, arm in (
                ("dense", dense_arm),
                ("bm25", bm25_arm),
                ("rrf", rrf_arm),
            ):
                ranks = arm["new_required_ranks"]
                for cutoff in cutoffs:
                    ranked_hits[arm_name][cutoff] += sum(
                        rank is not None and rank <= cutoff
                        for rank in ranks.values()
                    )

            union_ids = [candidate.skill_id for candidate in union_ranking]
            union_required_hits = [
                skill_id for skill_id in required if skill_id in union_ids
            ]
            union_hits += len(union_required_hits)
            union_pool_size += len(union_ranking)
            if required and len(union_required_hits) == len(required):
                union_full_transition_count += 1

            rows.append(
                {
                    "case_id": case.case_id,
                    "stage_id": stage.stage_id,
                    "runtime_evidence": list(stage.runtime_evidence),
                    "new_required": list(required),
                    "query": query,
                    "query_status": frozen.status,
                    "query_confidence": (
                        frozen.need.confidence if frozen.need else None
                    ),
                    "dense": dense_arm,
                    "bm25": bm25_arm,
                    "union": {
                        "candidate_ids": union_ids,
                        "candidate_set_size": len(union_ranking),
                        "new_required_hits": union_required_hits,
                        "candidate_recall": (
                            len(union_required_hits) / len(required)
                            if required else None
                        ),
                    },
                    "rrf": rrf_arm,
                }
            )

    transition_count = len(rows)
    denominator = required_occurrences or 1
    ranked_metrics = {
        arm: {
            f"recall_at_{cutoff}": ranked_hits[arm][cutoff] / denominator
            for cutoff in cutoffs
        }
        for arm in ("dense", "bm25", "rrf")
    }

    return {
        "comparison": "frozen old query: Dense vs BM25 vs Union vs RRF",
        "scope": "retrieval only; runtime transitions only; S1 excluded",
        "query_policy": "one identical old capability_need query per transition",
        "skill_representation": skill_representation,
        "transition_count": transition_count,
        "target_transition_count": target_transition_count,
        "new_required_skill_occurrences": required_occurrences,
        "per_retriever_k": per_retriever_k,
        "rrf_k": rrf_k,
        "cutoffs": list(cutoffs),
        "query_extraction_ok_rate": (
            extraction_ok / transition_count if transition_count else 0.0
        ),
        "dense": ranked_metrics["dense"],
        "bm25": ranked_metrics["bm25"],
        "union": {
            "candidate_recall": union_hits / denominator,
            "average_candidate_set_size": (
                union_pool_size / transition_count if transition_count else 0.0
            ),
            "max_candidate_budget": 2 * per_retriever_k,
            "full_transition_coverage": (
                union_full_transition_count / target_transition_count
                if target_transition_count else 0.0
            ),
        },
        "rrf": ranked_metrics["rrf"],
        "stages": rows,
    }


def print_frozen_query_retrieval_ablation(report: Mapping[str, Any]) -> None:
    print("=== FROZEN OLD QUERY RETRIEVAL A/B/C/D ===")
    print("A = Dense only")
    print("B = BM25 only")
    print("C = BM25@K union Dense@K")
    print("D = BM25@K + Dense@K RRF")
    print(f"skill_representation: {report['skill_representation']}")
    print(f"transitions: {report['transition_count']}")
    print(f"target_transitions: {report['target_transition_count']}")
    print(
        "new_required_skill_occurrences: "
        f"{report['new_required_skill_occurrences']}"
    )
    print(f"per_retriever_k: {report['per_retriever_k']}")
    print(f"rrf_k: {report['rrf_k']}")
    print(
        "query_extraction_ok_rate: "
        f"{float(report['query_extraction_ok_rate']):.4f}"
    )

    dense = report["dense"]
    bm25 = report["bm25"]
    rrf = report["rrf"]
    union = report["union"]
    assert isinstance(dense, Mapping)
    assert isinstance(bm25, Mapping)
    assert isinstance(rrf, Mapping)
    assert isinstance(union, Mapping)

    print()
    print("Ranked retrieval (same final cutoff):")
    for cutoff in report["cutoffs"]:
        key = f"recall_at_{cutoff}"
        print(
            f"Recall@{cutoff}: "
            f"Dense={float(dense[key]):.4f} "
            f"BM25={float(bm25[key]):.4f} "
            f"RRF={float(rrf[key]):.4f}"
        )

    print()
    print("Union candidate-generation view:")
    print(f"candidate_recall: {float(union['candidate_recall']):.4f}")
    print(
        "average_candidate_set_size: "
        f"{float(union['average_candidate_set_size']):.2f}"
    )
    print(f"max_candidate_budget: {int(union['max_candidate_budget'])}")
    print(
        "full_transition_coverage: "
        f"{float(union['full_transition_coverage']):.4f}"
    )

    for row in report["stages"]:
        print(f"\\n{row['case_id']}/{row['stage_id']}")
        print(f"  new_required: {row['new_required']}")
        print(f"  frozen_query: {row['query']}")
        print(f"  Dense ranks: {row['dense']['new_required_ranks']}")
        print(f"  BM25 ranks: {row['bm25']['new_required_ranks']}")
        print(
            "  Union hits/size: "
            f"{row['union']['new_required_hits']} / "
            f"{row['union']['candidate_set_size']}"
        )
        print(f"  RRF ranks: {row['rrf']['new_required_ranks']}")
