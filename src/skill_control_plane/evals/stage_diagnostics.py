from __future__ import annotations

from typing import Any

from skill_control_plane.evals.control_plane import (
    StageTransitionGoldCase,
    stage_retrieval_query,
)
from skill_control_plane.discovery import candidate_union


def _raw_evidence_query(case: StageTransitionGoldCase, stage_index: int) -> str:
    return "\n".join(case.stages[stage_index].runtime_evidence)


def _rank_map(candidates: list[Any]) -> dict[str, int]:
    return {candidate.skill_id: candidate.rank for candidate in candidates}


def _variant_result(
    query: str,
    *,
    target: str,
    bm25: Any,
    dense: Any,
    k: int,
    rank_depth: int,
) -> dict[str, Any]:
    bm25_full = bm25.search(query, k=rank_depth)
    dense_full = dense.search(query, k=rank_depth)
    bm25_top = bm25_full[:k]
    dense_top = dense_full[:k]
    union_top = candidate_union({"bm25": bm25_top, "dense": dense_top})

    bm25_ranks = _rank_map(bm25_full)
    dense_ranks = _rank_map(dense_full)
    union_ids = [candidate.skill_id for candidate in union_top]

    return {
        "query": query,
        "bm25_rank": bm25_ranks.get(target),
        "dense_rank": dense_ranks.get(target),
        "union_hit": target in set(union_ids),
        "union_candidate_ids": union_ids,
    }


def _best_rank(*variants: dict[str, Any]) -> int | None:
    ranks: list[int] = []
    for variant in variants:
        for key in ("bm25_rank", "dense_rank"):
            value = variant[key]
            if isinstance(value, int):
                ranks.append(value)
    return min(ranks) if ranks else None


def _diagnosis(
    *,
    b: dict[str, Any],
    c: dict[str, Any],
    k: int,
) -> str:
    if not b["union_hit"] and c["union_hit"]:
        return "ANCHORING_SIGNAL"

    best = _best_rank(b, c)
    if best is not None and k < best <= 2 * k:
        return "TOPK_BUDGET_SIGNAL"

    return "REPRESENTATION_OR_RETRIEVER_SIGNAL"


def diagnose_stage_retrieval(
    cases: list[StageTransitionGoldCase],
    *,
    bm25: Any,
    dense: Any,
    k: int,
    rank_depth: int,
    only_missed: bool = True,
) -> dict[str, Any]:
    """Compare three query constructions for stage-transition retrieval.

    A: initial_task only
    B: initial_task + raw runtime evidence (current L1 reroute)
    C: raw runtime evidence only

    By default only required Skills missed by B are reported.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if rank_depth < k:
        raise ValueError("rank_depth must be >= k")

    rows: list[dict[str, Any]] = []

    for case in cases:
        for stage_index, stage in enumerate(case.stages):
            if stage_index == 0:
                continue

            query_a = case.initial_task
            query_b = stage_retrieval_query(case, stage)
            query_c = _raw_evidence_query(case, stage_index)

            for target in stage.required_now:
                a = _variant_result(
                    query_a,
                    target=target,
                    bm25=bm25,
                    dense=dense,
                    k=k,
                    rank_depth=rank_depth,
                )
                b = _variant_result(
                    query_b,
                    target=target,
                    bm25=bm25,
                    dense=dense,
                    k=k,
                    rank_depth=rank_depth,
                )
                c = _variant_result(
                    query_c,
                    target=target,
                    bm25=bm25,
                    dense=dense,
                    k=k,
                    rank_depth=rank_depth,
                )

                if only_missed and b["union_hit"]:
                    continue

                rows.append(
                    {
                        "case_id": case.case_id,
                        "stage_id": stage.stage_id,
                        "target_skill": target,
                        "is_new_required": target in set(stage.new_required),
                        "A_initial_task": a,
                        "B_initial_plus_raw": b,
                        "C_raw_only": c,
                        "diagnosis": _diagnosis(b=b, c=c, k=k),
                    }
                )

    diagnosis_counts: dict[str, int] = {}
    for row in rows:
        label = str(row["diagnosis"])
        diagnosis_counts[label] = diagnosis_counts.get(label, 0) + 1

    return {
        "k_per_retriever": k,
        "rank_depth": rank_depth,
        "only_missed": only_missed,
        "diagnosed_target_count": len(rows),
        "diagnosis_counts": diagnosis_counts,
        "targets": rows,
    }
