from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill_control_plane.discovery import candidate_union


@dataclass(frozen=True, slots=True)
class RuntimeRetrievalGoldCase:
    case_id: str
    query: str
    required: tuple[str, ...]
    useful: tuple[str, ...]
    hard_negative: tuple[str, ...]
    rationale: str
    snapshot_id: str


def load_runtime_retrieval_gold(path: str | Path) -> list[RuntimeRetrievalGoldCase]:
    cases: list[RuntimeRetrievalGoldCase] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        required = tuple(payload.get("required", ()))
        if not required:
            raise ValueError(f"gold line {line_number} has no required Skills")
        cases.append(
            RuntimeRetrievalGoldCase(
                case_id=str(payload["case_id"]),
                query=str(payload["query"]),
                required=required,
                useful=tuple(payload.get("useful", ())),
                hard_negative=tuple(payload.get("hard_negative", ())),
                rationale=str(payload.get("rationale", "")),
                snapshot_id=str(payload["snapshot_id"]),
            )
        )

    if not cases:
        raise ValueError("gold set is empty")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("gold set contains duplicate case_id values")

    snapshot_ids = {case.snapshot_id for case in cases}
    if len(snapshot_ids) != 1:
        raise ValueError("gold set must reference exactly one snapshot_id")

    return cases


def evaluate_union_retrieval(
    cases: list[RuntimeRetrievalGoldCase],
    *,
    bm25: Any,
    dense: Any,
    k: int,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")

    rows: list[dict[str, Any]] = []
    required_total = 0
    required_recalled = 0
    fully_covered_cases = 0
    candidate_total = 0

    for case in cases:
        bm25_candidates = bm25.search(case.query, k=k)
        dense_candidates = dense.search(case.query, k=k)
        union_candidates = candidate_union(
            {
                "bm25": bm25_candidates,
                "dense": dense_candidates,
            }
        )

        bm25_ids = {candidate.skill_id for candidate in bm25_candidates}
        dense_ids = {candidate.skill_id for candidate in dense_candidates}
        union_ids = {candidate.skill_id for candidate in union_candidates}
        required = set(case.required)
        recalled = required & union_ids
        missing = required - union_ids

        required_total += len(required)
        required_recalled += len(recalled)
        candidate_total += len(union_ids)
        if not missing:
            fully_covered_cases += 1

        rows.append(
            {
                "case_id": case.case_id,
                "required": list(case.required),
                "bm25_required_hits": sorted(required & bm25_ids),
                "dense_required_hits": sorted(required & dense_ids),
                "union_required_hits": sorted(recalled),
                "missing_required": sorted(missing),
                "candidate_set_size": len(union_ids),
                "candidate_ids": [candidate.skill_id for candidate in union_candidates],
            }
        )

    case_count = len(cases)
    return {
        "case_count": case_count,
        "k_per_retriever": k,
        "required_skill_count": required_total,
        "recalled_required_skill_count": required_recalled,
        "candidate_recall": required_recalled / required_total if required_total else 1.0,
        "full_case_coverage": fully_covered_cases / case_count if case_count else 1.0,
        "average_candidate_set_size": candidate_total / case_count if case_count else 0.0,
        "cases": rows,
    }
