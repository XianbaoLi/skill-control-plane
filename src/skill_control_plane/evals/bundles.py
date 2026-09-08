from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.control_plane import (
    StageGold,
    StageTransitionGoldCase,
)
from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.retrieval import candidate_union
from skill_control_plane.runtime.bundles import (
    CapabilityShelf,
    GroupKey,
    build_capability_shelf,
    default_group_key,
    integrate_retrieval_delta,
)


def _active_skill_ids(shelf: CapabilityShelf) -> set[str]:
    active = set(shelf.active_bundle_ids)
    return {
        skill_id
        for bundle in shelf.bundles
        if bundle.bundle_id in active
        for skill_id in bundle.skill_ids
    }


def _recall(required: Sequence[str], available: set[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 1.0
    return len(required_set & available) / len(required_set)


def evaluate_bundle_trajectory(
    stages: Sequence[StageGold],
    stage_candidates: Mapping[str, Sequence[RetrievalCandidate]],
    records: Mapping[str, SkillRecord],
    *,
    max_bundles: int = 4,
    max_skills_per_bundle: int = 4,
    group_key: GroupKey = default_group_key,
) -> dict[str, Any]:
    """Evaluate a sequential Capability Shelf over one stage trajectory."""

    shelf = CapabilityShelf()
    rows: list[dict[str, Any]] = []
    active_recalls: list[float] = []
    shelf_recalls: list[float] = []
    transition_skill_count = 0
    shelf_reuse_skill_count = 0
    new_bundle_skill_count = 0

    future_transition_skills = {
        skill_id
        for stage in stages[1:]
        for skill_id in stage.new_required
        if skill_id in records
    }
    future_transition_bundles = {
        group_key(records[skill_id])
        for skill_id in future_transition_skills
    }
    initial_shelf_future_skill_hits: set[str] = set()
    initial_shelf_future_bundle_hits: set[str] = set()

    for index, stage in enumerate(stages):
        candidates = stage_candidates.get(stage.stage_id, ())
        before_bundle_ids = {bundle.bundle_id for bundle in shelf.bundles}

        if index == 0:
            shelf = build_capability_shelf(
                candidates,
                records,
                max_bundles=max_bundles,
                max_skills_per_bundle=max_skills_per_bundle,
                group_key=group_key,
            )
            initial_shelf_future_skill_hits = (
                set(shelf.registered_skill_ids) & future_transition_skills
            )
            initial_shelf_future_bundle_hits = (
                {bundle.bundle_id for bundle in shelf.bundles}
                & future_transition_bundles
            )
        else:
            for skill_id in stage.new_required:
                record = records.get(skill_id)
                if record is None:
                    continue
                transition_skill_count += 1
                bundle_id = group_key(record)
                if bundle_id in before_bundle_ids:
                    shelf_reuse_skill_count += 1
                else:
                    new_bundle_skill_count += 1

            shelf = integrate_retrieval_delta(
                shelf,
                candidates,
                records,
                max_new_bundles=max_bundles,
                max_skills_per_bundle=max_skills_per_bundle,
                group_key=group_key,
            )

        active_skills = _active_skill_ids(shelf)
        registered_skills = set(shelf.registered_skill_ids)
        active_recall = _recall(stage.required_now, active_skills)
        shelf_recall = _recall(stage.required_now, registered_skills)
        active_recalls.append(active_recall)
        shelf_recalls.append(shelf_recall)

        rows.append(
            {
                "stage_id": stage.stage_id,
                "active_bundle_ids": list(shelf.active_bundle_ids),
                "bundle_ids": [bundle.bundle_id for bundle in shelf.bundles],
                "registered_skill_ids": list(shelf.registered_skill_ids),
                "registered_skill_count": len(shelf.registered_skill_ids),
                "active_required_recall": active_recall,
                "shelf_required_recall": shelf_recall,
            }
        )

    denominator = transition_skill_count or 1
    future_skill_denominator = len(future_transition_skills) or 1
    future_bundle_denominator = len(future_transition_bundles) or 1

    return {
        "stage_count": len(stages),
        "mean_active_required_recall": (
            sum(active_recalls) / len(active_recalls) if active_recalls else 0.0
        ),
        "mean_shelf_required_recall": (
            sum(shelf_recalls) / len(shelf_recalls) if shelf_recalls else 0.0
        ),
        "shelf_reuse_rate": shelf_reuse_skill_count / denominator,
        "new_bundle_rate": new_bundle_skill_count / denominator,
        "transition_skill_count": transition_skill_count,
        "shelf_reuse_skill_count": shelf_reuse_skill_count,
        "new_bundle_skill_count": new_bundle_skill_count,
        "initial_shelf_future_skill_recall": (
            len(initial_shelf_future_skill_hits) / future_skill_denominator
        ),
        "initial_shelf_future_skill_hits": sorted(
            initial_shelf_future_skill_hits
        ),
        "future_transition_skill_count": len(future_transition_skills),
        "initial_shelf_future_bundle_recall": (
            len(initial_shelf_future_bundle_hits) / future_bundle_denominator
        ),
        "initial_shelf_future_bundle_hits": sorted(
            initial_shelf_future_bundle_hits
        ),
        "future_transition_bundle_count": len(future_transition_bundles),
        "final_registered_skill_count": len(shelf.registered_skill_ids),
        "stages": rows,
    }


def evaluate_stage_bundle_cases(
    cases: Sequence[StageTransitionGoldCase],
    records: Mapping[str, SkillRecord],
    *,
    bm25: Any,
    dense: Any,
    k: int = 5,
    max_bundles: int = 4,
    max_skills_per_bundle: int = 4,
    group_key: GroupKey = default_group_key,
) -> dict[str, Any]:
    """Run the direct-evidence Bundle baseline over stage-transition Gold.

    S1 retrieves from the initial task. Later stages retrieve only from current
    raw runtime evidence, with no SRC/LLM semantic expansion.
    """

    case_rows: list[dict[str, Any]] = []
    total_transition_skills = 0
    total_reused = 0
    total_new_bundle = 0
    total_future_skills = 0
    total_initial_skill_hits = 0
    total_future_bundles = 0
    total_initial_bundle_hits = 0
    weighted_active_recall = 0.0
    weighted_shelf_recall = 0.0
    total_stages = 0

    for case in cases:
        stage_candidates: dict[str, Sequence[RetrievalCandidate]] = {}
        query_rows: list[dict[str, Any]] = []

        for index, stage in enumerate(case.stages):
            query = (
                case.initial_task
                if index == 0
                else "\n".join(stage.runtime_evidence)
            )
            bm25_candidates = bm25.search(query, k=k)
            dense_candidates = dense.search(query, k=k)
            union_candidates = candidate_union(
                {
                    "bm25": bm25_candidates,
                    "dense": dense_candidates,
                }
            )
            stage_candidates[stage.stage_id] = union_candidates
            query_rows.append(
                {
                    "stage_id": stage.stage_id,
                    "query": query,
                    "bm25_ids": [item.skill_id for item in bm25_candidates],
                    "dense_ids": [item.skill_id for item in dense_candidates],
                    "union_ids": [item.skill_id for item in union_candidates],
                }
            )

        trajectory = evaluate_bundle_trajectory(
            case.stages,
            stage_candidates,
            records,
            max_bundles=max_bundles,
            max_skills_per_bundle=max_skills_per_bundle,
            group_key=group_key,
        )
        trajectory["case_id"] = case.case_id
        trajectory["queries"] = query_rows
        case_rows.append(trajectory)

        stage_count = int(trajectory["stage_count"])
        total_stages += stage_count
        weighted_active_recall += (
            float(trajectory["mean_active_required_recall"]) * stage_count
        )
        weighted_shelf_recall += (
            float(trajectory["mean_shelf_required_recall"]) * stage_count
        )
        total_transition_skills += int(trajectory["transition_skill_count"])
        total_reused += int(trajectory["shelf_reuse_skill_count"])
        total_new_bundle += int(trajectory["new_bundle_skill_count"])
        total_future_skills += int(trajectory["future_transition_skill_count"])
        total_initial_skill_hits += len(
            trajectory["initial_shelf_future_skill_hits"]
        )
        total_future_bundles += int(
            trajectory["future_transition_bundle_count"]
        )
        total_initial_bundle_hits += len(
            trajectory["initial_shelf_future_bundle_hits"]
        )

    stage_denominator = total_stages or 1
    transition_denominator = total_transition_skills or 1
    future_skill_denominator = total_future_skills or 1
    future_bundle_denominator = total_future_bundles or 1

    return {
        "case_count": len(cases),
        "stage_count": total_stages,
        "k_per_retriever": k,
        "max_bundles": max_bundles,
        "max_skills_per_bundle": max_skills_per_bundle,
        "mean_active_required_recall": (
            weighted_active_recall / stage_denominator
        ),
        "mean_shelf_required_recall": (
            weighted_shelf_recall / stage_denominator
        ),
        "shelf_reuse_rate": total_reused / transition_denominator,
        "new_bundle_rate": total_new_bundle / transition_denominator,
        "transition_skill_count": total_transition_skills,
        "initial_shelf_future_skill_recall": (
            total_initial_skill_hits / future_skill_denominator
        ),
        "initial_shelf_future_skill_hits": total_initial_skill_hits,
        "future_transition_skill_count": total_future_skills,
        "initial_shelf_future_bundle_recall": (
            total_initial_bundle_hits / future_bundle_denominator
        ),
        "initial_shelf_future_bundle_hits": total_initial_bundle_hits,
        "future_transition_bundle_count": total_future_bundles,
        "cases": case_rows,
    }
