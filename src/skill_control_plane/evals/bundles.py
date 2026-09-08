from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.control_plane import StageGold
from skill_control_plane.models import RetrievalCandidate, SkillRecord
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
    """Evaluate a sequential Capability Shelf over one stage trajectory.

    The caller supplies already-retrieved candidates for every stage. This keeps
    bundle evaluation independent from BM25/Dense implementation details and
    lets the same candidate traces compare multiple bundle strategies.
    """

    shelf = CapabilityShelf()
    rows: list[dict[str, Any]] = []
    active_recalls: list[float] = []
    shelf_recalls: list[float] = []
    transition_skill_count = 0
    shelf_reuse_skill_count = 0
    new_bundle_skill_count = 0

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
                "registered_skill_count": len(shelf.registered_skill_ids),
                "active_required_recall": active_recall,
                "shelf_required_recall": shelf_recall,
            }
        )

    denominator = transition_skill_count or 1
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
        "final_registered_skill_count": len(shelf.registered_skill_ids),
        "stages": rows,
    }
