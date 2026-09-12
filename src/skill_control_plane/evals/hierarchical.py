"""Paired evaluation of global-only and Shelf-aware evidence retrieval."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from skill_control_plane.evals.bundles import evaluate_bundle_trajectory
from skill_control_plane.evals.control_plane import StageTransitionGoldCase
from skill_control_plane.models import SkillRecord
from skill_control_plane.evals.legacy.stage_bundles import (
    CapabilityShelf, build_capability_shelf, integrate_retrieval_delta,
    default_group_key,
)
from skill_control_plane.evals.legacy.hierarchical import ShelfAwareRetriever, RetrieverFactory
from skill_control_plane.discovery.base import Retriever
from skill_control_plane.evals.legacy.capability_need import CapabilityNeedExtractor, extract_query


def evaluate_hierarchical_cases(
    cases: Sequence[StageTransitionGoldCase], records: Mapping[str, SkillRecord],
    *, bm25: Retriever, dense: Retriever, dense_factory: RetrieverFactory,
    k: int = 5, max_bundles: int = 4, max_skills_per_bundle: int = 4,
    min_score: float = 0.35, min_margin: float = 0.05,
    capability_need_extractor: CapabilityNeedExtractor | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "k_per_retriever": k, "max_bundles": max_bundles,
        "max_skills_per_bundle": max_skills_per_bundle,
        "match_min_score": min_score, "match_min_margin": min_margin,
        "rate_denominator": "evidence transitions only (S1 excluded)",
        "wrong_match_definition": "matched group contains none of new_required (required_now when new_required is empty); Gold used for scoring only",
        "corpus_size_definition": "unique Skill records searched per transition; excludes Bundle descriptor matching (reported separately)",
    }
    policies = ("global-only", "hierarchical")
    if capability_need_extractor is not None:
        policies = ("R0", "R1")
        result.update({
            "comparison": "hierarchical raw evidence vs hierarchical capability need",
            "recall_denominator": "evidence transitions only (S1 excluded)",
            "extraction_min_confidence": 0.5,
            "extraction_fallback": "global retrieval on raw evidence; no Bundle matching",
        })
    for policy in policies:
        rows = []
        transition_rows = []
        all_stages = []
        for case in cases:
            router = ShelfAwareRetriever(
                records, bm25=bm25, dense=dense, dense_factory=dense_factory,
                min_score=min_score, min_margin=min_margin,
            )
            shelf = CapabilityShelf()
            candidates_by_stage = {}
            traces = []
            for index, stage in enumerate(case.stages):
                query = case.initial_task if index == 0 else "\n".join(stage.runtime_evidence)
                before_registered = len(shelf.registered_skill_ids)
                before_active = sum(len(b.skill_ids) for b in shelf.bundles
                                    if b.bundle_id in shelf.active_bundle_ids)
                extraction_trace = {}
                allow_bundle_match = policy != "global-only"
                if index and capability_need_extractor is not None:
                    extraction_trace = {
                        "raw_evidence": list(stage.runtime_evidence),
                        "capability_need": None, "extractor_confidence": None,
                        "evidence_basis": "", "extraction_status": "not-used",
                        "before_bundle_ids": [b.bundle_id for b in shelf.bundles],
                    }
                    if policy == "R1":
                        extracted = extract_query(capability_need_extractor, case.initial_task,
                                                  stage.runtime_evidence)
                        query = extracted.query
                        allow_bundle_match = extracted.allow_bundle_match
                        extraction_trace.update({
                            "capability_need": extracted.need.capability_need if extracted.need else None,
                            "extractor_confidence": extracted.need.confidence if extracted.need else None,
                            "evidence_basis": extracted.need.evidence_basis if extracted.need else "",
                            "extraction_status": extracted.status,
                        })
                route = router.search(query, shelf, k=k, hierarchical=allow_bundle_match)
                candidates_by_stage[stage.stage_id] = route.candidates
                if index == 0:
                    shelf = build_capability_shelf(
                        route.candidates, records, max_bundles=max_bundles,
                        max_skills_per_bundle=max_skills_per_bundle,
                    )
                else:
                    shelf = integrate_retrieval_delta(
                        shelf, route.candidates, records, max_new_bundles=max_bundles,
                        max_skills_per_bundle=max_skills_per_bundle,
                    )
                required = set(stage.required_now)
                candidate_ids = {c.skill_id for c in route.candidates}
                active_ids = {s for b in shelf.bundles if b.bundle_id in shelf.active_bundle_ids
                              for s in b.skill_ids}
                trace = {
                    "stage_id": stage.stage_id, "query": query,
                    "route": route.route, "search_corpus_size": route.corpus_size,
                    "matcher_corpus_size": route.matcher_corpus_size,
                    "matched_bundle_id": route.matched_bundle_id,
                    "match_score": route.match_score, "match_margin": route.match_margin,
                    "wrong_bundle_match": route.matched_bundle_id is not None and
                        route.matched_bundle_id not in {default_group_key(records[s]) for s in (stage.new_required or stage.required_now)},
                    "candidate_ids": [c.skill_id for c in route.candidates],
                    "required_skill_recall": len(required & candidate_ids) / len(required) if required else 1.0,
                    "missing_shelf_ids": sorted(required - set(shelf.registered_skill_ids)),
                    "missing_active_ids": sorted(required - active_ids),
                    "active_skill_count": len(active_ids),
                    "registered_surface_growth": len(shelf.registered_skill_ids) - before_registered,
                    "active_surface_growth": len(active_ids) - before_active,
                }
                trace.update(extraction_trace)
                if index and capability_need_extractor is not None:
                    trace["missing_candidate_ids"] = sorted(required - candidate_ids)
                traces.append(trace)
                if index:
                    transition_rows.append(trace)
            trajectory = evaluate_bundle_trajectory(
                case.stages, candidates_by_stage, records, max_bundles=max_bundles,
                max_skills_per_bundle=max_skills_per_bundle,
            )
            for row, trace in zip(trajectory["stages"], traces, strict=True):
                row.update(trace)
                all_stages.append(row)
            trajectory["case_id"] = case.case_id
            rows.append(trajectory)
        def mean(items: list[dict[str, Any]], key: str) -> float:
            return sum(item[key] for item in items) / len(items) if items else 0.0
        local = [r for r in transition_rows if r["route"] == "bundle-local"]
        recall_rows = ([s for c in rows for s in c["stages"][1:]]
                       if capability_need_extractor is not None else all_stages)
        result[policy] = {
            "case_count": len(rows), "stage_count": len(all_stages),
            "transition_count": len(transition_rows),
            "required_skill_recall": mean(recall_rows, "required_skill_recall"),
            "mean_active_required_recall": mean(recall_rows, "active_required_recall"),
            "mean_shelf_required_recall": mean(recall_rows, "shelf_required_recall"),
            "global_retrieval_rate": sum(r["route"] == "global" for r in transition_rows) / len(transition_rows) if transition_rows else 0.0,
            "bundle_local_retrieval_rate": len(local) / len(transition_rows) if transition_rows else 0.0,
            "mean_search_corpus_size": mean(transition_rows, "search_corpus_size"),
            "mean_matcher_corpus_size": mean(transition_rows, "matcher_corpus_size"),
            "wrong_bundle_match_rate": mean(local, "wrong_bundle_match"),
            "wrong_bundle_match_count": sum(r["wrong_bundle_match"] for r in local),
            "bundle_match_count": len(local),
            "mean_registered_surface_growth": mean(transition_rows, "registered_surface_growth"),
            "mean_active_surface_growth": mean(transition_rows, "active_surface_growth"),
            "mean_final_registered_skill_count": sum(c["final_registered_skill_count"] for c in rows) / len(rows) if rows else 0.0,
            "mean_final_active_skill_count": sum(c["stages"][-1]["active_skill_count"] for c in rows) / len(rows) if rows else 0.0,
            "cases": rows,
        }
    return result


def evaluate_capability_need_cases(
    cases: Sequence[StageTransitionGoldCase], records: Mapping[str, SkillRecord],
    *, extractor: CapabilityNeedExtractor, bm25: Retriever, dense: Retriever,
    dense_factory: RetrieverFactory, k: int = 5, max_bundles: int = 4,
    max_skills_per_bundle: int = 4,
) -> dict[str, Any]:
    """R0/R1 share the trajectory engine; matcher defaults remain 0.35/0.05."""
    return evaluate_hierarchical_cases(
        cases, records, bm25=bm25, dense=dense, dense_factory=dense_factory,
        k=k, max_bundles=max_bundles, max_skills_per_bundle=max_skills_per_bundle,
        capability_need_extractor=extractor,
    )


def print_capability_need_report(report: Mapping[str, Any]) -> None:
    print("=== CAPABILITY NEED R0/R1 (runtime transitions only) ===")
    for policy in ("R0", "R1"):
        arm = report[policy]
        print(f"\n{policy}")
        for key, value in arm.items():
            if key != "cases":
                print(f"{key}: {value}")
        for case in arm["cases"]:
            for stage in case["stages"][1:]:
                print(f"\n{case['case_id']}/{stage['stage_id']}")
                for key in ("raw_evidence", "capability_need", "extractor_confidence",
                            "evidence_basis", "extraction_status", "query", "match_score",
                            "match_margin", "matched_bundle_id", "route", "search_corpus_size",
                            "matcher_corpus_size", "candidate_ids", "missing_candidate_ids",
                            "missing_shelf_ids", "missing_active_ids"):
                    print(f"  {key}: {stage[key]}")
