from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill_control_plane.retrieval import candidate_union


@dataclass(frozen=True, slots=True)
class MultiSkillGoldCase:
    case_id: str
    initial_task: str
    required: tuple[str, ...]
    useful: tuple[str, ...]
    hard_negative: tuple[str, ...]
    rationale: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class StageGold:
    stage_id: str
    observed_state: str
    transition_trigger: str | None
    required_now: tuple[str, ...]
    new_required: tuple[str, ...]
    useful: tuple[str, ...]
    hard_negative: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageTransitionGoldCase:
    case_id: str
    initial_task: str
    stages: tuple[StageGold, ...]
    rationale: str
    snapshot_id: str


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"gold line {line_number} must be a JSON object")
        rows.append(payload)
    if not rows:
        raise ValueError("gold set is empty")
    return rows


def _validate_case_ids(case_ids: list[str]) -> None:
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("gold set contains duplicate case_id values")


def _validate_snapshot_ids(snapshot_ids: set[str]) -> None:
    if len(snapshot_ids) != 1:
        raise ValueError("gold set must reference exactly one snapshot_id")


def load_multi_skill_gold(path: str | Path) -> list[MultiSkillGoldCase]:
    cases: list[MultiSkillGoldCase] = []
    for line_number, payload in enumerate(_read_jsonl(path), start=1):
        if payload.get("type") != "multi_skill":
            raise ValueError(f"gold line {line_number} is not type=multi_skill")
        required = tuple(payload.get("required", ()))
        if len(required) < 2:
            raise ValueError(
                f"multi-skill gold line {line_number} must require at least two Skills"
            )
        cases.append(
            MultiSkillGoldCase(
                case_id=str(payload["case_id"]),
                initial_task=str(payload["initial_task"]),
                required=required,
                useful=tuple(payload.get("useful", ())),
                hard_negative=tuple(payload.get("hard_negative", ())),
                rationale=str(payload.get("rationale", "")),
                snapshot_id=str(payload["snapshot_id"]),
            )
        )

    _validate_case_ids([case.case_id for case in cases])
    _validate_snapshot_ids({case.snapshot_id for case in cases})
    return cases


def load_stage_transition_gold(path: str | Path) -> list[StageTransitionGoldCase]:
    cases: list[StageTransitionGoldCase] = []
    for line_number, payload in enumerate(_read_jsonl(path), start=1):
        if payload.get("type") != "stage_transition":
            raise ValueError(f"gold line {line_number} is not type=stage_transition")

        raw_stages = payload.get("stages", ())
        if not isinstance(raw_stages, list) or len(raw_stages) < 2:
            raise ValueError(
                f"stage-transition gold line {line_number} must have at least two stages"
            )

        stages: list[StageGold] = []
        for stage_number, raw_stage in enumerate(raw_stages, start=1):
            required_now = tuple(raw_stage.get("required_now", ()))
            new_required = tuple(raw_stage.get("new_required", ()))
            if not required_now:
                raise ValueError(
                    f"gold line {line_number} stage {stage_number} has no required_now"
                )
            if not set(new_required).issubset(required_now):
                raise ValueError(
                    f"gold line {line_number} stage {stage_number}: "
                    "new_required must be a subset of required_now"
                )
            stages.append(
                StageGold(
                    stage_id=str(raw_stage["stage_id"]),
                    observed_state=str(raw_stage["observed_state"]),
                    transition_trigger=(
                        str(raw_stage["transition_trigger"])
                        if raw_stage.get("transition_trigger")
                        else None
                    ),
                    required_now=required_now,
                    new_required=new_required,
                    useful=tuple(raw_stage.get("useful", ())),
                    hard_negative=tuple(raw_stage.get("hard_negative", ())),
                )
            )

        stage_ids = [stage.stage_id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(
                f"stage-transition gold line {line_number} has duplicate stage_id values"
            )

        cases.append(
            StageTransitionGoldCase(
                case_id=str(payload["case_id"]),
                initial_task=str(payload["initial_task"]),
                stages=tuple(stages),
                rationale=str(payload.get("rationale", "")),
                snapshot_id=str(payload["snapshot_id"]),
            )
        )

    _validate_case_ids([case.case_id for case in cases])
    _validate_snapshot_ids({case.snapshot_id for case in cases})
    return cases


def stage_retrieval_query(case: StageTransitionGoldCase, stage: StageGold) -> str:
    parts = [case.initial_task]
    if stage.transition_trigger:
        parts.append(stage.transition_trigger)
    parts.append(stage.observed_state)
    return "\n".join(parts)


def _union_ids(query: str, *, bm25: Any, dense: Any, k: int) -> tuple[list[str], set[str]]:
    bm25_candidates = bm25.search(query, k=k)
    dense_candidates = dense.search(query, k=k)
    union_candidates = candidate_union(
        {
            "bm25": bm25_candidates,
            "dense": dense_candidates,
        }
    )
    ordered_ids = [candidate.skill_id for candidate in union_candidates]
    return ordered_ids, set(ordered_ids)


def evaluate_control_plane(
    multi_skill_cases: list[MultiSkillGoldCase],
    stage_transition_cases: list[StageTransitionGoldCase],
    *,
    bm25: Any,
    dense: Any,
    k: int,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")

    multi_rows: list[dict[str, Any]] = []
    multi_required_total = 0
    multi_required_hits = 0
    multi_full_coverage = 0
    multi_candidate_total = 0

    for case in multi_skill_cases:
        candidate_ids, candidate_set = _union_ids(
            case.initial_task,
            bm25=bm25,
            dense=dense,
            k=k,
        )
        required = set(case.required)
        hits = required & candidate_set
        missing = required - candidate_set

        multi_required_total += len(required)
        multi_required_hits += len(hits)
        multi_candidate_total += len(candidate_set)
        if not missing:
            multi_full_coverage += 1

        multi_rows.append(
            {
                "case_id": case.case_id,
                "required": list(case.required),
                "required_hits": sorted(hits),
                "missing_required": sorted(missing),
                "candidate_set_size": len(candidate_set),
                "candidate_ids": candidate_ids,
            }
        )

    stage_rows: list[dict[str, Any]] = []
    stage_total = 0
    one_shot_full_coverage = 0
    reroute_full_coverage = 0
    new_required_total = 0
    new_required_hits = 0
    reroute_candidate_total = 0

    for case in stage_transition_cases:
        one_shot_ids, one_shot_set = _union_ids(
            case.initial_task,
            bm25=bm25,
            dense=dense,
            k=k,
        )

        for stage_index, stage in enumerate(case.stages):
            query = stage_retrieval_query(case, stage)
            reroute_ids, reroute_set = _union_ids(
                query,
                bm25=bm25,
                dense=dense,
                k=k,
            )
            required = set(stage.required_now)
            one_shot_missing = required - one_shot_set
            reroute_missing = required - reroute_set

            stage_total += 1
            reroute_candidate_total += len(reroute_set)
            if not one_shot_missing:
                one_shot_full_coverage += 1
            if not reroute_missing:
                reroute_full_coverage += 1

            # S1 is initialization, not recovery. Recovery measures only skills
            # introduced by a transition after the initial stage.
            recovery_target = set(stage.new_required) if stage_index > 0 else set()
            recovery_hits = recovery_target & reroute_set
            new_required_total += len(recovery_target)
            new_required_hits += len(recovery_hits)

            stage_rows.append(
                {
                    "case_id": case.case_id,
                    "stage_id": stage.stage_id,
                    "query": query,
                    "required_now": list(stage.required_now),
                    "new_required": list(stage.new_required),
                    "one_shot_required_hits": sorted(required & one_shot_set),
                    "one_shot_missing_required": sorted(one_shot_missing),
                    "reroute_required_hits": sorted(required & reroute_set),
                    "reroute_missing_required": sorted(reroute_missing),
                    "new_required_recovered": sorted(recovery_hits),
                    "new_required_missing": sorted(recovery_target - reroute_set),
                    "one_shot_candidate_ids": one_shot_ids,
                    "reroute_candidate_ids": reroute_ids,
                    "reroute_candidate_set_size": len(reroute_set),
                }
            )

    multi_case_count = len(multi_skill_cases)
    stage_case_count = len(stage_transition_cases)
    one_shot_rate = (
        one_shot_full_coverage / stage_total if stage_total else 1.0
    )
    reroute_rate = (
        reroute_full_coverage / stage_total if stage_total else 1.0
    )

    return {
        "k_per_retriever": k,
        "multi_skill": {
            "case_count": multi_case_count,
            "required_skill_count": multi_required_total,
            "required_skill_hits": multi_required_hits,
            "required_skill_recall": (
                multi_required_hits / multi_required_total
                if multi_required_total
                else 1.0
            ),
            "full_required_set_coverage": (
                multi_full_coverage / multi_case_count
                if multi_case_count
                else 1.0
            ),
            "average_candidate_set_size": (
                multi_candidate_total / multi_case_count
                if multi_case_count
                else 0.0
            ),
            "cases": multi_rows,
        },
        "stage_transition": {
            "case_count": stage_case_count,
            "stage_count": stage_total,
            "one_shot_stage_full_coverage": one_shot_rate,
            "reroute_stage_full_coverage": reroute_rate,
            "reroute_gain": reroute_rate - one_shot_rate,
            "new_skill_recovery": (
                new_required_hits / new_required_total
                if new_required_total
                else 1.0
            ),
            "new_required_skill_count": new_required_total,
            "new_required_skill_hits": new_required_hits,
            "average_reroute_candidate_set_size": (
                reroute_candidate_total / stage_total
                if stage_total
                else 0.0
            ),
            "stages": stage_rows,
        },
        "activation_metrics_available": False,
        "activation_metrics_note": (
            "Premature activation and active-Skill precision require a real "
            "runtime judge/activation policy. Candidate presence is not activation."
        ),
    }
