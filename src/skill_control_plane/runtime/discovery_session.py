"""Per-user-turn Candidate Closure and Capability Sufficiency state."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from skill_control_plane.discovery import SkillDiscovery, SkillDiscoveryResult
from .capability_memory import CapabilityMemory


@dataclass(frozen=True)
class CoverageClaim:
    need: str
    covered_by: str


@dataclass(frozen=True)
class CapabilityDecision:
    action: Literal["DIRECT", "EXTEND", "CREATE"]
    skill_ids: tuple[str, ...]
    reason: str
    target_bundle_id: str | None = None
    purpose: str | None = None
    coverage: tuple[CoverageClaim, ...] = ()
    remaining_gaps: tuple[str, ...] = ()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def validate_decision(
    raw: str,
    skills: SkillDiscoveryResult,
    memory: CapabilityMemory,
    *,
    require_coverage: bool = True,
) -> CapabilityDecision:
    data = json.loads(raw, object_pairs_hook=_strict_object)
    if not isinstance(data, dict) or not isinstance(data.get("action"), str):
        raise ValueError("decision must be an object with action")
    action = data["action"]
    fields = {
        "DIRECT": {"action", "skill_ids", "reason"},
        "EXTEND": {"action", "skill_ids", "reason", "target_bundle_id"},
        "CREATE": {"action", "skill_ids", "reason", "purpose"},
    }
    coverage_fields = {"coverage", "remaining_gaps"}
    extended = require_coverage or bool(set(data) & coverage_fields)
    expected = fields.get(action, set()) | (coverage_fields if extended else set())
    if action not in fields or set(data) != expected:
        raise ValueError("invalid action or action-specific fields")
    ids = data["skill_ids"]
    if not isinstance(ids, list) or not ids or any(
        not isinstance(skill_id, str) or not skill_id.strip() for skill_id in ids
    ):
        raise ValueError("skill_ids must be a non-empty array of strings")
    candidates = {candidate.skill_id for candidate in skills.candidates}
    if set(ids) - candidates:
        raise ValueError("skill_id outside supplied candidates")
    if not isinstance(data["reason"], str) or not data["reason"].strip():
        raise ValueError("reason must be non-empty text")
    target = data.get("target_bundle_id")
    purpose = data.get("purpose")
    if action == "EXTEND":
        if not isinstance(target, str):
            raise ValueError("target_bundle_id must be a string")
        existing = next((bundle for bundle in memory.state.active_bundles
                         if bundle.bundle_id == target), None)
        if existing is None:
            raise ValueError("target_bundle_id is not active")
        if not set(ids) - set(existing.skill_ids):
            raise ValueError("EXTEND requires at least one new skill")
    if action == "CREATE" and (not isinstance(purpose, str) or not purpose.strip()):
        raise ValueError("CREATE requires non-empty purpose")

    coverage: list[CoverageClaim] = []
    remaining_gaps: tuple[str, ...] = ()
    if extended:
        raw_coverage = data["coverage"]
        if not isinstance(raw_coverage, list):
            raise ValueError("coverage must be an array")
        bundle_ids = {bundle.bundle_id for bundle in memory.state.active_bundles}
        selected = set(ids)
        covered_selected: set[str] = set()
        for claim in raw_coverage:
            if not isinstance(claim, dict) or set(claim) != {"need", "covered_by"}:
                raise ValueError("coverage items require only need and covered_by")
            need, covered_by = claim["need"], claim["covered_by"]
            if (not isinstance(need, str) or not need.strip()
                    or not isinstance(covered_by, str) or not covered_by.strip()):
                raise ValueError("coverage need and covered_by must be non-empty text")
            if covered_by.startswith("bundle:"):
                if covered_by.removeprefix("bundle:") not in bundle_ids:
                    raise ValueError("coverage references a non-current Bundle")
            elif covered_by.startswith("skill:"):
                covered_skill = covered_by.removeprefix("skill:")
                if covered_skill not in candidates:
                    raise ValueError("coverage references a Skill outside pending candidates")
                if covered_skill not in selected:
                    raise ValueError("coverage Skill must be selected for commitment")
                covered_selected.add(covered_skill)
            else:
                raise ValueError("covered_by must use bundle: or skill:")
            coverage.append(CoverageClaim(need.strip(), covered_by))
        if selected - covered_selected:
            raise ValueError("every selected Skill requires a coverage claim")
        raw_gaps = data["remaining_gaps"]
        if not isinstance(raw_gaps, list) or any(
            not isinstance(gap, str) or not gap.strip() for gap in raw_gaps
        ):
            raise ValueError("remaining_gaps must be an array of non-empty strings")
        remaining_gaps = tuple(dict.fromkeys(gap.strip() for gap in raw_gaps))
    return CapabilityDecision(
        action, tuple(dict.fromkeys(ids)), data["reason"].strip(), target,
        purpose.strip() if purpose is not None else None,
        tuple(coverage), remaining_gaps,
    )


class DiscoverySession:
    """All mutable discovery state for one user turn."""

    def __init__(self, discovery: SkillDiscovery, memory: CapabilityMemory, *,
                 max_searches: int = 3) -> None:
        if max_searches < 1:
            raise ValueError("max_searches must be positive")
        self.discovery = discovery
        self.memory = memory
        self.max_searches = max_searches
        self.retrieval_call_count = 0
        self.begin_turn()

    def begin_turn(self) -> None:
        self.pending_candidates: SkillDiscoveryResult | None = None
        self.search_count = 0
        self.search_attempt_count = 0
        self.repeated_search_count = 0
        self.no_progress_count = 0
        self.search_budget_hits = 0
        self.search_needs: list[str] = []
        self.last_search_control: dict = {}
        self.sufficiency_transitions: list[str] = []
        self.remaining_gaps: list[str] = []
        self.apply_count = 0

    def search(self, need: str, *, k: int = 10) -> SkillDiscoveryResult:
        self.search_attempt_count += 1
        if self.search_count >= self.max_searches:
            self.search_budget_hits += 1
            raise ValueError(
                "Per-turn load_capability budget exhausted. Do not search again; "
                "use current candidates or leave remaining gaps unresolved."
            )
        previous = self.pending_candidates
        previous_ids = ({candidate.skill_id for candidate in previous.candidates}
                        if previous is not None else set())
        previous_query = previous.query.casefold().strip() if previous is not None else None
        had_previous_search = self.search_count > 0
        self.search_count += 1
        self.search_needs.append(need)
        self.retrieval_call_count += 1
        result = self.discovery.discover_skills(need, k=k)
        if previous is None:
            self.pending_candidates = result
        else:
            merged = {candidate.skill_id: candidate for candidate in previous.candidates}
            merged.update((candidate.skill_id, candidate) for candidate in result.candidates)
            representations = dict(previous.representations)
            representations.update(result.representations)
            self.pending_candidates = replace(
                result, candidates=tuple(merged.values()),
                representations=tuple(representations.items()),
            )
        current_ids = {candidate.skill_id for candidate in self.pending_candidates.candidates}
        new_ids = sorted(current_ids - previous_ids)
        no_progress = not new_ids
        repeated_query = previous_query == need.casefold().strip()
        if no_progress:
            self.no_progress_count += 1
        if repeated_query or (had_previous_search and no_progress):
            self.repeated_search_count += 1
        self.last_search_control = {
            "search_count": self.search_count,
            "remaining_search_budget": self.max_searches - self.search_count,
            "new_candidate_skill_ids": new_ids,
            "meaningful_expansion": not no_progress,
            "repeated_query": repeated_query,
            "message": None if not no_progress else (
                "No meaningful new candidates were discovered. Do not repeat the "
                "same search. Use current candidates or leave the gap unresolved."
            ),
        }
        self.sufficiency_transitions.append("SEARCH_MORE")
        self.remaining_gaps = [need]
        return result

    def model_visible_candidates(self, result: SkillDiscoveryResult) -> dict:
        return {**self.discovery.model_visible_payload(result),
                "search_control": dict(self.last_search_control)}

    def apply(self, raw_decision: str) -> dict:
        if self.pending_candidates is None:
            raise ValueError("apply_capability requires a fresh load_capability result")
        decision = validate_decision(raw_decision, self.pending_candidates, self.memory)
        target, bodies = self.memory.commit(
            action=decision.action,
            skill_ids=decision.skill_ids,
            target_bundle_id=decision.target_bundle_id,
            purpose=decision.purpose,
        )
        result = {
            "action": decision.action,
            "affected_bundle_id": target,
            "selected_skill_ids": list(decision.skill_ids),
            "coverage": [asdict(claim) for claim in decision.coverage],
            "remaining_gaps": list(decision.remaining_gaps),
            "skill_bodies": bodies,
        }
        self.pending_candidates = None
        self.apply_count += 1
        outcome = "UNSATISFIED" if decision.remaining_gaps else "COVERED"
        self.sufficiency_transitions.append(outcome)
        self.remaining_gaps = list(decision.remaining_gaps)
        return result

    @property
    def sufficiency(self) -> str:
        return "UNSATISFIED" if self.remaining_gaps else "COVERED"

    def audit(self) -> dict:
        return {
            "search_count": self.search_count,
            "search_attempt_count": self.search_attempt_count,
            "repeated_search_count": self.repeated_search_count,
            "no_progress_search_count": self.no_progress_count,
            "search_budget_hits": self.search_budget_hits,
            "search_needs": list(self.search_needs),
            "apply_count": self.apply_count,
            "capability_sufficiency_outcome": self.sufficiency,
            "sufficiency_transitions": list(self.sufficiency_transitions),
            "unresolved_gaps": list(self.remaining_gaps),
        }
