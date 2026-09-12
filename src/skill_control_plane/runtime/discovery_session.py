"""Per-user-turn Candidate Closure and Capability Sufficiency state."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from skill_control_plane.discovery import SkillDiscovery, SkillDiscoveryResult
from .capability_memory import CapabilityMemory, SkillBody


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


@dataclass(frozen=True, slots=True)
class SearchControl:
    search_count: int
    remaining_search_budget: int
    new_candidate_skill_ids: tuple[str, ...]
    meaningful_expansion: bool
    repeated_query: bool
    message: str | None


@dataclass(frozen=True, slots=True)
class CapabilityApplication:
    action: Literal["DIRECT", "EXTEND", "CREATE"]
    affected_bundle_id: str | None
    selected_skill_ids: tuple[str, ...]
    coverage: tuple[CoverageClaim, ...]
    remaining_gaps: tuple[str, ...]
    skill_bodies: tuple[SkillBody, ...]


@dataclass(frozen=True, slots=True)
class TurnAudit:
    search_count: int
    search_attempt_count: int
    repeated_search_count: int
    no_progress_search_count: int
    search_budget_hits: int
    search_needs: tuple[str, ...]
    apply_count: int
    capability_sufficiency_outcome: Literal["COVERED", "UNSATISFIED"]
    sufficiency_transitions: tuple[str, ...]
    unresolved_gaps: tuple[str, ...]


def validate_decision(
    decision: CapabilityDecision,
    skills: SkillDiscoveryResult,
    memory: CapabilityMemory,
    *,
    require_coverage: bool = True,
) -> CapabilityDecision:
    if not isinstance(decision, CapabilityDecision):
        raise TypeError("decision must be a CapabilityDecision")
    action = decision.action
    if action not in {"DIRECT", "EXTEND", "CREATE"}:
        raise ValueError("invalid capability action")
    ids = decision.skill_ids
    if not isinstance(ids, tuple) or not ids or any(
        not isinstance(skill_id, str) or not skill_id.strip() for skill_id in ids
    ):
        raise ValueError("skill_ids must be a non-empty tuple of strings")
    candidates = {candidate.skill_id for candidate in skills.candidates}
    if set(ids) - candidates:
        raise ValueError("skill_id outside supplied candidates")
    if not isinstance(decision.reason, str) or not decision.reason.strip():
        raise ValueError("reason must be non-empty text")
    target = decision.target_bundle_id
    purpose = decision.purpose
    if action == "DIRECT" and (target is not None or purpose is not None):
        raise ValueError("DIRECT does not accept target_bundle_id or purpose")
    if action == "EXTEND":
        if not isinstance(target, str) or purpose is not None:
            raise ValueError("target_bundle_id must be a string")
        existing = next((bundle for bundle in memory.state.active_bundles
                         if bundle.bundle_id == target), None)
        if existing is None:
            raise ValueError("target_bundle_id is not active")
        if not set(ids) - set(existing.skill_ids):
            raise ValueError("EXTEND requires at least one new skill")
    if action == "CREATE":
        if target is not None or not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("CREATE requires non-empty purpose")

    coverage: list[CoverageClaim] = []
    remaining_gaps: tuple[str, ...] = ()
    if require_coverage or decision.coverage or decision.remaining_gaps:
        raw_coverage = decision.coverage
        if not isinstance(raw_coverage, tuple):
            raise ValueError("coverage must be a tuple")
        bundle_ids = {bundle.bundle_id for bundle in memory.state.active_bundles}
        selected = set(ids)
        covered_selected: set[str] = set()
        for claim in raw_coverage:
            if not isinstance(claim, CoverageClaim):
                raise ValueError("coverage items must be CoverageClaim values")
            need, covered_by = claim.need, claim.covered_by
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
        raw_gaps = decision.remaining_gaps
        if not isinstance(raw_gaps, tuple) or any(
            not isinstance(gap, str) or not gap.strip() for gap in raw_gaps
        ):
            raise ValueError("remaining_gaps must be a tuple of non-empty strings")
        remaining_gaps = tuple(dict.fromkeys(gap.strip() for gap in raw_gaps))
    return CapabilityDecision(
        action, tuple(dict.fromkeys(ids)), decision.reason.strip(), target,
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
        self.last_search_control: SearchControl | None = None
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
        self.last_search_control = SearchControl(
            search_count=self.search_count,
            remaining_search_budget=self.max_searches - self.search_count,
            new_candidate_skill_ids=tuple(new_ids),
            meaningful_expansion=not no_progress,
            repeated_query=repeated_query,
            message=None if not no_progress else (
                "No meaningful new candidates were discovered. Do not repeat the "
                "same search. Use current candidates or leave the gap unresolved."
            ),
        )
        self.sufficiency_transitions.append("SEARCH_MORE")
        self.remaining_gaps = [need]
        return result

    def apply(self, decision: CapabilityDecision) -> CapabilityApplication:
        if self.pending_candidates is None:
            raise ValueError("apply_capability requires a fresh load_capability result")
        decision = validate_decision(decision, self.pending_candidates, self.memory)
        target, bodies = self.memory.commit(
            action=decision.action,
            skill_ids=decision.skill_ids,
            target_bundle_id=decision.target_bundle_id,
            purpose=decision.purpose,
        )
        result = CapabilityApplication(
            action=decision.action,
            affected_bundle_id=target,
            selected_skill_ids=decision.skill_ids,
            coverage=decision.coverage,
            remaining_gaps=decision.remaining_gaps,
            skill_bodies=bodies,
        )
        self.pending_candidates = None
        self.apply_count += 1
        outcome = "UNSATISFIED" if decision.remaining_gaps else "COVERED"
        self.sufficiency_transitions.append(outcome)
        self.remaining_gaps = list(decision.remaining_gaps)
        return result

    @property
    def sufficiency(self) -> str:
        return "UNSATISFIED" if self.remaining_gaps else "COVERED"

    def audit(self) -> TurnAudit:
        return TurnAudit(
            search_count=self.search_count,
            search_attempt_count=self.search_attempt_count,
            repeated_search_count=self.repeated_search_count,
            no_progress_search_count=self.no_progress_count,
            search_budget_hits=self.search_budget_hits,
            search_needs=tuple(self.search_needs),
            apply_count=self.apply_count,
            capability_sufficiency_outcome=self.sufficiency,
            sufficiency_transitions=tuple(self.sufficiency_transitions),
            unresolved_gaps=tuple(self.remaining_gaps),
        )
