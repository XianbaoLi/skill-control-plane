"""The single Agent-facing V1 facade over Skill Control Plane Core."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from skill_control_plane.discovery import (
    SkillDiscovery,
    SkillDiscoveryResult,
    load_retrieval_cards,
    validate_retrieval_cards,
)
from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.registry import SkillStore

from .capability_memory import (
    BundleSnapshot,
    CapabilityMemory,
    RuntimeCapabilityState,
    SkillBodyLoadResult,
)
from .discovery_session import (
    CapabilityApplication,
    CapabilityDecision,
    DiscoverySession,
    SearchControl,
    TurnAudit,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    skill_id: str
    name: str
    description: str
    rank: int
    minimal_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    query: str
    candidates: tuple[RetrievalCandidate, ...]
    representations: tuple[tuple[str, str], ...]
    backend: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class CapabilitySearchResult:
    query: str
    candidates: tuple[Candidate, ...]
    search_control: SearchControl
    retrieval_trace: RetrievalTrace


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    direct_skill_ids: tuple[str, ...]
    maintained_bundles: tuple[BundleSnapshot, ...]
    pending_query: str | None
    pending_candidate_skill_ids: tuple[str, ...]
    search_count: int
    remaining_search_budget: int
    capability_sufficiency_outcome: str

    @property
    def skill_body_states(self) -> dict[str, str]:
        return {
            member.skill_id: member.body_state
            for bundle in self.maintained_bundles
            for member in bundle.members
        }


@dataclass(frozen=True, slots=True)
class ControlPlaneTurnAudit:
    session: TurnAudit
    retrieval_call_count: int
    body_load_count: int


class ControlPlaneConfigurationError(ValueError):
    """The production runtime was not given complete Discovery configuration."""


class SkillControlPlane:
    """Stable structured API for Agent integrations.

    Provider messages, tool-call envelopes, prompts, schemas, and JSON text are
    intentionally outside this class. Components are private so integrations
    cannot couple to Store, Session, or Memory implementation state.
    """

    def __init__(
        self,
        store: SkillStore,
        *,
        discovery: SkillDiscovery | None = None,
        state: RuntimeCapabilityState | None = None,
        memory: CapabilityMemory | None = None,
        discovery_session: DiscoverySession | None = None,
        max_searches_per_turn: int = 3,
    ) -> None:
        if discovery is None:
            raise ControlPlaneConfigurationError(
                "SkillControlPlane requires configured SkillDiscovery with "
                "Retrieval Cards, Dense, and RRF")
        if discovery.registry is not store:
            raise ControlPlaneConfigurationError(
                "discovery must use the supplied SkillStore")
        try:
            validate_retrieval_cards(store, discovery.retrieval_cards)
        except ValueError as exc:
            raise ControlPlaneConfigurationError(
                f"invalid Retrieval Card configuration: {exc}") from exc
        if discovery.dense is None:
            raise ControlPlaneConfigurationError(
                "SkillControlPlane requires a configured Dense backend")
        if discovery.fusion_backend != "rrf":
            raise ControlPlaneConfigurationError(
                "SkillControlPlane requires the RRF fusion path")
        if memory is not None and state is not None:
            raise ValueError("provide state or memory, not both")
        if memory is None:
            memory = CapabilityMemory(
                store,
                state,
                capability_phrases={
                    skill.skill_id: discovery.capability_phrases(skill.skill_id)
                    for skill in store
                },
                bundle_capability_phrases={
                    skill.skill_id: discovery.bundle_capability_phrases(skill.skill_id)
                    for skill in store
                },
            )
        elif memory.store is not store:
            raise ValueError("memory must use the supplied SkillStore")
        if discovery_session is None:
            discovery_session = DiscoverySession(
                discovery, memory, max_searches=max_searches_per_turn)
        elif (discovery_session.discovery is not discovery
              or discovery_session.memory is not memory):
            raise ValueError("DiscoverySession must use the supplied components")
        self._store = store
        self._discovery = discovery
        self._memory = memory
        self._session = discovery_session

    @classmethod
    def from_tree(
        cls,
        skill_root,
        *,
        retrieval_cards,
        dense_factory,
        max_searches_per_turn: int = 3,
    ) -> "SkillControlPlane":
        """Build the production runtime from mandatory complete Discovery inputs."""

        if retrieval_cards is None:
            raise ControlPlaneConfigurationError(
                "SkillControlPlane.from_tree requires Retrieval Cards")
        if dense_factory is None:
            raise ControlPlaneConfigurationError(
                "SkillControlPlane.from_tree requires a Dense factory")
        store = SkillStore.from_tree(skill_root)
        cards = (load_retrieval_cards(retrieval_cards)
                 if isinstance(retrieval_cards, (str, Path))
                 else dict(retrieval_cards))
        discovery = SkillDiscovery(
            store, dense_factory=dense_factory, retrieval_cards=cards)
        return cls(
            store,
            discovery=discovery,
            max_searches_per_turn=max_searches_per_turn,
        )

    def begin_turn(self) -> None:
        self._session.begin_turn()

    def end_turn(self) -> None:
        """Discard uncommitted Candidate Closure state at a completed turn."""

        self._session.pending_candidates = None

    def context_snapshot(self, *, compact: bool = True) -> ContextSnapshot:
        memory = self._memory.snapshot(compact=compact)
        pending = self._session.pending_candidates
        return ContextSnapshot(
            direct_skill_ids=memory.direct_skill_ids,
            maintained_bundles=memory.maintained_bundles,
            pending_query=pending.query if pending is not None else None,
            pending_candidate_skill_ids=tuple(
                candidate.skill_id for candidate in pending.candidates
            ) if pending is not None else (),
            search_count=self._session.search_count,
            remaining_search_budget=(
                self._session.max_searches - self._session.search_count),
            capability_sufficiency_outcome=self._session.sufficiency,
        )

    def search_capability(self, need: str, *, k: int = 10) -> CapabilitySearchResult:
        result = self._session.search(need, k=k)
        control = self._session.last_search_control
        assert control is not None
        return CapabilitySearchResult(
            query=result.query,
            candidates=self._compact_candidates(result),
            search_control=control,
            retrieval_trace=RetrievalTrace(
                query=result.query,
                candidates=result.candidates,
                representations=result.representations,
                backend=result.backend,
                truncated=result.truncated,
            ),
        )

    def apply_capability(
        self, decision: CapabilityDecision,
    ) -> CapabilityApplication:
        return self._session.apply(decision)

    def load_skill_body(self, skill_id: str) -> SkillBodyLoadResult:
        return self._memory.load_skill_body(skill_id)

    def mark_skill_body_evicted(self, skill_id: str) -> None:
        self._memory.mark_skill_body_evicted(skill_id)

    def mark_all_skill_bodies_evicted(self) -> None:
        self._memory.mark_all_skill_bodies_evicted()

    def turn_audit(self) -> ControlPlaneTurnAudit:
        return ControlPlaneTurnAudit(
            session=self._session.audit(),
            retrieval_call_count=self._session.retrieval_call_count,
            body_load_count=self._memory.body_load_count,
        )

    def _compact_candidates(
        self, result: SkillDiscoveryResult,
    ) -> tuple[Candidate, ...]:
        compact: list[Candidate] = []
        for candidate in result.candidates:
            record = self._store.get(candidate.skill_id)
            evidence: dict[str, object] = {}
            matched = next((
                item.removeprefix("matched_terms: ").split(", ")
                for item in candidate.evidence
                if item.startswith("matched_terms: ")
            ), None)
            if matched:
                evidence["matched_terms"] = tuple(matched[:6])
            if "dense" in candidate.source_scores:
                evidence["semantic_similarity"] = round(
                    float(candidate.source_scores["dense"]), 4)
            compact.append(Candidate(
                skill_id=candidate.skill_id,
                name=record.name,
                description=record.description,
                rank=candidate.rank,
                minimal_evidence=evidence,
            ))
        return tuple(compact)
