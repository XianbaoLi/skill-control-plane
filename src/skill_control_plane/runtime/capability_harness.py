"""Deprecated V0.x compatibility adapter.

New integrations must use :class:`SkillControlPlane`. This adapter preserves
historical experiment entry points, including the combined resolver loader, but
is not part of the supported Agent-facing API.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from typing import Mapping

from skill_control_plane.discovery import SkillDiscovery, SkillDiscoveryResult

from .capability_memory import CapabilityMemory, RuntimeCapabilityState, validate_state
from .control_plane import SkillControlPlane
from .discovery_session import CapabilityDecision, CoverageClaim, DiscoverySession


def _drop_none(value):
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_drop_none(item) for item in value]
    return value


class RuntimeCapabilityHarness:
    """Deprecated compatibility surface for V0.x experiments only."""

    def __init__(self, loader=None, state: RuntimeCapabilityState | None = None, *,
                 discovery: SkillDiscovery | None = None,
                 max_searches_per_turn: int = 3) -> None:
        warnings.warn(
            "RuntimeCapabilityHarness is deprecated; use SkillControlPlane",
            DeprecationWarning,
            stacklevel=2,
        )
        if (loader is None) == (discovery is None):
            raise ValueError("provide exactly one loader or discovery")
        self.loader = loader
        self.discovery = loader.discovery if loader is not None else discovery
        assert self.discovery is not None
        store = self.discovery.registry
        self.memory = CapabilityMemory(
            store,
            state,
            capability_phrases={
                skill.skill_id: self.discovery.capability_phrases(skill.skill_id)
                for skill in store
            },
            bundle_capability_phrases={
                skill.skill_id: self.discovery.bundle_capability_phrases(skill.skill_id)
                for skill in store
            },
        )
        self.discovery_session = DiscoverySession(
            self.discovery, self.memory, max_searches=max_searches_per_turn)
        self.control_plane = SkillControlPlane(
            store,
            discovery=self.discovery,
            memory=self.memory,
            discovery_session=self.discovery_session,
        )

    @property
    def state(self):
        return self.memory.state

    @state.setter
    def state(self, value) -> None:
        validate_state(value, self.memory.store)
        self.memory.state = value

    @property
    def pending_candidates(self):
        return self.discovery_session.pending_candidates

    @pending_candidates.setter
    def pending_candidates(self, value) -> None:
        self.discovery_session.pending_candidates = value

    @property
    def loaded_skill_ids(self) -> tuple[str, ...]:
        return self.memory.loaded_skill_ids

    @property
    def retrieval_call_count(self) -> int:
        return self.discovery_session.retrieval_call_count

    @property
    def body_load_count(self) -> int:
        return self.memory.body_load_count

    @property
    def max_searches_per_turn(self) -> int:
        return self.discovery_session.max_searches

    @property
    def turn_search_count(self) -> int:
        return self.discovery_session.search_count

    @property
    def turn_search_attempt_count(self) -> int:
        return self.discovery_session.search_attempt_count

    @property
    def turn_repeated_search_count(self) -> int:
        return self.discovery_session.repeated_search_count

    @property
    def turn_no_progress_count(self) -> int:
        return self.discovery_session.no_progress_count

    @property
    def turn_search_budget_hits(self) -> int:
        return self.discovery_session.search_budget_hits

    @property
    def turn_search_needs(self) -> list[str]:
        return self.discovery_session.search_needs

    @property
    def last_search_control(self) -> dict:
        control = self.discovery_session.last_search_control
        return _drop_none(asdict(control)) if control is not None else {}

    def render_bundle_context(self, *, compact: bool = True) -> str:
        snapshot = self.memory.snapshot(compact=compact)
        return json.dumps(
            {"maintained_bundles": _drop_none(asdict(snapshot))["maintained_bundles"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def render_context(self) -> str:
        """Return legacy state JSON without prompts or tool schemas."""

        snapshot = self.memory.snapshot()
        return json.dumps({
            "direct_skills": list(snapshot.direct_skill_ids),
            "maintained_bundles": [
                {
                    "bundle_id": bundle.bundle_id,
                    "purpose": bundle.purpose,
                    "skill_ids": [member.skill_id for member in bundle.members],
                    "members": [
                        {"skill_id": member.skill_id, "body_state": member.body_state}
                        for member in bundle.members
                    ],
                }
                for bundle in snapshot.maintained_bundles
            ],
        }, ensure_ascii=False, separators=(",", ":"))

    def search_capability(self, need: str, *, k: int = 10) -> SkillDiscoveryResult:
        return self.discovery_session.search(need, k=k)

    def begin_turn(self) -> None:
        self.discovery_session.begin_turn()

    def model_visible_candidates(self, result: SkillDiscoveryResult) -> dict:
        return {
            **self.discovery.model_visible_payload(result),
            "search_control": self.last_search_control,
        }

    def apply_capability(self, decision: CapabilityDecision | str | Mapping) -> dict:
        """Compatibility bridge; new code passes a CapabilityDecision to Core."""

        if isinstance(decision, str):
            decision = json.loads(decision)
        if isinstance(decision, Mapping):
            decision = self._legacy_decision(decision)
        result = self.discovery_session.apply(decision)
        return _drop_none(asdict(result))

    def load_skill_body(self, skill_id: str) -> dict:
        return _drop_none(asdict(self.memory.load_skill_body(skill_id)))

    def mark_skill_body_evicted(self, skill_id: str) -> None:
        self.memory.mark_skill_body_evicted(skill_id)

    def mark_all_skill_bodies_evicted(self) -> None:
        self.memory.mark_all_skill_bodies_evicted()

    def load_capability(self, need: str):
        """Historical combined resolver compatibility; absent from SkillControlPlane."""

        if self.loader is None:
            raise ValueError("combined loading requires a loader; use search_capability")
        self.pending_candidates = None
        result = self.loader.load_capability(need, self.state)
        self.state = result.resulting_state
        return result

    @staticmethod
    def _legacy_decision(data: Mapping) -> CapabilityDecision:
        action = data.get("action")
        fields = {
            "DIRECT": {"action", "skill_ids", "reason", "coverage", "remaining_gaps"},
            "EXTEND": {"action", "skill_ids", "reason", "coverage", "remaining_gaps",
                       "target_bundle_id"},
            "CREATE": {"action", "skill_ids", "reason", "coverage", "remaining_gaps",
                       "purpose"},
        }
        if action not in fields or set(data) != fields[action]:
            raise ValueError("invalid action or action-specific fields")
        coverage = data.get("coverage")
        gaps = data.get("remaining_gaps")
        if not isinstance(coverage, list) or not isinstance(gaps, list):
            raise ValueError("coverage and remaining_gaps must be arrays")
        claims = []
        for item in coverage:
            if not isinstance(item, Mapping) or set(item) != {"need", "covered_by"}:
                raise ValueError("coverage items require only need and covered_by")
            claims.append(CoverageClaim(item["need"], item["covered_by"]))
        ids = data.get("skill_ids")
        if not isinstance(ids, list):
            raise ValueError("skill_ids must be an array")
        return CapabilityDecision(
            action=action,
            skill_ids=tuple(ids),
            reason=data.get("reason"),
            target_bundle_id=data.get("target_bundle_id"),
            purpose=data.get("purpose"),
            coverage=tuple(claims),
            remaining_gaps=tuple(gaps),
        )
