"""Evidence-driven capability rerouting owned by the runtime."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from .capability_memory import BundleSnapshot


RuntimeEvidenceKind = Literal[
    "tool_error",
    "test_failure",
    "verifier_failure",
    "new_subgoal",
    "host_signal",
]
SUPPORTED_EVIDENCE_KINDS = frozenset({
    "tool_error",
    "test_failure",
    "verifier_failure",
    "new_subgoal",
    "host_signal",
})

INTERNAL_EVIDENCE_SOURCES = frozenset({
    "capability_gap_decision",
    "capability_gap_check",
    "discover_capability",
    "load_capability",
    "search_capability",
    "apply_capability",
    "load_skill_body",
    "telemetry",
})


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    evidence_id: str
    kind: RuntimeEvidenceKind
    source: str
    text: str
    fingerprint: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source", "text", "fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RuntimeEvidence {name} must be a non-empty string")
        if self.kind not in SUPPORTED_EVIDENCE_KINDS:
            raise ValueError(f"unsupported RuntimeEvidence kind: {self.kind}")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("RuntimeEvidence metadata must be a mapping")

    @classmethod
    def create(
        cls,
        *,
        evidence_id: str,
        kind: RuntimeEvidenceKind,
        source: str,
        text: str,
        fingerprint: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "RuntimeEvidence":
        normalized = " ".join(text.split())
        digest = fingerprint or hashlib.sha256(
            f"{kind}\0{source}\0{normalized}".encode("utf-8")
        ).hexdigest()
        return cls(
            evidence_id=evidence_id,
            kind=kind,
            source=source,
            text=text,
            fingerprint=digest,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class GapDecision:
    needs_capability: bool
    need: str | None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.needs_capability, bool):
            raise ValueError("needs_capability must be a boolean")
        need = self.need.strip() if isinstance(self.need, str) else None
        if self.needs_capability and not need:
            raise ValueError("need must be non-empty when a capability is needed")
        if not self.needs_capability and need is not None:
            raise ValueError("need must be null when no capability is needed")
        if need is not None and len(need) > 240:
            raise ValueError("need must be at most 240 characters")


class CapabilityGapDecider(Protocol):
    def decide_gap(
        self,
        evidence: RuntimeEvidence,
        active_bundle_cards: tuple[BundleSnapshot, ...],
        current_subgoal_context: str | None,
    ) -> GapDecision: ...


class CompletionCapabilityGapDecider:
    """Narrow structured-output decider; provider ownership stays external."""

    def __init__(self, completion: Callable[[str], str]) -> None:
        self._completion = completion

    def decide_gap(
        self,
        evidence: RuntimeEvidence,
        active_bundle_cards: tuple[BundleSnapshot, ...],
        current_subgoal_context: str | None,
    ) -> GapDecision:
        cards = [{
            "bundle_id": bundle.bundle_id,
            "purpose": bundle.purpose,
            "capabilities": list(bundle.capabilities),
            "active_skill_ids": [member.skill_id for member in bundle.members],
        } for bundle in active_bundle_cards]
        prompt = (
            "Decide whether this single runtime evidence item reveals a missing "
            "capability. Active Bundle Cards are summaries, not Skill bodies. "
            "Return exactly one JSON object with needs_capability (boolean), need "
            "(concise string or null), and optional rationale (short string). "
            "When needs_capability is false, need must be null.\n"
            + json.dumps({
                "evidence": {
                    "kind": evidence.kind,
                    "source": evidence.source,
                    "text": evidence.text[:2000],
                    "metadata": dict(evidence.metadata),
                },
                "active_bundle_cards": cards,
                "current_subgoal_context": current_subgoal_context,
            }, ensure_ascii=False, separators=(",", ":"))
        )
        raw = json.loads(self._completion(prompt))
        if not isinstance(raw, dict) or set(raw) - {
            "needs_capability", "need", "rationale",
        }:
            raise ValueError("gap decision must be one object with known fields")
        if "needs_capability" not in raw or "need" not in raw:
            raise ValueError("gap decision requires needs_capability and need")
        rationale = raw.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError("gap decision rationale must be a string or null")
        return GapDecision(raw["needs_capability"], raw["need"], rationale)


class RerouteState(StrEnum):
    NEW = "NEW"
    CHECKING = "CHECKING"
    GAP_FOUND = "GAP_FOUND"
    DISCOVERED = "DISCOVERED"
    SELECTED = "SELECTED"
    COMMITTED = "COMMITTED"
    NO_GAP = "NO_GAP"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RerouteTelemetryEvent:
    event: str
    evidence_id: str
    data: Mapping[str, object]


@dataclass(slots=True)
class RerouteRecord:
    evidence: RuntimeEvidence
    state: RerouteState = RerouteState.NEW
    state_history: tuple[RerouteState, ...] = (RerouteState.NEW,)
    need: str | None = None
    rationale: str | None = None
    candidates: tuple[object, ...] = ()
    selected_skill_ids: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RerouteOutcome:
    status: str
    evidence_id: str
    state: RerouteState | None = None
    need: str | None = None
    candidates: tuple[object, ...] = ()
    duplicate_of: str | None = None
    error_type: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RerouteSnapshot:
    evidence: tuple[RerouteRecord, ...]
    telemetry: tuple[RerouteTelemetryEvent, ...]


class RerouteController:
    """Turn-local evidence state machine coordinating gap checks and discovery."""

    def __init__(
        self,
        *,
        gap_decider: CapabilityGapDecider,
        discover_capability: Callable[[str], object],
        active_bundle_cards: Callable[[], tuple[BundleSnapshot, ...]],
        active_skill_ids: Callable[[], tuple[str, ...]],
    ) -> None:
        self._gap_decider = gap_decider
        self._discover_capability = discover_capability
        self._active_bundle_cards = active_bundle_cards
        self._active_skill_ids = active_skill_ids
        self._records: dict[str, RerouteRecord] = {}
        self._fingerprints: dict[str, str] = {}
        self._telemetry: list[RerouteTelemetryEvent] = []

    def begin_turn(self) -> None:
        self._clear_turn_state()

    def end_turn(self) -> None:
        self._clear_turn_state()

    def observe(
        self,
        evidence: RuntimeEvidence,
        *,
        current_subgoal_context: str | None = None,
    ) -> RerouteOutcome:
        self._emit("reroute_evidence_observed", evidence.evidence_id, {
            "fingerprint": evidence.fingerprint,
            "kind": evidence.kind,
            "source": evidence.source,
        })
        duplicate_of = self._fingerprints.get(evidence.fingerprint)
        if duplicate_of is not None:
            self._emit("reroute_evidence_eligibility", evidence.evidence_id, {
                "eligible": False, "reason": "duplicate",
                "duplicate_of": duplicate_of,
            })
            return RerouteOutcome(
                "duplicate", evidence.evidence_id, duplicate_of=duplicate_of,
            )
        eligible, reason = self._eligible(evidence)
        self._emit("reroute_evidence_eligibility", evidence.evidence_id, {
            "eligible": eligible, "reason": reason,
        })
        if not eligible:
            return RerouteOutcome("ignored", evidence.evidence_id)

        self._fingerprints[evidence.fingerprint] = evidence.evidence_id
        record = RerouteRecord(evidence=evidence)
        self._records[evidence.evidence_id] = record
        self._emit("reroute_state_transition", evidence.evidence_id, {
            "from_state": None, "to_state": RerouteState.NEW,
        })
        stage = "gap_decision"
        try:
            self._transition(record, RerouteState.CHECKING)
            cards = self._active_bundle_cards()
            skill_ids = self._active_skill_ids()
            decision = self._gap_decider.decide_gap(
                evidence, cards, current_subgoal_context,
            )
            record.need = decision.need.strip() if decision.need else None
            record.rationale = decision.rationale
            self._emit("reroute_gap_decision", evidence.evidence_id, {
                "needs_capability": decision.needs_capability,
                "generated_need": record.need,
                "active_skill_ids": skill_ids,
                "rationale": decision.rationale,
            })
            if not decision.needs_capability:
                self._transition(record, RerouteState.NO_GAP)
                return self._outcome("no_gap", record)

            self._transition(record, RerouteState.GAP_FOUND)
            stage = "discovery"
            result = self._discover_capability(record.need or "")
            record.candidates = tuple(getattr(result, "candidates"))
            self._transition(record, RerouteState.DISCOVERED)
            self._emit("reroute_discovery", evidence.evidence_id, {
                "status": "discovered",
                "query": getattr(result, "query", record.need),
                "need": record.need,
                "candidates": tuple(
                    getattr(candidate, "skill_id", str(candidate))
                    for candidate in record.candidates
                ),
                "ranks": tuple(
                    getattr(candidate, "rank", index)
                    for index, candidate in enumerate(record.candidates, 1)
                ),
            })
            return self._outcome("discovered", record)
        except Exception as exc:
            self._fail(record, stage, exc)
            event = "reroute_discovery" if stage == "discovery" else "reroute_gap_decision"
            self._emit(event, evidence.evidence_id, {
                "status": "failed",
                "stage": stage,
                "need": record.need,
                "error_type": record.error_type,
                "error_message": record.error_message,
            })
            return self._outcome("failed", record)

    def pending(self) -> tuple[RerouteRecord, ...]:
        return tuple(
            record for record in self._records.values()
            if record.state in {
                RerouteState.DISCOVERED, RerouteState.SELECTED,
            }
        )

    def mark_selected(self, evidence_id: str, skill_ids: tuple[str, ...]) -> None:
        record = self._record(evidence_id)
        try:
            if record.state is not RerouteState.DISCOVERED:
                raise ValueError("reroute evidence must be DISCOVERED before selection")
            candidate_ids = {
                getattr(candidate, "skill_id", str(candidate))
                for candidate in record.candidates
            }
            if not skill_ids or not set(skill_ids).issubset(candidate_ids):
                raise ValueError("selected Skills must be non-empty discovery candidates")
            record.selected_skill_ids = tuple(skill_ids)
            self._transition(record, RerouteState.SELECTED)
            self._emit("reroute_candidate_selected", evidence_id, {
                "selected_skill_ids": record.selected_skill_ids,
            })
        except Exception as exc:
            self._fail(record, "selection", exc)
            self._emit("reroute_candidate_selected", evidence_id, {
                "status": "failed",
                "stage": "selection",
                "selected_skill_ids": tuple(skill_ids),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
            raise

    def mark_committed(self, evidence_id: str, skill_ids: tuple[str, ...]) -> None:
        record = self._record(evidence_id)
        try:
            if record.state is not RerouteState.SELECTED:
                raise ValueError("reroute evidence must be SELECTED before commit")
            if tuple(skill_ids) != record.selected_skill_ids:
                raise ValueError("committed Skills must match the reroute selection")
            self._transition(record, RerouteState.COMMITTED)
            self._emit("reroute_apply_result", evidence_id, {
                "status": "committed",
                "selected_skill_ids": record.selected_skill_ids,
            })
            self._emit("reroute_activation", evidence_id, {
                "selected_skill_ids": record.selected_skill_ids,
            })
        except Exception as exc:
            self._fail(record, "apply", exc)
            self._emit("reroute_apply_result", evidence_id, {
                "status": "failed",
                "stage": "apply",
                "selected_skill_ids": record.selected_skill_ids,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
            raise

    def mark_apply_failed(self, evidence_id: str, error: Exception) -> None:
        record = self._record(evidence_id)
        self._fail(record, "apply", error)
        self._emit("reroute_apply_result", evidence_id, {
            "status": "failed",
            "stage": "apply",
            "selected_skill_ids": record.selected_skill_ids,
            "error_type": record.error_type,
            "error_message": record.error_message,
        })

    def record_body_load(self, skill_id: str, status: str) -> None:
        for record in self._records.values():
            if (record.state is RerouteState.COMMITTED
                    and skill_id in record.selected_skill_ids):
                self._emit("reroute_body_loaded", record.evidence.evidence_id, {
                    "skill_id": skill_id, "status": status,
                })

    def snapshot(self) -> RerouteSnapshot:
        return RerouteSnapshot(
            evidence=tuple(self._records.values()),
            telemetry=tuple(self._telemetry),
        )

    def telemetry(self) -> tuple[RerouteTelemetryEvent, ...]:
        return tuple(self._telemetry)

    @staticmethod
    def _eligible(evidence: RuntimeEvidence) -> tuple[bool, str]:
        if evidence.source in INTERNAL_EVIDENCE_SOURCES:
            return False, "control_plane_internal"
        if evidence.kind == "host_signal":
            relevant = evidence.metadata.get("capability_relevant") is True
            return relevant, "host_capability_signal" if relevant else "ordinary_host_signal"
        return True, evidence.kind

    def _record(self, evidence_id: str) -> RerouteRecord:
        try:
            return self._records[evidence_id]
        except KeyError as exc:
            raise ValueError(f"unknown reroute evidence: {evidence_id}") from exc

    def _emit(self, event: str, evidence_id: str, data: Mapping[str, object]) -> None:
        self._telemetry.append(RerouteTelemetryEvent(event, evidence_id, dict(data)))

    def _transition(self, record: RerouteRecord, state: RerouteState) -> None:
        previous = record.state
        record.state = state
        record.state_history = (*record.state_history, state)
        self._emit("reroute_state_transition", record.evidence.evidence_id, {
            "from_state": previous, "to_state": state,
        })

    def _fail(self, record: RerouteRecord, stage: str, error: Exception) -> None:
        if record.state is RerouteState.FAILED:
            return
        record.error_type = type(error).__name__
        record.error_message = str(error)[:500]
        record.failure_stage = stage
        record.failure_reason = record.error_message
        if record.state is not RerouteState.FAILED:
            self._transition(record, RerouteState.FAILED)

    @staticmethod
    def _outcome(status: str, record: RerouteRecord) -> RerouteOutcome:
        return RerouteOutcome(
            status=status,
            evidence_id=record.evidence.evidence_id,
            state=record.state,
            need=record.need,
            candidates=record.candidates,
            error_type=record.error_type,
            failure_stage=record.failure_stage,
            failure_reason=record.failure_reason,
        )

    def _clear_turn_state(self) -> None:
        self._records.clear()
        self._fingerprints.clear()
