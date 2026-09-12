"""Cross-turn Bundle memory and Skill body residency."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping
from uuid import uuid4

from skill_control_plane.registry import SkillStore


@dataclass(frozen=True)
class ActiveBundle:
    bundle_id: str
    purpose: str
    skill_ids: tuple[str, ...]


BodyState = Literal["resident", "evicted"]


@dataclass(frozen=True, slots=True)
class BundleMemberSnapshot:
    skill_id: str
    name: str
    body_state: BodyState
    short_description: str | None = None


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    bundle_id: str
    purpose: str
    capabilities: tuple[str, ...]
    members: tuple[BundleMemberSnapshot, ...]


@dataclass(frozen=True, slots=True)
class CapabilityMemorySnapshot:
    direct_skill_ids: tuple[str, ...]
    maintained_bundles: tuple[BundleSnapshot, ...]

    @property
    def skill_body_states(self) -> dict[str, BodyState]:
        return {
            member.skill_id: member.body_state
            for bundle in self.maintained_bundles
            for member in bundle.members
        }


@dataclass(frozen=True, slots=True)
class SkillBody:
    skill_id: str
    body: str


@dataclass(frozen=True, slots=True)
class SkillBodyLoadResult:
    status: Literal["loaded", "already_resident"]
    skill_id: str
    bundle_ids: tuple[str, ...]
    body: str | None = None


@dataclass
class RuntimeCapabilityState:
    """Serializable runtime state retained for compatibility with V0.x callers.

    ``direct_skills`` are temporary model-context activations.  They are never
    rendered as, or committed into, the maintained Bundle memory surface.
    """

    active_bundles: list[ActiveBundle] = field(default_factory=list)
    direct_skills: set[str] = field(default_factory=set)
    skill_body_states: dict[str, BodyState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for bundle in self.active_bundles:
            for skill_id in bundle.skill_ids:
                self.skill_body_states.setdefault(skill_id, "evicted")


def validate_state(state: RuntimeCapabilityState, store: SkillStore) -> None:
    bundle_ids = [bundle.bundle_id for bundle in state.active_bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("duplicate active bundle id")
    known = {skill.skill_id for skill in store}
    if not state.direct_skills <= known:
        raise ValueError("unknown direct skill")
    bundle_skills: set[str] = set()
    for bundle in state.active_bundles:
        if not bundle.bundle_id.strip() or not bundle.purpose.strip() or not bundle.skill_ids:
            raise ValueError("active bundle requires id, purpose and skills")
        if len(bundle.skill_ids) != len(set(bundle.skill_ids)) or set(bundle.skill_ids) - known:
            raise ValueError("unknown or duplicate active bundle skill")
        bundle_skills.update(bundle.skill_ids)
    if set(state.skill_body_states) != bundle_skills:
        raise ValueError("body state must exist exactly for Bundle members")
    if set(state.skill_body_states.values()) - {"resident", "evicted"}:
        raise ValueError("invalid Skill body state")


class CapabilityMemory:
    """Compression-resistant Bundle Cards plus exact Skill body state."""

    def __init__(
        self,
        store: SkillStore,
        state: RuntimeCapabilityState | None = None,
        *,
        capability_phrases: Mapping[str, tuple[str, ...]] | None = None,
        bundle_capability_phrases: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.store = store
        self.state = state if state is not None else RuntimeCapabilityState()
        self.capability_phrases = dict(capability_phrases or {})
        self.bundle_capability_phrases = dict(bundle_capability_phrases or {})
        self.body_load_count = 0
        validate_state(self.state, self.store)

    @property
    def loaded_skill_ids(self) -> tuple[str, ...]:
        ids = set(self.state.direct_skills)
        for bundle in self.state.active_bundles:
            ids.update(bundle.skill_ids)
        return tuple(sorted(ids))

    @staticmethod
    def _clean_phrase(raw: str, *, max_chars: int) -> str:
        phrase = " ".join(raw.split()).strip()
        if len(phrase) > max_chars:
            phrase = phrase[: max_chars - 1].rstrip() + "…"
        return phrase

    @staticmethod
    def _phrase_key(phrase: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", phrase.casefold()).split())

    @classmethod
    def _is_redundant(cls, phrase: str, selected: list[str]) -> bool:
        key = cls._phrase_key(phrase)
        return any(
            key == other
            or f" {key} " in f" {other} "
            or f" {other} " in f" {key} "
            for other in (cls._phrase_key(value) for value in selected)
        )

    def _old_capabilities(
        self, skill_ids: tuple[str, ...], *, max_items: int = 8,
        max_chars: int = 120,
    ) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for skill_id in sorted(skill_ids):
            structured = self.capability_phrases.get(skill_id) or (
                self.store.get(skill_id).description,
            )
            for raw in structured:
                phrase = self._clean_phrase(raw, max_chars=max_chars)
                key = phrase.casefold()
                if not phrase or key in seen:
                    continue
                seen.add(key)
                phrases.append(phrase)
                if len(phrases) == max_items:
                    return phrases
        return phrases

    def _compact_capabilities(
        self, skill_ids: tuple[str, ...], *, max_items: int = 5,
        max_chars: int = 96,
    ) -> list[str]:
        member_phrases = [
            [self._clean_phrase(raw, max_chars=max_chars) for raw in (
                self.bundle_capability_phrases.get(skill_id)
                or (self.store.get(skill_id).description,)
            )]
            for skill_id in sorted(skill_ids)
        ]
        selected: list[str] = []
        offset = 0
        while len(selected) < max_items:
            consumed = False
            for phrases in member_phrases:
                if offset >= len(phrases):
                    continue
                consumed = True
                phrase = phrases[offset]
                if phrase and not self._is_redundant(phrase, selected):
                    selected.append(phrase)
                    if len(selected) == max_items:
                        return selected
            if not consumed:
                break
            offset += 1
        return selected

    def snapshot(self, *, compact: bool = True) -> CapabilityMemorySnapshot:
        """Return capability memory as data, never as provider-facing text."""

        capabilities = self._compact_capabilities if compact else self._old_capabilities
        return CapabilityMemorySnapshot(
            direct_skill_ids=tuple(sorted(self.state.direct_skills)),
            maintained_bundles=tuple(
                BundleSnapshot(
                    bundle_id=bundle.bundle_id,
                    purpose=bundle.purpose,
                    capabilities=tuple(capabilities(bundle.skill_ids)),
                    members=tuple(
                        BundleMemberSnapshot(
                            skill_id=skill_id,
                            name=self.store.get(skill_id).name,
                            body_state=self.state.skill_body_states[skill_id],
                            short_description=(None if compact else " ".join(
                                self.store.get(skill_id).description.split())[:240]),
                        )
                        for skill_id in sorted(bundle.skill_ids)
                    ),
                )
                for bundle in sorted(
                    self.state.active_bundles, key=lambda item: item.bundle_id)
            ),
        )

    def commit(
        self,
        *,
        action: str,
        skill_ids: tuple[str, ...],
        target_bundle_id: str | None = None,
        purpose: str | None = None,
    ) -> tuple[str | None, tuple[SkillBody, ...]]:
        for skill_id in skill_ids:
            self.store.get(skill_id)
        active = list(self.state.active_bundles)
        direct = set(self.state.direct_skills)
        target = None
        if action == "DIRECT":
            direct.update(skill_ids)
        elif action == "EXTEND":
            target = target_bundle_id
            active = [
                ActiveBundle(
                    bundle.bundle_id,
                    bundle.purpose,
                    tuple(dict.fromkeys((*bundle.skill_ids, *skill_ids))),
                ) if bundle.bundle_id == target else bundle
                for bundle in active
            ]
        elif action == "CREATE":
            target = "cap-" + uuid4().hex
            while target in {bundle.bundle_id for bundle in active}:
                target = "cap-" + uuid4().hex
            active.append(ActiveBundle(target, purpose or "", skill_ids))
        else:
            raise ValueError("unknown capability action")

        next_state = RuntimeCapabilityState(
            active, direct, dict(self.state.skill_body_states))
        bundle_skill_ids = {
            skill_id for bundle in next_state.active_bundles
            for skill_id in bundle.skill_ids
        }
        for skill_id in skill_ids:
            if skill_id in bundle_skill_ids:
                next_state.skill_body_states[skill_id] = "resident"
        validate_state(next_state, self.store)
        bodies = tuple(
            SkillBody(skill_id, self.store.get(skill_id).body)
            for skill_id in skill_ids
        )
        self.state = next_state
        return target, bodies

    def load_skill_body(self, skill_id: str) -> SkillBodyLoadResult:
        validate_state(self.state, self.store)
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("load_skill_body requires a non-empty string skill_id")
        bundle_ids = sorted(
            bundle.bundle_id for bundle in self.state.active_bundles
            if skill_id in bundle.skill_ids
        )
        if not bundle_ids:
            raise ValueError("load_skill_body requires a current Bundle member")
        if self.state.skill_body_states[skill_id] == "resident":
            return SkillBodyLoadResult(
                "already_resident", skill_id, tuple(bundle_ids))
        body = self.store.load_skill_body(skill_id)
        self.body_load_count += 1
        self.state.skill_body_states[skill_id] = "resident"
        return SkillBodyLoadResult("loaded", skill_id, tuple(bundle_ids), body)

    def mark_skill_body_evicted(self, skill_id: str) -> None:
        validate_state(self.state, self.store)
        if skill_id not in self.state.skill_body_states:
            raise ValueError("body eviction requires a current Bundle member")
        self.state.skill_body_states[skill_id] = "evicted"

    def mark_all_skill_bodies_evicted(self) -> None:
        validate_state(self.state, self.store)
        for skill_id in self.state.skill_body_states:
            self.state.skill_body_states[skill_id] = "evicted"
