"""Cross-turn Bundle memory and Skill body residency."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, get_args
from uuid import uuid4

from skill_control_plane.registry import SkillStore


STATE_SNAPSHOT_VERSION = "state-snapshot-v1"


@dataclass(frozen=True)
class ActiveBundle:
    bundle_id: str
    purpose: str
    skill_ids: tuple[str, ...]


BodyState = Literal["resident", "evicted"]
MemberRole = Literal["maintained", "direct"]


@dataclass(frozen=True, slots=True)
class BundleMemberSnapshot:
    skill_id: str
    name: str
    member_role: MemberRole
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


@dataclass(frozen=True, slots=True)
class StateSnapshotMemberV1:
    skill_id: str
    member_role: MemberRole
    body_state: BodyState


@dataclass(frozen=True, slots=True)
class StateSnapshotBundleV1:
    bundle_id: str
    purpose: str
    members: tuple[StateSnapshotMemberV1, ...]


@dataclass(frozen=True, slots=True)
class StateSnapshotV1:
    """Versioned persistent Capability Memory, excluding turn-local state."""

    version: Literal["state-snapshot-v1"] = STATE_SNAPSHOT_VERSION
    store_fingerprint: str = ""
    bundles: tuple[StateSnapshotBundleV1, ...] = ()


@dataclass
class RuntimeCapabilityState:
    """Serializable runtime state retained for compatibility with V0.x callers.

    ``bundle_member_roles`` is the canonical V1 role state. ``direct_skills`` is
    retained as a deprecated compatibility projection, plus any detached DIRECT
    IDs loaded from a V0.x state. New V1 commits never create detached DIRECT IDs.
    """

    active_bundles: list[ActiveBundle] = field(default_factory=list)
    direct_skills: set[str] = field(default_factory=set)
    skill_body_states: dict[str, BodyState] = field(default_factory=dict)
    bundle_member_roles: dict[str, dict[str, MemberRole]] = field(
        default_factory=dict)

    def __post_init__(self) -> None:
        self.active_bundles = list(self.active_bundles)
        self.direct_skills = set(self.direct_skills)
        self.skill_body_states = dict(self.skill_body_states)
        self.bundle_member_roles = {
            bundle_id: dict(member_roles)
            for bundle_id, member_roles in self.bundle_member_roles.items()
        }
        for bundle in self.active_bundles:
            roles = self.bundle_member_roles.setdefault(bundle.bundle_id, {})
            for skill_id in bundle.skill_ids:
                self.skill_body_states.setdefault(skill_id, "evicted")
                roles.setdefault(
                    skill_id,
                    "direct" if skill_id in self.direct_skills else "maintained",
                )
                if roles[skill_id] == "direct":
                    self.direct_skills.add(skill_id)
        bundle_skills = {
            skill_id for bundle in self.active_bundles
            for skill_id in bundle.skill_ids
        }
        self.direct_skills = {
            *(self.direct_skills - bundle_skills),
            *(skill_id for roles in self.bundle_member_roles.values()
              for skill_id, role in roles.items() if role == "direct"),
        }


def _ensure_compatibility_shape(state: RuntimeCapabilityState) -> None:
    """Add V1 role metadata to mutable V0.x state objects at the boundary."""

    if not hasattr(state, "bundle_member_roles"):
        state.bundle_member_roles = {}
    for bundle in state.active_bundles:
        roles = state.bundle_member_roles.setdefault(bundle.bundle_id, {})
        for skill_id in bundle.skill_ids:
            roles.setdefault(
                skill_id,
                "direct" if skill_id in state.direct_skills else "maintained",
            )
    bundle_skills = {
        skill_id for bundle in state.active_bundles
        for skill_id in bundle.skill_ids
    }
    state.direct_skills = {
        *(state.direct_skills - bundle_skills),
        *(skill_id for roles in state.bundle_member_roles.values()
          for skill_id, role in roles.items() if role == "direct"),
    }


def validate_state(state: RuntimeCapabilityState, store: SkillStore) -> None:
    _ensure_compatibility_shape(state)
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
        roles = state.bundle_member_roles.get(bundle.bundle_id)
        if roles is None or set(roles) != set(bundle.skill_ids):
            raise ValueError("member role must exist exactly for Bundle members")
        if set(roles.values()) - {"maintained", "direct"}:
            raise ValueError("invalid Bundle member role")
        bundle_skills.update(bundle.skill_ids)
    if set(state.bundle_member_roles) != set(bundle_ids):
        raise ValueError("member roles must exist exactly for active Bundles")
    if set(state.skill_body_states) != bundle_skills:
        raise ValueError("body state must exist exactly for Bundle members")
    if set(state.skill_body_states.values()) - {"resident", "evicted"}:
        raise ValueError("invalid Skill body state")


def capability_store_fingerprint(store: SkillStore) -> str:
    """Return a stable semantic identity for the Skill corpus."""

    digest = hashlib.sha256()
    for skill in sorted(store, key=lambda item: item.skill_id):
        for value in (
            skill.skill_id,
            skill.name,
            skill.description,
            skill.body,
            skill.content_hash,
            "\x1f".join(skill.tags),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return "skill-store-v1:" + digest.hexdigest()


def validate_state_snapshot(
    snapshot: StateSnapshotV1,
    store: SkillStore,
) -> RuntimeCapabilityState:
    """Validate a snapshot completely and project it to canonical runtime state."""

    if not isinstance(snapshot, StateSnapshotV1):
        raise ValueError("restore_state requires StateSnapshotV1")
    if snapshot.version != STATE_SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported state snapshot version: {snapshot.version!r}")
    expected_fingerprint = capability_store_fingerprint(store)
    if snapshot.store_fingerprint != expected_fingerprint:
        raise ValueError("state snapshot Skill Store fingerprint mismatch")
    if not isinstance(snapshot.bundles, tuple):
        raise ValueError("state snapshot bundles must be a tuple")

    active_bundles: list[ActiveBundle] = []
    member_roles: dict[str, dict[str, MemberRole]] = {}
    skill_body_states: dict[str, BodyState] = {}
    bundle_ids: set[str] = set()
    for bundle in snapshot.bundles:
        if not isinstance(bundle, StateSnapshotBundleV1):
            raise ValueError("state snapshot bundle has an invalid type")
        if not isinstance(bundle.bundle_id, str) or not bundle.bundle_id.strip():
            raise ValueError("state snapshot bundle requires a non-empty bundle_id")
        if bundle.bundle_id in bundle_ids:
            raise ValueError("duplicate active bundle id")
        if not isinstance(bundle.purpose, str) or not bundle.purpose.strip():
            raise ValueError("state snapshot Bundle requires a non-empty purpose")
        if not isinstance(bundle.members, tuple):
            raise ValueError("state snapshot Bundle members must be a tuple")

        roles: dict[str, MemberRole] = {}
        member_ids: list[str] = []
        for member in bundle.members:
            if not isinstance(member, StateSnapshotMemberV1):
                raise ValueError("state snapshot member has an invalid type")
            if not isinstance(member.skill_id, str) or not member.skill_id.strip():
                raise ValueError("state snapshot member requires a skill_id")
            if member.skill_id in member_ids:
                raise ValueError("duplicate active bundle skill")
            try:
                store.get(member.skill_id)
            except KeyError as exc:
                raise ValueError(
                    f"unknown state snapshot Skill: {member.skill_id}") from exc
            if member.member_role not in get_args(MemberRole):
                raise ValueError("invalid Bundle member role")
            if member.body_state not in get_args(BodyState):
                raise ValueError("invalid Skill body state")

            member_ids.append(member.skill_id)
            roles[member.skill_id] = member.member_role
            skill_body_states[member.skill_id] = member.body_state

        bundle_ids.add(bundle.bundle_id)
        active_bundles.append(ActiveBundle(
            bundle.bundle_id, bundle.purpose, tuple(member_ids)))
        member_roles[bundle.bundle_id] = roles

    direct_skills = {
        skill_id
        for roles in member_roles.values()
        for skill_id, role in roles.items()
        if role == "direct"
    }
    state = RuntimeCapabilityState(
        active_bundles=active_bundles,
        direct_skills=direct_skills,
        skill_body_states=skill_body_states,
        bundle_member_roles=member_roles,
    )
    validate_state(state, store)
    return state


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
                            member_role=self.state.bundle_member_roles[
                                bundle.bundle_id][skill_id],
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
        roles = {
            bundle_id: dict(member_roles)
            for bundle_id, member_roles in self.state.bundle_member_roles.items()
        }
        previous_bundle_skills = {
            skill_id for bundle in active for skill_id in bundle.skill_ids
        }
        detached_legacy_direct = (
            set(self.state.direct_skills) - previous_bundle_skills)
        target = None
        if action == "DIRECT":
            target = target_bundle_id
            existing = next((
                bundle for bundle in active if bundle.bundle_id == target
            ), None)
            if existing is None:
                raise ValueError("DIRECT requires an existing target Bundle")
            new_skill_ids = tuple(
                skill_id for skill_id in skill_ids
                if skill_id not in existing.skill_ids
            )
            if not new_skill_ids:
                raise ValueError("DIRECT requires at least one new Bundle member")
            active = [
                ActiveBundle(
                    bundle.bundle_id,
                    bundle.purpose,
                    tuple(dict.fromkeys((*bundle.skill_ids, *skill_ids))),
                ) if bundle.bundle_id == target else bundle
                for bundle in active
            ]
            roles[target].update((skill_id, "direct") for skill_id in new_skill_ids)
        elif action == "EXTEND":
            target = target_bundle_id
            existing = next((
                bundle for bundle in active if bundle.bundle_id == target
            ), None)
            if existing is None:
                raise ValueError("EXTEND requires an existing target Bundle")
            new_skill_ids = tuple(
                skill_id for skill_id in skill_ids
                if skill_id not in existing.skill_ids
            )
            if not new_skill_ids:
                raise ValueError("EXTEND requires at least one new Bundle member")
            active = [
                ActiveBundle(
                    bundle.bundle_id,
                    bundle.purpose,
                    tuple(dict.fromkeys((*bundle.skill_ids, *skill_ids))),
                ) if bundle.bundle_id == target else bundle
                for bundle in active
            ]
            roles[target].update(
                (skill_id, "maintained") for skill_id in new_skill_ids)
            detached_legacy_direct.difference_update(skill_ids)
        elif action == "CREATE":
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError("CREATE requires non-empty purpose")
            target = "cap-" + uuid4().hex
            while target in {bundle.bundle_id for bundle in active}:
                target = "cap-" + uuid4().hex
            active.append(ActiveBundle(target, purpose or "", skill_ids))
            roles[target] = {skill_id: "maintained" for skill_id in skill_ids}
            detached_legacy_direct.difference_update(skill_ids)
        else:
            raise ValueError("unknown capability action")

        direct = set(detached_legacy_direct)
        direct.update(
            skill_id
            for member_roles in roles.values()
            for skill_id, role in member_roles.items()
            if role == "direct"
        )
        next_state = RuntimeCapabilityState(
            active,
            direct,
            dict(self.state.skill_body_states),
            roles,
        )
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
