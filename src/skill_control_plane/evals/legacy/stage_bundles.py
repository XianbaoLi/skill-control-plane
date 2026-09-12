from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from skill_control_plane.models import RetrievalCandidate, SkillRecord


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    """A runtime capability working set.

    Membership is the source of truth. The descriptor and use_when fields are a
    compact discovery surface for the main Agent.
    """

    bundle_id: str
    descriptor: str
    skill_ids: tuple[str, ...]
    best_rank: int
    use_when: str = ""


@dataclass(frozen=True, slots=True)
class CapabilityShelf:
    """Bundles discovered for the current task/session.

    active_bundle_ids is monotonic by default in V0.2. A later policy may mark
    bundles dormant without forgetting them from the shelf.
    """

    bundles: tuple[CapabilityBundle, ...] = ()
    active_bundle_ids: tuple[str, ...] = ()

    @property
    def registered_skill_ids(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for bundle in self.bundles:
            for skill_id in bundle.skill_ids:
                if skill_id not in ordered:
                    ordered.append(skill_id)
        return tuple(ordered)

    def get(self, bundle_id: str) -> CapabilityBundle | None:
        return next(
            (bundle for bundle in self.bundles if bundle.bundle_id == bundle_id),
            None,
        )

    def discovery_surface(self) -> tuple[str, ...]:
        """Compact surface suitable for keeping visible to the main Agent."""

        rows: list[str] = []
        active = set(self.active_bundle_ids)
        for bundle in self.bundles:
            marker = "active" if bundle.bundle_id in active else "shelf"
            suffix = f": {bundle.use_when}" if bundle.use_when else ""
            rows.append(f"[{marker}] {bundle.descriptor}{suffix}")
        return tuple(rows)


GroupKey = Callable[[SkillRecord], str]


def default_group_key(record: SkillRecord) -> str:
    """Deterministic V0.2 grouping baseline.

    Prefer corpus category, then the first explicit tag, then a per-Skill group.
    Future experiments may replace this with embedding clustering or MMR.
    """

    if record.category.strip():
        return record.category.strip().lower()
    if record.tags:
        return record.tags[0].strip().lower()
    return record.skill_id.strip().lower()


def _humanize_group_key(key: str) -> str:
    return " ".join(
        part for part in key.replace("_", "-").split("-") if part
    ).title()


def _use_when(records: Sequence[SkillRecord]) -> str:
    for record in records:
        description = record.description.strip()
        if description:
            return description
    return ""


def build_capability_shelf(
    candidates: Sequence[RetrievalCandidate],
    records: Mapping[str, SkillRecord],
    *,
    max_bundles: int = 4,
    max_skills_per_bundle: int = 4,
    group_key: GroupKey = default_group_key,
    activate_best: bool = True,
) -> CapabilityShelf:
    """Group a retrieval candidate pool into a compact capability shelf.

    This is a deterministic baseline, not the final bundle-selection algorithm.
    Candidate rank is preserved, so the group containing the best-ranked Skill
    becomes active by default.
    """

    if max_bundles < 1:
        raise ValueError("max_bundles must be >= 1")
    if max_skills_per_bundle < 1:
        raise ValueError("max_skills_per_bundle must be >= 1")

    grouped: defaultdict[
        str, list[tuple[RetrievalCandidate, SkillRecord]]
    ] = defaultdict(list)
    group_order: list[str] = []

    for candidate in sorted(candidates, key=lambda item: item.rank):
        record = records.get(candidate.skill_id)
        if record is None:
            continue
        key = group_key(record) or record.skill_id
        if key not in grouped:
            group_order.append(key)
        if len(grouped[key]) < max_skills_per_bundle:
            grouped[key].append((candidate, record))

    bundles: list[CapabilityBundle] = []
    for key in group_order[:max_bundles]:
        members = grouped[key]
        if not members:
            continue
        bundles.append(
            CapabilityBundle(
                bundle_id=key,
                descriptor=_humanize_group_key(key),
                skill_ids=tuple(candidate.skill_id for candidate, _ in members),
                best_rank=min(candidate.rank for candidate, _ in members),
                use_when=_use_when([record for _, record in members]),
            )
        )

    active = (bundles[0].bundle_id,) if activate_best and bundles else ()
    return CapabilityShelf(bundles=tuple(bundles), active_bundle_ids=active)


def integrate_retrieval_delta(
    shelf: CapabilityShelf,
    candidates: Sequence[RetrievalCandidate],
    records: Mapping[str, SkillRecord],
    *,
    max_new_bundles: int = 4,
    max_skills_per_bundle: int = 4,
    group_key: GroupKey = default_group_key,
    activate_best_delta: bool = True,
) -> CapabilityShelf:
    """Monotonically integrate a later evidence-driven retrieval pass.

    If a delta group already exists on the shelf, its new Skills are appended.
    Otherwise a new bundle is created. Existing active bundles remain active;
    the best delta bundle can additionally become active.
    """

    delta = build_capability_shelf(
        candidates,
        records,
        max_bundles=max_new_bundles,
        max_skills_per_bundle=max_skills_per_bundle,
        group_key=group_key,
        activate_best=activate_best_delta,
    )

    by_id = {bundle.bundle_id: bundle for bundle in shelf.bundles}
    order = [bundle.bundle_id for bundle in shelf.bundles]

    for incoming in delta.bundles:
        existing = by_id.get(incoming.bundle_id)
        if existing is None:
            by_id[incoming.bundle_id] = incoming
            order.append(incoming.bundle_id)
            continue

        skill_ids = list(existing.skill_ids)
        for skill_id in incoming.skill_ids:
            if skill_id not in skill_ids:
                skill_ids.append(skill_id)

        by_id[incoming.bundle_id] = replace(
            existing,
            skill_ids=tuple(skill_ids),
            best_rank=min(existing.best_rank, incoming.best_rank),
            use_when=existing.use_when or incoming.use_when,
        )

    active = list(shelf.active_bundle_ids)
    for bundle_id in delta.active_bundle_ids:
        if bundle_id not in active:
            active.append(bundle_id)

    return CapabilityShelf(
        bundles=tuple(by_id[bundle_id] for bundle_id in order),
        active_bundle_ids=tuple(active),
    )
