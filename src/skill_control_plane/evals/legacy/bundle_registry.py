"""Purpose-based compositions, distinct from experimental stage working sets."""
from dataclasses import dataclass
from collections.abc import Iterable
from types import MappingProxyType

from skill_control_plane.registry import SkillRegistry


@dataclass(frozen=True)
class Bundle:
    bundle_id: str
    purpose: str
    skill_ids: tuple[str, ...]
    status: str = "active"
    validated: bool = False

    def __post_init__(self):
        if not self.bundle_id.strip() or not self.purpose.strip() or not self.skill_ids:
            raise ValueError("bundle requires id, purpose and skills")
        if not isinstance(self.skill_ids, tuple) or len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("skill_ids must be a unique tuple")
        if self.status not in {"active", "draft", "inactive"}:
            raise ValueError("invalid bundle status")


class BundleRegistry:
    def __init__(self, skills: SkillRegistry, bundles: Iterable[Bundle] = ()):
        self._skill_ids = frozenset(s.skill_id for s in skills)
        self._bundles: dict[str, Bundle] = {}
        self._inverted: dict[str, frozenset[str]] = {}
        for bundle in bundles:
            self.add(bundle)

    def add(self, bundle: Bundle) -> None:
        if bundle.bundle_id in self._bundles:
            raise ValueError("duplicate bundle_id")
        unknown = set(bundle.skill_ids) - self._skill_ids
        if unknown:
            raise ValueError(f"unknown skills: {sorted(unknown)}")
        self._bundles[bundle.bundle_id] = bundle
        for skill_id in bundle.skill_ids:
            self._inverted[skill_id] = self._inverted.get(skill_id, frozenset()) | {bundle.bundle_id}

    @property
    def skill_to_bundles(self):
        return MappingProxyType(self._inverted)

    @property
    def skill_ids(self) -> frozenset[str]:
        return self._skill_ids

    def get(self, bundle_id: str) -> Bundle:
        return self._bundles[bundle_id]

    def values(self) -> tuple[Bundle, ...]:
        return tuple(self._bundles.values())
