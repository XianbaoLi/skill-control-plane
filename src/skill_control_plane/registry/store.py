from __future__ import annotations

from collections.abc import Iterable, Iterator

from skill_control_plane.models import SkillRecord


class SkillStore:
    """Canonical Skill Store with exact-ID lookup.

    The store owns parsed Skill metadata and bodies.  Search representations and
    Retrieval Cards are deliberately constructed by Capability Discovery.
    """

    def __init__(self, skills: Iterable[SkillRecord] = ()) -> None:
        self._skills: dict[str, SkillRecord] = {}
        for skill in skills:
            self.add(skill)

    def add(self, skill: SkillRecord) -> None:
        if skill.skill_id in self._skills:
            raise ValueError(f"duplicate skill_id: {skill.skill_id}")
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> SkillRecord:
        return self._skills[skill_id]

    def load_skill_body(self, skill_id: str) -> str:
        """Read a Skill body by canonical ID without invoking retrieval."""

        return self.get(skill_id).body

    def values(self) -> tuple[SkillRecord, ...]:
        return tuple(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[SkillRecord]:
        return iter(self._skills.values())

    @classmethod
    def from_tree(cls, root) -> "SkillStore":
        """Load and register canonical records without building a search index."""
        from .loader import load_skill_tree

        return cls(load_skill_tree(root))


# Backward-compatible import name.  V1 architecture names this component Skill Store.
SkillRegistry = SkillStore
