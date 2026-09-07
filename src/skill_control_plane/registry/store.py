from __future__ import annotations

from collections.abc import Iterable, Iterator

from skill_control_plane.models import SkillRecord


class SkillRegistry:
    """Small in-memory registry used by V0.1 retrieval and evaluation."""

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

    def values(self) -> tuple[SkillRecord, ...]:
        return tuple(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[SkillRecord]:
        return iter(self._skills.values())
