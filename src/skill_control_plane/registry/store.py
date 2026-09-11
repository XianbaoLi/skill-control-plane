from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace

from skill_control_plane.models import SkillRecord


class SkillRegistry:
    """Canonical in-memory records with registration-time retrieval text."""

    def __init__(self, skills: Iterable[SkillRecord] = (), *,
                 retrieval_cards=None) -> None:
        self._skills: dict[str, SkillRecord] = {}
        self._retrieval_cards = dict(retrieval_cards or {})
        for skill in skills:
            self.add(skill)

    def add(self, skill: SkillRecord) -> None:
        if skill.skill_id in self._skills:
            raise ValueError(f"duplicate skill_id: {skill.skill_id}")
        if not skill.retrieval_representation:
            from skill_control_plane.retrieval.dense import metadata_text

            skill = replace(skill, retrieval_representation=metadata_text(skill))
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> SkillRecord:
        return self._skills[skill_id]

    def load_skill_body(self, skill_id: str) -> str:
        """Read a Skill body by canonical ID without invoking retrieval."""

        return self.get(skill_id).body

    def capability_phrases(self, skill_id: str) -> tuple[str, ...]:
        """Structured coverage phrases retained from the offline card."""

        self.get(skill_id)  # Preserve the canonical exact-ID check.
        card = self._retrieval_cards.get(skill_id)
        if card is None:
            return ()
        return tuple((*card.capabilities, *card.use_when))

    def values(self) -> tuple[SkillRecord, ...]:
        return tuple(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[SkillRecord]:
        return iter(self._skills.values())

    @classmethod
    def from_tree(cls, root, *, cards=None) -> SkillRegistry:
        """Register canonical records and prepare retrieval text once, offline."""
        from .loader import load_skill_tree
        from skill_control_plane.retrieval.cards import apply_retrieval_cards
        from skill_control_plane.retrieval.dense import metadata_text

        skills = load_skill_tree(root)
        projected = apply_retrieval_cards(skills, cards) if cards is not None else skills
        return cls(
            (replace(skill, retrieval_representation=metadata_text(indexed))
             for skill, indexed in zip(skills, projected, strict=True)),
            retrieval_cards=cards,
        )
