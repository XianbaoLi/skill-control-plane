"""Deterministic complete-Discovery fixtures; never call embedding services."""
from dataclasses import replace
from typing import Iterable

from skill_control_plane.discovery import BM25Retriever, RetrievalCard, SkillDiscovery
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore


class DeterministicFakeDenseRetriever:
    """Stable test double with Dense-shaped scores and BM25-like selectivity."""

    def __init__(self, records: Iterable[SkillRecord]) -> None:
        self._retriever = BM25Retriever(records)

    def search(self, query: str, k: int = 5):
        return [replace(
            candidate,
            source_scores={"dense": candidate.score},
        ) for candidate in self._retriever.search(query, k)]


def complete_cards(store: SkillStore) -> dict[str, RetrievalCard]:
    return {
        skill.skill_id: RetrievalCard(
            skill_id=skill.skill_id,
            source_content_hash=skill.content_hash,
            purpose=skill.description or skill.name,
            use_when=(f"when {skill.description or skill.name}",),
            capabilities=(skill.description or skill.name,),
            lexical_cues=(skill.name, skill.skill_id),
        )
        for skill in store
    }


def full_discovery(
    store: SkillStore,
    *,
    cards: dict[str, RetrievalCard] | None = None,
) -> SkillDiscovery:
    return SkillDiscovery(
        store,
        retrieval_cards=complete_cards(store) if cards is None else cards,
        dense_factory=DeterministicFakeDenseRetriever,
    )
