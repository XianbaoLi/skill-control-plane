"""Single-search Capability Discovery over Retrieval Cards, BM25, Dense and RRF."""
from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence

from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.registry import SkillStore
from .base import Retriever
from .bm25 import BM25Retriever
from .dense import dense_text
from .fusion import reciprocal_rank_fusion


@dataclass(frozen=True)
class SkillDiscoveryResult:
    query: str
    candidates: tuple[RetrievalCandidate, ...]
    representations: tuple[tuple[str, str], ...]
    backend: str
    truncated: bool


class SkillDiscovery:
    """Immutable index snapshot. Optional dense_factory receives identical text."""

    def __init__(self, registry: SkillStore, *,
                 dense_factory: Callable[[Sequence[SkillRecord]], Retriever] | None = None,
                 source_k: int = 10, retrieval_cards=None):
        if source_k < 1:
            raise ValueError("source_k must be positive")
        self.source_k = source_k
        self.registry = registry
        self.retrieval_cards = dict(retrieval_cards or {})
        self.records = {s.skill_id: s for s in registry}
        records = indexed_skill_records(registry, self.retrieval_cards)
        self.texts = {
            skill.skill_id: dense_text(skill)
            for skill in records
        }
        indexed = tuple(replace(s, name="", description=self.texts[s.skill_id],
                                tags=(), body="") for s in registry)
        self.bm25 = BM25Retriever(indexed)
        self.dense = dense_factory(indexed) if dense_factory else None

    @property
    def fusion_backend(self) -> str:
        """Configured search path; production requires ``rrf``."""

        return "rrf" if self.dense is not None else "bm25"

    def capability_phrases(self, skill_id: str) -> tuple[str, ...]:
        """Expose structured Retrieval Card coverage without searching."""

        self.registry.get(skill_id)
        card = self.retrieval_cards.get(skill_id)
        if card is None:
            return ()
        return tuple((*card.capabilities, *card.use_when))

    def bundle_capability_phrases(self, skill_id: str) -> tuple[str, ...]:
        """Expose the prioritized compact Bundle representation without searching."""

        record = self.registry.get(skill_id)
        card = self.retrieval_cards.get(skill_id)
        if card is not None:
            if card.capabilities:
                return tuple(card.capabilities)
            if card.use_when:
                return tuple(card.use_when)
        description = " ".join(record.description.split()).strip()
        return (description,) if description else ()

    @classmethod
    def from_tree(cls, root, *, cards=None, dense_factory=None,
                  source_k: int = 10) -> "SkillDiscovery":
        """Build a Skill Store and its separate discovery index."""

        store = SkillStore.from_tree(root)
        if cards is not None:
            from .cards import validate_retrieval_cards

            validate_retrieval_cards(store, cards)
        return cls(store, dense_factory=dense_factory,
                   source_k=source_k, retrieval_cards=cards)

    def discover_skills(self, query: str, k: int = 5) -> SkillDiscoveryResult:
        if not query.strip() or k < 1:
            raise ValueError("query must be non-empty and k must be positive")
        depth = max(self.source_k, k + 1)
        rankings = {"bm25": self.bm25.search(query, k=depth)}
        if self.dense is not None:
            rankings["dense"] = self.dense.search(query, k=depth)
            candidates = reciprocal_rank_fusion(rankings)
        else:
            candidates = rankings["bm25"]
        chosen = tuple(candidates[:k])
        return SkillDiscoveryResult(query, chosen,
            tuple((c.skill_id, self.texts[c.skill_id]) for c in chosen),
            self.fusion_backend, len(candidates) > k)

    def model_visible_payload(self, result: SkillDiscoveryResult) -> dict:
        """Serialize candidates for an LLM without exposing retrieval cards/debug data."""
        unknown = {candidate.skill_id for candidate in result.candidates} - set(self.records)
        if unknown:
            raise ValueError(f"unknown discovery candidates: {sorted(unknown)}")
        candidates = []
        for candidate in result.candidates:
            record = self.records[candidate.skill_id]
            minimal_evidence: dict[str, object] = {}
            matched = next((item.removeprefix("matched_terms: ").split(", ")
                            for item in candidate.evidence
                            if item.startswith("matched_terms: ")), None)
            if matched:
                minimal_evidence["matched_terms"] = matched[:6]
            if "dense" in candidate.source_scores:
                minimal_evidence["semantic_similarity"] = round(
                    float(candidate.source_scores["dense"]), 4)
            candidates.append({
                "skill_id": candidate.skill_id,
                "name": record.name,
                "description": record.description,
                "rank": candidate.rank,
                "minimal_evidence": minimal_evidence,
            })
        return {"query": result.query, "candidates": candidates}


def discover_skills(query: str, k: int = 5, *, discovery: SkillDiscovery) -> SkillDiscoveryResult:
    return discovery.discover_skills(query, k)


def indexed_skill_records(registry, retrieval_cards) -> tuple[SkillRecord, ...]:
    """Return the exact Skill records used to build Discovery indexes."""

    cards = dict(retrieval_cards or {})
    return tuple(
        replace(
            skill,
            description="\n".join(part for part in (
                skill.description,
                cards[skill.skill_id].augmentation_text(),
            ) if part),
            body="",
        ) if skill.skill_id in cards else skill
        for skill in registry
    )
