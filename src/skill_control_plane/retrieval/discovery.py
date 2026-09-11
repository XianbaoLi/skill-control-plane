"""Stable discovery over the existing representation, BM25, Dense and RRF."""
from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence

from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.registry import SkillRegistry
from .base import Retriever
from .bm25 import BM25Retriever
from .dense import metadata_text
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

    def __init__(self, registry: SkillRegistry, *,
                 dense_factory: Callable[[Sequence[SkillRecord]], Retriever] | None = None,
                 source_k: int = 10):
        if source_k < 1:
            raise ValueError("source_k must be positive")
        self.source_k = source_k
        self.registry = registry
        self.records = {s.skill_id: s for s in registry}
        self.texts = {s.skill_id: s.retrieval_representation or metadata_text(s)
                      for s in registry}
        indexed = tuple(replace(s, name="", description=self.texts[s.skill_id],
                                tags=(), body="") for s in registry)
        self.bm25 = BM25Retriever(indexed)
        self.dense = dense_factory(indexed) if dense_factory else None

    def load_skill_body(self, skill_id: str) -> str:
        """Delegate an exact body read to the canonical Skill store."""

        return self.registry.load_skill_body(skill_id)

    def capability_phrases(self, skill_id: str) -> tuple[str, ...]:
        """Expose offline structured coverage metadata without searching."""

        return self.registry.capability_phrases(skill_id)

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
            "rrf" if self.dense is not None else "bm25", len(candidates) > k)

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
