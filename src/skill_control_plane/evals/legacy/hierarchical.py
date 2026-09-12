"""Experimental known-Shelf-first retrieval; no Gold or SRC dependencies."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.discovery import BM25Retriever, DenseRetriever, candidate_union
from skill_control_plane.discovery.base import Retriever
from skill_control_plane.discovery.dense import metadata_text
from skill_control_plane.evals.legacy.stage_bundles import CapabilityShelf, default_group_key

RetrieverFactory = Callable[[Sequence[SkillRecord]], Retriever]


@dataclass(frozen=True)
class RetrievalRoute:
    candidates: tuple[RetrievalCandidate, ...]
    route: str
    corpus_size: int
    matched_bundle_id: str | None = None
    match_score: float | None = None
    match_margin: float | None = None
    matcher_corpus_size: int = 0


class ShelfAwareRetriever:
    """Match compact registered Bundle descriptions, then search their category.

    The local universe includes unregistered records sharing the existing V0.2
    group key. Searching only registered members could not expand a Bundle.
    Thresholds are fixed experimental defaults, not fitted using Gold labels.
    """

    def __init__(
        self,
        records: Mapping[str, SkillRecord],
        *,
        bm25: Retriever,
        dense: Retriever,
        dense_factory: RetrieverFactory,
        min_score: float = 0.35,
        min_margin: float = 0.05,
    ) -> None:
        self.records = records
        self.bm25 = bm25
        self.dense = dense
        self.dense_factory = dense_factory
        self.min_score = min_score
        self.min_margin = min_margin
        self._local: dict[str, tuple[Retriever, Retriever, int]] = {}

    def search(
        self, query: str, shelf: CapabilityShelf, *, k: int = 5,
        hierarchical: bool = True,
    ) -> RetrievalRoute:
        if k < 1:
            raise ValueError("k must be >= 1")
        score = margin = None
        matcher_size = 0
        if hierarchical and shelf.bundles and query.strip():
            descriptors = [
                SkillRecord(
                    skill_id=b.bundle_id,
                    name=b.descriptor,
                    description="\n".join(
                        [b.use_when] + [metadata_text(self.records[s]) for s in b.skill_ids]
                    ),
                    body="", source_path="",
                )
                for b in shelf.bundles
            ]
            matcher_size = len(descriptors)
            matches = self.dense_factory(descriptors).search(query, k=2)
            if matches:
                best = matches[0]
                score = best.score
                margin = score - matches[1].score if len(matches) > 1 else score
                if score >= self.min_score and margin >= self.min_margin:
                    bundle_id = best.skill_id
                    if bundle_id not in self._local:
                        subset = [
                            r for r in self.records.values()
                            if default_group_key(r) == bundle_id
                        ]
                        self._local[bundle_id] = (
                            BM25Retriever([replace(r, body="") for r in subset]),
                            self.dense_factory(subset), len(subset),
                        )
                    bm25, dense, size = self._local[bundle_id]
                    return RetrievalRoute(
                        tuple(candidate_union({"bm25": bm25.search(query, k=k),
                                               "dense": dense.search(query, k=k)})),
                        "bundle-local", size, bundle_id, score, margin, matcher_size,
                    )
        return RetrievalRoute(
            tuple(candidate_union({"bm25": self.bm25.search(query, k=k),
                                   "dense": self.dense.search(query, k=k)})),
            "global", len(self.records), None, score, margin, matcher_size,
        )


def dense_factory_for(dense: DenseRetriever) -> RetrieverFactory:
    """Share the loaded encoder across descriptor and local indices."""
    return lambda records: DenseRetriever(
        records, model=dense.model, model_name=dense.model_name,
    )
