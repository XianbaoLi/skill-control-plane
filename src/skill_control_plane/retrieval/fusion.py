from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from skill_control_plane.models import RetrievalCandidate


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalCandidate]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievalCandidate]:
    """Fuse ranked lists while preserving source scores and evidence."""

    fused: defaultdict[str, float] = defaultdict(float)
    source_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    evidence: defaultdict[str, list[str]] = defaultdict(list)

    for source, candidates in rankings.items():
        for position, candidate in enumerate(candidates, start=1):
            fused[candidate.skill_id] += 1.0 / (k + position)
            source_scores[candidate.skill_id][source] = candidate.score
            evidence[candidate.skill_id].extend(candidate.evidence)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]

    return [
        RetrievalCandidate(
            skill_id=skill_id,
            score=score,
            rank=rank,
            source_scores=source_scores[skill_id],
            evidence=tuple(dict.fromkeys(evidence[skill_id])),
        )
        for rank, (skill_id, score) in enumerate(ordered, start=1)
    ]
