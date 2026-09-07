from __future__ import annotations

from collections.abc import Iterable, Sequence

from skill_control_plane.models import RetrievalCandidate


def _ids(candidates: Sequence[RetrievalCandidate], k: int | None = None) -> set[str]:
    selected = candidates if k is None else candidates[:k]
    return {candidate.skill_id for candidate in selected}


def required_recall_at_k(
    candidates: Sequence[RetrievalCandidate],
    required: Iterable[str],
    *,
    k: int,
) -> float:
    required_set = set(required)
    if not required_set:
        return 1.0
    return len(required_set & _ids(candidates, k)) / len(required_set)


def hard_negative_hit_rate(
    candidates: Sequence[RetrievalCandidate],
    hard_negatives: Iterable[str],
    *,
    k: int,
) -> float:
    negatives = set(hard_negatives)
    if not negatives:
        return 0.0
    return len(negatives & _ids(candidates, k)) / len(negatives)


def target_skill_accuracy(predicted: str | None, gold: str | None) -> float:
    return float(predicted == gold)
