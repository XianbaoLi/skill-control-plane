from __future__ import annotations

from skill_control_plane.evals import StageGold, StageTransitionGoldCase, diagnose_stage_retrieval
from skill_control_plane.models import RetrievalCandidate


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)
            for rank, skill_id in enumerate(self.results.get(query, [])[:k], start=1)
        ]


def _case() -> StageTransitionGoldCase:
    return StageTransitionGoldCase(
        case_id="ST",
        initial_task="initial task",
        stages=(
            StageGold(
                stage_id="S1",
                runtime_evidence=(),
                required_now=("base",),
                new_required=("base",),
                useful=(),
                hard_negative=(),
            ),
            StageGold(
                stage_id="S2",
                runtime_evidence=("raw evidence",),
                required_now=("target",),
                new_required=("target",),
                useful=(),
                hard_negative=(),
            ),
        ),
        rationale="",
        snapshot_id="snapshot",
    )


def test_diagnose_stage_retrieval_detects_anchoring_signal() -> None:
    # target is outside Top-3 when initial task remains in B,
    # but moves into Top-3 with raw evidence only in C.
    bm25 = FakeRetriever(
        {
            "initial task": ["a1", "a2", "a3", "target", "a5"],
            "initial task\nraw evidence": ["b1", "b2", "b3", "target", "b5"],
            "raw evidence": ["c1", "target", "c3", "c4", "c5"],
        }
    )
    dense = FakeRetriever(
        {
            "initial task": ["d1", "d2", "d3", "d4", "target"],
            "initial task\nraw evidence": ["e1", "e2", "e3", "e4", "target"],
            "raw evidence": ["f1", "f2", "target", "f4", "f5"],
        }
    )

    report = diagnose_stage_retrieval(
        [_case()],
        bm25=bm25,
        dense=dense,
        k=3,
        rank_depth=5,
    )

    assert report["diagnosed_target_count"] == 1
    row = report["targets"][0]
    assert row["B_initial_plus_raw"]["union_hit"] is False
    assert row["C_raw_only"]["union_hit"] is True
    assert row["diagnosis"] == "ANCHORING_SIGNAL"


def test_diagnose_stage_retrieval_detects_topk_budget_signal() -> None:
    bm25 = FakeRetriever(
        {
            "initial task": ["a1", "a2", "a3", "target", "a5", "a6"],
            "initial task\nraw evidence": ["b1", "b2", "b3", "target", "b5", "b6"],
            "raw evidence": ["c1", "c2", "c3", "c4", "target", "c6"],
        }
    )
    dense = FakeRetriever(
        {
            "initial task": ["d1", "d2", "d3", "d4", "d5", "target"],
            "initial task\nraw evidence": ["e1", "e2", "e3", "e4", "e5", "target"],
            "raw evidence": ["f1", "f2", "f3", "f4", "f5", "target"],
        }
    )

    report = diagnose_stage_retrieval(
        [_case()],
        bm25=bm25,
        dense=dense,
        k=3,
        rank_depth=6,
    )

    assert report["targets"][0]["diagnosis"] == "TOPK_BUDGET_SIGNAL"
