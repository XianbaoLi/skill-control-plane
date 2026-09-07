from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.retrieval import reciprocal_rank_fusion


def c(skill_id: str, score: float, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(skill_id=skill_id, score=score, rank=rank)


def test_rrf_rewards_cross_retriever_agreement() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [c("a", 10.0, 1), c("b", 8.0, 2)],
            "dense": [c("b", 0.9, 1), c("c", 0.8, 2)],
        },
        limit=3,
    )

    assert fused[0].skill_id == "b"
    assert set(fused[0].source_scores) == {"bm25", "dense"}
