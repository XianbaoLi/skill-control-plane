from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.retrieval import candidate_union, reciprocal_rank_fusion


def c(skill_id: str, score: float, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        skill_id=skill_id,
        score=score,
        rank=rank,
        evidence=(f"evidence:{skill_id}",),
    )


def test_candidate_union_deduplicates_and_preserves_sources() -> None:
    union = candidate_union(
        {
            "bm25": [c("a", 10.0, 1), c("b", 8.0, 2)],
            "dense": [c("b", 0.9, 1), c("c", 0.8, 2)],
        }
    )

    assert [candidate.skill_id for candidate in union] == ["a", "b", "c"]
    assert set(union[1].source_scores) == {"bm25", "dense"}
    assert union[1].score == 0.0


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
