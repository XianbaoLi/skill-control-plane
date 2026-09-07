from skill_control_plane.evals import hard_negative_hit_rate, required_recall_at_k
from skill_control_plane.models import RetrievalCandidate


def candidates(*ids: str) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(skill_id=skill_id, score=1.0 / rank, rank=rank)
        for rank, skill_id in enumerate(ids, start=1)
    ]


def test_required_recall_at_k() -> None:
    assert required_recall_at_k(
        candidates("docker", "python-env", "git"),
        {"docker", "python-env"},
        k=2,
    ) == 1.0


def test_hard_negative_hit_rate() -> None:
    assert hard_negative_hit_rate(
        candidates("docker", "docker-security"),
        {"docker-security", "kubernetes"},
        k=2,
    ) == 0.5
