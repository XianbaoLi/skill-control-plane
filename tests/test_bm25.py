from skill_control_plane.models import SkillRecord
from skill_control_plane.discovery import BM25Retriever


def _skill(skill_id: str, description: str, body: str) -> SkillRecord:
    return SkillRecord(skill_id, skill_id, description, body, f"{skill_id}/SKILL.md")


def test_bm25_prefers_matching_terms() -> None:
    skills = [
        _skill("python-env", "Python dependency troubleshooting", "pip uv conda install failure"),
        _skill("git-conflict", "Resolve Git merge conflicts", "rebase conflict cherry-pick"),
        _skill("docker-security", "Harden Docker images", "container image vulnerability"),
    ]
    results = BM25Retriever(skills).search("uv dependency install failed", k=2)

    assert results[0].skill_id == "python-env"
    assert results[0].source_scores["bm25"] > 0
    assert results[0].evidence
