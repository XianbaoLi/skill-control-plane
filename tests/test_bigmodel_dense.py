from skill_control_plane.models import SkillRecord
from skill_control_plane.retrieval.bigmodel import BigModelDenseRetriever


def _skill(skill_id: str, name: str, description: str) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        body="ignored body",
        source_path=f"{skill_id}/SKILL.md",
        tags=(),
    )


def test_bigmodel_dense_retriever_uses_cosine_and_metadata_only():
    skills = [
        _skill("debug", "debugging", "root cause analysis for failing tests"),
        _skill("calendar", "calendar", "create and update calendar events"),
    ]

    def embed(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "root cause" in lowered or "debug" in lowered:
                vectors.append([3.0, 0.0])
            elif "calendar" in lowered:
                vectors.append([0.0, 2.0])
            else:
                vectors.append([1.0, 1.0])
        return vectors

    retriever = BigModelDenseRetriever(skills, embed_batch=embed)
    results = retriever.search("root cause debugging", k=2)

    assert [item.skill_id for item in results] == ["debug", "calendar"]
    assert results[0].score == 1.0
    assert "ignored body" not in retriever._texts[0]
