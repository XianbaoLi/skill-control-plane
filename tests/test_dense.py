from __future__ import annotations

from skill_control_plane.models import SkillRecord
from skill_control_plane.retrieval import DenseRetriever, metadata_text


class FakeEmbeddingModel:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("notes" in lowered or "meeting" in lowered),
                    float("git" in lowered or "merge" in lowered),
                ]
            )
        return vectors


def _skill(
    skill_id: str,
    description: str,
    body: str = "",
    tags: tuple[str, ...] = (),
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        description=description,
        body=body,
        source_path=f"{skill_id}/SKILL.md",
        tags=tags,
    )


def test_dense_prefers_semantically_matching_metadata() -> None:
    skills = [
        _skill("meeting-notes", "Extract decisions and owners from meeting notes"),
        _skill("git-conflict", "Resolve Git merge conflicts"),
    ]

    results = DenseRetriever(skills, model=FakeEmbeddingModel()).search(
        "organize meeting notes",
        k=2,
    )

    assert results[0].skill_id == "meeting-notes"
    assert results[0].source_scores["dense"] > results[1].source_scores["dense"]
    assert results[0].evidence


def test_dense_metadata_excludes_body() -> None:
    skill = _skill(
        "apple-notes",
        "Manage Apple Notes",
        body="git merge conflict rebase",
        tags=("Notes", "Apple"),
    )

    text = metadata_text(skill)

    assert "Manage Apple Notes" in text
    assert "Notes Apple" in text
    assert "git merge conflict rebase" not in text
