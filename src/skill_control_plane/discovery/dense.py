from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from skill_control_plane.models import RetrievalCandidate, SkillRecord

DEFAULT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def metadata_text(skill: SkillRecord) -> str:
    """Compact semantic representation used for V0.1 Skill retrieval."""

    parts = [
        skill.name,
        skill.description,
        " ".join(skill.tags),
    ]
    return "\n".join(part for part in parts if part)


def dense_text(skill: SkillRecord) -> str:
    """Return the exact text bound to the production Dense index."""

    return skill.retrieval_representation or metadata_text(skill)


def _to_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _encode(model: Any, texts: Sequence[str]) -> list[list[float]]:
    vectors = model.encode(list(texts), normalize_embeddings=True)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [_to_vector(vector) for vector in vectors]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class DenseRetriever:
    """Dense semantic baseline over compact Skill metadata.

    Full Skill bodies are intentionally excluded. V0.1 retrieves candidate
    Skills from name + description + tags, then loads richer Skill content only
    after candidate selection.
    """

    def __init__(
        self,
        skills: Iterable[SkillRecord],
        *,
        model: Any | None = None,
        model_name: str = DEFAULT_DENSE_MODEL,
    ) -> None:
        self.skills = tuple(skills)

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'Dense retrieval requires the optional dependency: '
                    'pip install -e ".[dense]"'
                ) from exc
            model = SentenceTransformer(model_name)

        self.model = model
        self.model_name = model_name
        self._texts = tuple(metadata_text(skill) for skill in self.skills)
        self._embeddings = _encode(self.model, self._texts) if self.skills else []

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        if k <= 0 or not self.skills or not query.strip():
            return []

        query_vector = _encode(self.model, [query])[0]
        scored = [
            (_dot(query_vector, embedding), skill)
            for skill, embedding in zip(self.skills, self._embeddings, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].skill_id))

        return [
            RetrievalCandidate(
                skill_id=skill.skill_id,
                score=score,
                rank=rank,
                source_scores={"dense": score},
                evidence=(f"semantic_similarity: {score:.4f}",),
            )
            for rank, (score, skill) in enumerate(scored[:k], start=1)
        ]
