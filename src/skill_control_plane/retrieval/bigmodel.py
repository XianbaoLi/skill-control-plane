from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.retrieval.dense import metadata_text

DEFAULT_BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_BIGMODEL_EMBEDDING_MODEL = "embedding-3"
DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS = 2048
BIGMODEL_MAX_BATCH = 64
DEFAULT_DENSE_BATCH_SIZE = 8

EmbeddingBatchFn = Callable[[Sequence[str]], list[list[float]]]


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class BigModelEmbeddingClient:
    """Minimal HTTP client for BigModel's embedding API.

    Embedding configuration is separate from the chat Coding Plan endpoint.
    Credentials prefer PARATERA_API_KEY, with legacy BigModel fallbacks.
    No SDK dependency is required.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("PARATERA_API_KEY")
            or os.environ.get("BIGMODEL_EMBEDDING_API_KEY")
            or os.environ.get("BIGMODEL_API_KEY", "")
        )
        if not self.api_key:
            raise RuntimeError(
                "PARATERA_API_KEY, BIGMODEL_EMBEDDING_API_KEY, or "
                "BIGMODEL_API_KEY is required for the dense backend"
            )
        self.base_url = (
            base_url or os.environ.get("BIGMODEL_EMBEDDING_BASE_URL") or DEFAULT_BIGMODEL_BASE_URL
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("BIGMODEL_EMBEDDING_MODEL")
            or DEFAULT_BIGMODEL_EMBEDDING_MODEL
        )
        configured_dimensions = (
            str(dimensions)
            if dimensions is not None
            else os.environ.get("BIGMODEL_EMBEDDING_DIMENSIONS")
        )
        self.dimensions = int(
            configured_dimensions or DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS
        )
        if self.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.timeout = timeout

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > BIGMODEL_MAX_BATCH:
            raise ValueError(f"BigModel embedding batch exceeds {BIGMODEL_MAX_BATCH}")

        payload = json.dumps(
            {
                "model": self.model,
                "input": list(texts),
                "dimensions": self.dimensions,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"BigModel embedding request failed: {type(exc).__name__}"
            ) from exc

        rows = data.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("BigModel embedding response has no data array")
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not isinstance(v, list) for v in vectors):
            raise RuntimeError("BigModel embedding response shape mismatch")
        return [[float(value) for value in vector] for vector in vectors]


class BigModelDenseRetriever:
    """Commercial dense retrieval over the same compact Skill metadata baseline."""

    def __init__(
        self,
        skills: Iterable[SkillRecord],
        *,
        model_name: str = DEFAULT_BIGMODEL_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_BIGMODEL_EMBEDDING_DIMENSIONS,
        embed_batch: EmbeddingBatchFn | None = None,
    ) -> None:
        self.skills = tuple(skills)
        self.model_name = model_name
        self.dimensions = dimensions
        self._texts = tuple(metadata_text(skill) for skill in self.skills)
        self._embed_batch = embed_batch or BigModelEmbeddingClient(
            model=model_name,
            dimensions=dimensions,
        )
        self.batch_size = int(
            os.environ.get("BIGMODEL_EMBEDDING_BATCH_SIZE", DEFAULT_DENSE_BATCH_SIZE)
        )
        if not 1 <= self.batch_size <= BIGMODEL_MAX_BATCH:
            raise ValueError(
                f"embedding batch size must be between 1 and {BIGMODEL_MAX_BATCH}"
            )
        self._embeddings = self._encode_many(self._texts)

    def _encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return [_normalize(vector) for vector in vectors]

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        if k <= 0 or not self.skills or not query.strip():
            return []

        query_vector = _normalize(self._embed_batch([query])[0])
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
