"""Versioned precomputed BigModel Dense corpus vectors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill_control_plane.discovery.bigmodel import (
    BIGMODEL_MAX_BATCH,
    DEFAULT_DENSE_BATCH_SIZE,
    BigModelEmbeddingClient,
    EmbeddingBatchFn,
)
from skill_control_plane.discovery.cards import (
    load_retrieval_cards,
    validate_retrieval_cards,
)
from skill_control_plane.discovery.dense import dense_text
from skill_control_plane.discovery.discovery import indexed_skill_records
from skill_control_plane.models import RetrievalCandidate, SkillRecord
from skill_control_plane.registry import SkillStore

DENSE_INDEX_VERSION = "dense-index-v1"
DENSE_INDEX_FIELDS = {"version", "embedding_model", "dimensions", "records"}
DENSE_INDEX_RECORD_FIELDS = {"skill_id", "representation_hash", "embedding"}


class DenseIndexError(ValueError):
    """The Dense Index is malformed or incompatible with the current corpus."""


@dataclass(frozen=True, slots=True)
class DenseIndexRecord:
    skill_id: str
    representation_hash: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DenseIndexV1:
    version: str
    embedding_model: str
    dimensions: int
    records: tuple[DenseIndexRecord, ...]


def representation_hash(text: str) -> str:
    """Return the exact representation identity stored in DenseIndexV1."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DenseIndexError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DenseIndexError(f"{label} must be a positive integer")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DenseIndexError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def decode_dense_index(value: Any) -> DenseIndexV1:
    if not isinstance(value, dict):
        raise DenseIndexError("Dense Index must be a JSON object")
    if set(value) != DENSE_INDEX_FIELDS:
        raise DenseIndexError("Dense Index has invalid fields")
    if value["version"] != DENSE_INDEX_VERSION:
        raise DenseIndexError(
            f"unsupported Dense Index version: {value['version']!r}"
        )
    embedding_model = _string(value["embedding_model"], "embedding_model")
    dimensions = _integer(value["dimensions"], "dimensions")
    raw_records = value["records"]
    if not isinstance(raw_records, list):
        raise DenseIndexError("Dense Index records must be an array")

    records: list[DenseIndexRecord] = []
    skill_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise DenseIndexError(f"Dense Index record {index} must be an object")
        if set(raw_record) != DENSE_INDEX_RECORD_FIELDS:
            raise DenseIndexError(f"Dense Index record {index} has invalid fields")
        skill_id = _string(raw_record["skill_id"], f"record {index} skill_id")
        if skill_id in skill_ids:
            raise DenseIndexError(f"duplicate Dense Index skill_id: {skill_id}")
        skill_ids.add(skill_id)
        record_hash = _string(
            raw_record["representation_hash"],
            f"record {index} representation_hash",
        )
        embedding_value = raw_record["embedding"]
        if not isinstance(embedding_value, list):
            raise DenseIndexError(f"record {index} embedding must be an array")
        embedding: list[float] = []
        for vector_index, item in enumerate(embedding_value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise DenseIndexError(
                    f"record {index} embedding item {vector_index} must be a number"
                )
            number = float(item)
            if not math.isfinite(number):
                raise DenseIndexError(
                    f"record {index} embedding item {vector_index} must be finite"
                )
            embedding.append(number)
        records.append(DenseIndexRecord(
            skill_id=skill_id,
            representation_hash=record_hash,
            embedding=tuple(embedding),
        ))
    return DenseIndexV1(
        version=DENSE_INDEX_VERSION,
        embedding_model=embedding_model,
        dimensions=dimensions,
        records=tuple(records),
    )


def load_dense_index(
    path: str | Path,
    skills: Iterable[SkillRecord],
    *,
    embedding_model: str,
    dimensions: int,
) -> DenseIndexV1:
    """Load and validate an artifact against exact Dense representations."""

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, DenseIndexError) as exc:
        raise DenseIndexError(f"cannot load Dense Index: {exc}") from exc
    index = decode_dense_index(raw)
    if index.embedding_model != embedding_model:
        raise DenseIndexError(
            "Dense Index embedding model mismatch: "
            f"index={index.embedding_model!r}, runtime={embedding_model!r}"
        )
    if index.dimensions != dimensions:
        raise DenseIndexError(
            "Dense Index dimensions mismatch: "
            f"index={index.dimensions}, runtime={dimensions}"
        )

    expected = {skill.skill_id: dense_text(skill) for skill in skills}
    actual = {record.skill_id: record for record in index.records}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise DenseIndexError(
            f"Dense Index Skill mismatch: missing={missing} extra={extra}"
        )
    for skill_id, text in expected.items():
        record = actual[skill_id]
        if len(record.embedding) != dimensions:
            raise DenseIndexError(
                f"Dense Index vector dimension mismatch for {skill_id}: "
                f"expected={dimensions}, actual={len(record.embedding)}"
            )
        if record.representation_hash != representation_hash(text):
            raise DenseIndexError(
                f"Dense Index representation is stale for Skill: {skill_id}"
            )
    return index


class PrecomputedDenseRetriever:
    """Dot-product Dense search over vectors loaded from DenseIndexV1."""

    def __init__(
        self,
        skills: Iterable[SkillRecord],
        *,
        model_name: str,
        dimensions: int,
        embeddings: Iterable[Sequence[float]],
        embed_batch: EmbeddingBatchFn,
    ) -> None:
        self.skills = tuple(skills)
        self.model_name = model_name
        self.dimensions = dimensions
        self._texts = tuple(dense_text(skill) for skill in self.skills)
        self._embeddings = tuple(
            _normalize(embedding) for embedding in embeddings
        )
        self._embed_batch = embed_batch
        if len(self.skills) != len(self._embeddings):
            raise DenseIndexError("precomputed vectors must be one per Skill")
        if any(len(embedding) != dimensions for embedding in self._embeddings):
            raise DenseIndexError("precomputed vector dimension mismatch")

    @classmethod
    def from_dense_index(
        cls,
        index: DenseIndexV1,
        skills: Iterable[SkillRecord],
        *,
        embed_batch: EmbeddingBatchFn,
    ) -> "PrecomputedDenseRetriever":
        vectors = {
            record.skill_id: record.embedding for record in index.records
        }
        return cls(
            skills,
            model_name=index.embedding_model,
            dimensions=index.dimensions,
            embeddings=(vectors[skill.skill_id] for skill in skills),
            embed_batch=embed_batch,
        )

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        if k <= 0 or not self.skills or not query.strip():
            return []

        query_vector = _normalize(self._embed_batch([query])[0])
        scored = [
            (_dot(query_vector, embedding), skill)
            for skill, embedding in zip(
                self.skills, self._embeddings, strict=True
            )
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


def load_precomputed_dense_retriever(
    path: str | Path,
    skills: Iterable[SkillRecord],
    *,
    embedding_model: str,
    dimensions: int,
    embed_batch: EmbeddingBatchFn,
) -> PrecomputedDenseRetriever:
    index = load_dense_index(
        path,
        skills,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    return PrecomputedDenseRetriever.from_dense_index(
        index, skills, embed_batch=embed_batch
    )


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_dense_index(
    *,
    skill_root: str | Path,
    retrieval_cards: str | Path,
    embedding_client: BigModelEmbeddingClient,
) -> DenseIndexV1:
    """Build corpus embeddings once using production Discovery representations."""

    store = SkillStore.from_tree(skill_root)
    cards = load_retrieval_cards(retrieval_cards)
    validate_retrieval_cards(store, cards)
    records = indexed_skill_records(store, cards)
    batch_size = int(
        os.environ.get("BIGMODEL_EMBEDDING_BATCH_SIZE", DEFAULT_DENSE_BATCH_SIZE)
    )
    if not 1 <= batch_size <= BIGMODEL_MAX_BATCH:
        raise DenseIndexError(
            f"embedding batch size must be between 1 and {BIGMODEL_MAX_BATCH}"
        )

    embeddings: list[tuple[float, ...]] = []
    for start in range(0, len(records), batch_size):
        rows = records[start:start + batch_size]
        vectors = embedding_client([dense_text(record) for record in rows])
        embeddings.extend(tuple(vector) for vector in vectors)
    return DenseIndexV1(
        version=DENSE_INDEX_VERSION,
        embedding_model=embedding_client.model,
        dimensions=embedding_client.dimensions,
        records=tuple(
            DenseIndexRecord(
                skill_id=record.skill_id,
                representation_hash=representation_hash(dense_text(record)),
                embedding=embedding,
            )
            for record, embedding in zip(records, embeddings, strict=True)
        ),
    )


def write_dense_index(index: DenseIndexV1, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": index.version,
        "embedding_model": index.embedding_model,
        "dimensions": index.dimensions,
        "records": [
            {
                "skill_id": record.skill_id,
                "representation_hash": record.representation_hash,
                "embedding": list(record.embedding),
            }
            for record in index.records
        ],
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output
