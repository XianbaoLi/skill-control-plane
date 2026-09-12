from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from skill_control_plane.discovery import (
    DENSE_INDEX_VERSION,
    BigModelDenseRetriever,
    DenseIndexError,
    build_dense_index,
    load_dense_index,
    load_precomputed_dense_retriever,
    write_dense_index,
)
from skill_control_plane.discovery.cards import RetrievalCard
from skill_control_plane.discovery.cards import load_retrieval_cards
from skill_control_plane.discovery.discovery import indexed_skill_records
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore


def _skills() -> tuple[SkillRecord, ...]:
    return (
        SkillRecord(
            "debug", "debugging", "root cause analysis for failing tests",
            "DEBUG BODY", "/debug",
        ),
        SkillRecord(
            "calendar", "calendar", "create and update calendar events",
            "CALENDAR BODY", "/calendar",
        ),
    )


def _store() -> SkillStore:
    return SkillStore(_skills())


def _write_cards(store: SkillStore, path: Path) -> None:
    rows = []
    for skill in store:
        card = RetrievalCard(
            skill_id=skill.skill_id,
            source_content_hash=skill.content_hash,
            purpose=f"{skill.name} work",
            use_when=(f"when {skill.description}", f"for {skill.name} work"),
            capabilities=(skill.description,),
            lexical_cues=(skill.name, "skill", "work"),
        )
        rows.append({
            "skill_id": card.skill_id,
            "source_content_hash": card.source_content_hash,
            "purpose": card.purpose,
            "use_when": list(card.use_when),
            "capabilities": list(card.capabilities),
            "lexical_cues": list(card.lexical_cues),
            "version": card.version,
        })
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


class CountingEmbedding:
    model = "embedding-3"
    dimensions = 2

    def __init__(self) -> None:
        self.corpus_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def __call__(self, texts):
        if len(texts) > 1:
            self.corpus_calls.append(tuple(texts))
        else:
            self.query_calls.append(texts[0])
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "root cause" in lowered:
                vectors.append([3.0, 0.0])
            elif "calendar" in lowered:
                vectors.append([0.0, 2.0])
            else:
                vectors.append([1.0, 1.0])
        return vectors


def _build(tmp_path: Path):
    for skill in _skills():
        skill_dir = tmp_path / skill.skill_id
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {skill.skill_id}\n"
            f"description: {skill.description}\n"
            "---\n"
            f"{skill.body}\n",
            encoding="utf-8",
        )
    store = SkillStore.from_tree(tmp_path)
    card_path = tmp_path / "cards.jsonl"
    _write_cards(store, card_path)
    embedding = CountingEmbedding()
    index = build_dense_index(
        skill_root=tmp_path,
        retrieval_cards=card_path,
        embedding_client=embedding,
    )
    index_path = write_dense_index(index, tmp_path / "dense-index.json")
    return store, card_path, index, index_path, embedding


def test_offline_build_load_and_query_round_trip(tmp_path):
    store, card_path, index, index_path, embedding = _build(tmp_path)
    indexed_skills = indexed_skill_records(
        store, load_retrieval_cards(card_path)
    )

    assert index.version == DENSE_INDEX_VERSION
    assert len(index.records) == 2
    assert len(embedding.corpus_calls) == 1
    assert len(embedding.corpus_calls[0]) == 2
    assert load_dense_index(
        index_path, indexed_skills,
        embedding_model=embedding.model, dimensions=embedding.dimensions,
    ) == index

    retriever = load_precomputed_dense_retriever(
        index_path, indexed_skills,
        embedding_model=embedding.model,
        dimensions=embedding.dimensions,
        embed_batch=embedding,
    )
    assert embedding.query_calls == []
    results = retriever.search("root cause debugging", k=2)

    assert [result.skill_id for result in results] == ["debug", "calendar"]
    assert results[0].score == pytest.approx(1.0)
    assert embedding.query_calls == ["root cause debugging"]
    assert len(embedding.corpus_calls) == 1


def test_ranking_matches_existing_bigmodel_dense_retriever(tmp_path):
    store, card_path, _, index_path, embedding = _build(tmp_path)
    legacy = BigModelDenseRetriever(
        store,
        model_name=embedding.model,
        dimensions=embedding.dimensions,
        embed_batch=embedding,
    )
    precomputed = load_precomputed_dense_retriever(
        index_path, indexed_skill_records(
            store, load_retrieval_cards(card_path)
        ),
        embedding_model=embedding.model,
        dimensions=embedding.dimensions,
        embed_batch=embedding,
    )

    for query in ["root cause debugging", "calendar update events"]:
        assert legacy.search(query, k=2) == precomputed.search(query, k=2)


def test_missing_index_fails_closed(tmp_path):
    _, _, _, _, embedding = _build(tmp_path)
    with pytest.raises(DenseIndexError, match="cannot load Dense Index"):
        load_precomputed_dense_retriever(
            tmp_path / "missing.json", _store(),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


@pytest.mark.parametrize("raw", ["{", "[}"])
def test_malformed_json_fails_closed(tmp_path, raw):
    _, _, _, _, embedding = _build(tmp_path)
    path = tmp_path / "malformed.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(DenseIndexError, match="cannot load Dense Index"):
        load_precomputed_dense_retriever(
            path, _store(),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


def test_invalid_index_shape_fails_closed(tmp_path):
    _, _, _, _, embedding = _build(tmp_path)
    path = tmp_path / "invalid-shape.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DenseIndexError, match="must be a JSON object"):
        load_precomputed_dense_retriever(
            path, _store(),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


def _write_mutated_index(source: Path, path: Path, mutation):
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_unsupported_version_fails_closed(tmp_path):
    _, _, index, source, embedding = _build(tmp_path)
    path = _write_mutated_index(
        source, tmp_path / "wrong-version.json",
        lambda payload: payload.update(version="dense-index-v0"),
    )
    with pytest.raises(DenseIndexError, match="unsupported Dense Index version"):
        load_precomputed_dense_retriever(
            path, _store(),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )
    assert index.version == DENSE_INDEX_VERSION


def test_missing_skill_fails_closed(tmp_path):
    _, _, _, source, embedding = _build(tmp_path)
    with pytest.raises(DenseIndexError, match="extra=\\['calendar'\\]"):
        load_precomputed_dense_retriever(
            source, _store().values()[:1],
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


def test_extra_skill_fails_closed(tmp_path):
    store, card_path, _, _, embedding = _build(tmp_path)
    indexed_skills = indexed_skill_records(
        store, load_retrieval_cards(card_path)
    )
    one_skill_index = replace(
        load_dense_index(
            tmp_path / "dense-index.json", indexed_skills,
            embedding_model=embedding.model, dimensions=embedding.dimensions,
        ),
        records=(),
    )
    path = write_dense_index(one_skill_index, tmp_path / "extra.json")
    with pytest.raises(DenseIndexError, match="missing=\\['calendar', 'debug'\\]"):
        load_precomputed_dense_retriever(
            path, indexed_skills,
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


def test_stale_representation_hash_fails_closed(tmp_path):
    _, _, _, source, embedding = _build(tmp_path)
    def mutate(payload):
        payload["records"][0]["representation_hash"] = "0" * 64
    path = _write_mutated_index(source, tmp_path / "stale.json", mutate)
    with pytest.raises(DenseIndexError, match="representation is stale"):
        load_precomputed_dense_retriever(
            path, _store(),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


@pytest.mark.parametrize("model,dimensions", [
    ("wrong-model", 2),
    ("embedding-3", 3),
])
def test_wrong_embedding_model_or_dimensions_fail_closed(
    tmp_path, model, dimensions,
):
    _, _, _, source, _ = _build(tmp_path)
    def mutate(payload):
        payload["embedding_model"] = model
        payload["dimensions"] = dimensions
    path = _write_mutated_index(source, tmp_path / "wrong.json", mutate)
    with pytest.raises(DenseIndexError, match="mismatch"):
        load_precomputed_dense_retriever(
            path, _store(),
            embedding_model="embedding-3",
            dimensions=2,
            embed_batch=CountingEmbedding(),
        )


def test_vector_dimension_mismatch_fails_closed(tmp_path):
    store, card_path, _, source, embedding = _build(tmp_path)
    def mutate(payload):
        payload["records"][0]["embedding"].pop()
    path = _write_mutated_index(source, tmp_path / "wrong-shape.json", mutate)
    with pytest.raises(DenseIndexError, match="vector dimension mismatch"):
        load_precomputed_dense_retriever(
            path, indexed_skill_records(
                store, load_retrieval_cards(card_path)
            ),
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )


def test_retrieval_card_change_rejects_old_index(tmp_path):
    store, card_path, _, index_path, embedding = _build(tmp_path)
    for skill in store:
        card = RetrievalCard(
            skill_id=skill.skill_id,
            source_content_hash=skill.content_hash,
            purpose="changed purpose",
            use_when=("when changed", "for changed work"),
            capabilities=("changed capability",),
            lexical_cues=(skill.name, "changed", "work"),
        )
        row = {
            "skill_id": card.skill_id,
            "source_content_hash": card.source_content_hash,
            "purpose": card.purpose,
            "use_when": list(card.use_when),
            "capabilities": list(card.capabilities),
            "lexical_cues": list(card.lexical_cues),
            "version": card.version,
        }
        card_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(DenseIndexError, match="representation is stale"):
        load_precomputed_dense_retriever(
            index_path, store,
            embedding_model=embedding.model,
            dimensions=embedding.dimensions,
            embed_batch=embedding,
        )
