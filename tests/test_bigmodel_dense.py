from skill_control_plane.models import SkillRecord
from skill_control_plane.discovery.bigmodel import BigModelDenseRetriever


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


def test_embedding_does_not_inherit_chat_coding_endpoint(monkeypatch):
    from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
    monkeypatch.setenv('BIGMODEL_BASE_URL', 'https://open.bigmodel.cn/api/coding/paas/v4')
    monkeypatch.setenv('BIGMODEL_API_KEY', 'test-only-key')
    monkeypatch.delenv('BIGMODEL_EMBEDDING_BASE_URL', raising=False)
    monkeypatch.delenv('BIGMODEL_EMBEDDING_API_KEY', raising=False)
    client = BigModelEmbeddingClient()
    assert client.base_url == 'https://open.bigmodel.cn/api/paas/v4'
    assert client.api_key == 'test-only-key'


def test_embedding_specific_configuration_and_explicit_precedence(monkeypatch):
    from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
    monkeypatch.setenv('BIGMODEL_EMBEDDING_BASE_URL', 'https://embedding.example.test/v4/')
    monkeypatch.setenv('BIGMODEL_EMBEDDING_API_KEY', 'embedding-test-key')
    client = BigModelEmbeddingClient()
    assert client.base_url == 'https://embedding.example.test/v4'
    assert client.api_key == 'embedding-test-key'
    explicit = BigModelEmbeddingClient(base_url='https://explicit.example.test/v4', api_key='explicit-test-key')
    assert explicit.base_url == 'https://explicit.example.test/v4'
    assert explicit.api_key == 'explicit-test-key'


def test_paratera_and_embedding_model_configuration_take_precedence(monkeypatch):
    from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
    monkeypatch.setenv('PARATERA_API_KEY', 'paratera-test-key')
    monkeypatch.setenv('BIGMODEL_EMBEDDING_API_KEY', 'legacy-embedding-key')
    monkeypatch.setenv('BIGMODEL_EMBEDDING_MODEL', 'GLM-Embedding-3')
    monkeypatch.setenv('BIGMODEL_EMBEDDING_DIMENSIONS', '2048')
    client = BigModelEmbeddingClient()
    assert client.api_key == 'paratera-test-key'
    assert client.model == 'GLM-Embedding-3'
    assert client.dimensions == 2048


def test_dense_retriever_uses_configured_batch_size(monkeypatch):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setenv('BIGMODEL_EMBEDDING_BATCH_SIZE', '2')
    BigModelDenseRetriever([
        _skill(str(index), str(index), str(index)) for index in range(5)
    ], embed_batch=embed)
    assert calls == [2, 2, 1]
