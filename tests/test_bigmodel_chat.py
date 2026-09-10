import json

import pytest

from skill_control_plane.corpus.bigmodel_chat import (
    BigModelChatClient,
    canonical_json_text,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_canonical_json_text_accepts_plain_and_fenced_json():
    plain = canonical_json_text('{"purpose":"debug","use_when":[]}')
    fenced = canonical_json_text(
        '```json\n{"purpose":"debug","use_when":[]}\n```'
    )
    assert json.loads(plain) == json.loads(fenced)
    assert json.loads(plain)["purpose"] == "debug"


def test_canonical_json_text_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        canonical_json_text('["not", "an", "object"]')


def test_bigmodel_chat_client_uses_openai_compatible_endpoint_and_returns_json():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"purpose":"debug Python","use_when":["traceback insufficient"],'
                                '"capabilities":["step through code"],'
                                '"lexical_cues":["breakpoint"]}'
                            )
                        }
                    }
                ]
            }
        )

    client = BigModelChatClient(
        api_key="test-key",
        base_url="https://example.test/v4/",
        model="glm-test",
        temperature=0.1,
        max_tokens=900,
        reasoning_effort="low",
        timeout=12,
        urlopen_fn=fake_urlopen,
    )
    result = json.loads(client("extract this skill"))

    assert captured["url"] == "https://example.test/v4/chat/completions"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "glm-test"
    assert captured["payload"]["max_tokens"] == 900
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["do_sample"] is False
    assert "temperature" not in captured["payload"]
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"][-1]["content"] == "extract this skill"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert result["purpose"] == "debug Python"


def test_glm_53_rejects_unsupported_reasoning_effort_locally():
    with pytest.raises(ValueError, match="only: low, high, max"):
        BigModelChatClient(
            api_key="test-key",
            base_url="https://example.test/v4",
            model="glm-5.3-flash",
            reasoning_effort="none",
            urlopen_fn=lambda *args, **kwargs: None,
        )


def test_chat_specific_configuration_precedes_legacy(monkeypatch):
    monkeypatch.setenv("BIGMODEL_CHAT_API_KEY", "chat-key")
    monkeypatch.setenv("BIGMODEL_API_KEY", "legacy-key")
    monkeypatch.setenv("BIGMODEL_CHAT_BASE_URL", "https://chat.example.test/v4/")
    monkeypatch.setenv("BIGMODEL_BASE_URL", "https://legacy.example.test/v4")
    monkeypatch.setenv("BIGMODEL_CHAT_MODEL", "GLM-5.3-Flash")

    client = BigModelChatClient()

    assert client.api_key == "chat-key"
    assert client.base_url == "https://chat.example.test/v4"
    assert client.model == "GLM-5.3-Flash"
