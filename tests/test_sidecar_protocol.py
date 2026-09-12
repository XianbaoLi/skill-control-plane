from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path

import pytest

from skill_control_plane import (
    CapabilityDecision,
    CoverageClaim,
    SkillControlPlane,
    StateSnapshotBundleV1,
    StateSnapshotMemberV1,
    StateSnapshotV1,
)
from skill_control_plane.sidecar import (
    PROTOCOL_VERSION,
    SUPPORTED_METHODS,
    SidecarServer,
    build_sidecar_server,
)
from skill_control_plane.sidecar.protocol import ERROR_CODES
from skill_control_plane.registry import SkillStore
from tests.support import complete_cards, full_discovery


def _store():
    from skill_control_plane.models import SkillRecord
    from skill_control_plane.registry import SkillStore

    return SkillStore([
        SkillRecord("ocr", "OCR", "extract scanned text", "OCR BODY", "/ocr"),
        SkillRecord("slides", "Slides", "create slides", "SLIDE BODY", "/slides"),
    ])


@pytest.fixture
def server():
    store = _store()
    return SidecarServer(
        SkillControlPlane(store, discovery=full_discovery(store)),
        diagnostics=StringIO(),
    )


def _request(method, params=None, id="req-1"):
    return json.dumps({"id": id, "method": method, "params": params or {}})


def _call(server, method, params=None, id="req-1"):
    response = server.handle_line(_request(method, params, id))
    assert response is not None
    return response


def _search(server, need="create slides", *, id="search"):
    response = _call(server, "search_capability", {"need": need}, id)
    assert response["ok"] is True
    return response["result"]


def _create(server, id="create"):
    _call(server, "begin_turn")
    _search(server)
    return _call(server, "apply_capability", {
        "action": "CREATE",
        "skill_ids": ["slides"],
        "reason": "reusable presentation work",
        "purpose": "Presentation work",
        "coverage": [{"need": "slides", "covered_by": "skill:slides"}],
    }, id)["result"]


def test_protocol_constants_and_handshake(server):
    response = _call(server, "handshake")
    assert response["id"] == "req-1" and response["ok"] is True
    assert response["result"]["protocol_version"] == PROTOCOL_VERSION == (
        "sidecar-protocol-v0.1"
    )
    assert response["result"]["supported_methods"] == list(SUPPORTED_METHODS)
    readiness = response["result"]["readiness"]
    assert readiness["ready"] is True
    assert readiness["probe_ready"] is True
    assert set(ERROR_CODES) == {
        "INVALID_REQUEST", "INVALID_PARAMS", "UNKNOWN_METHOD",
        "CORE_VALIDATION_ERROR", "NOT_READY", "INTERNAL_ERROR",
    }


def test_sequential_requests_and_context_snapshot(server):
    begin = _call(server, "begin_turn")
    assert begin["ok"] is True and begin["result"] == {"turn_started": True}
    _search(server)
    response = _call(server, "context_snapshot", {"compact": True})
    snapshot = response["result"]
    assert snapshot["pending_query"] == "create slides"
    assert snapshot["search_count"] == 1
    assert snapshot["remaining_search_budget"] == 2
    assert response is not begin


@pytest.mark.parametrize("raw", ['{"id":"x"', '"text"', "[]", "", "null"])
def test_malformed_and_non_object_lines(server, raw):
    response = server.handle_line(raw)
    assert response == {
        "id": None, "ok": False,
        "error": {"code": "INVALID_REQUEST", "message": response["error"]["message"]},
    }


def test_duplicate_and_unknown_envelope_fields(server):
    response = server.handle_line(
        '{"id":"a","id":"b","method":"handshake","params":{}}'
    )
    assert response["error"]["code"] == "INVALID_REQUEST"
    response = _call(server, "handshake", {"extra": True})
    assert response["error"]["code"] == "INVALID_PARAMS"


@pytest.mark.parametrize("payload", [
    {"method": "handshake", "params": {}},
    {"id": "", "method": "handshake", "params": {}},
    {"id": 1, "method": "handshake", "params": {}},
    {"id": "x", "method": "", "params": {}},
    {"id": "x", "method": "handshake"},
    {"id": "x", "method": "handshake", "params": []},
])
def test_invalid_envelope(server, payload):
    response = server.handle_line(json.dumps(payload))
    assert response["ok"] is False
    assert response["error"]["code"] in {"INVALID_REQUEST", "INVALID_PARAMS"}
    if isinstance(payload.get("id"), str) and payload["id"]:
        assert response["id"] == payload["id"]
    else:
        assert response["id"] is None


def test_unknown_method(server):
    response = _call(server, "not_a_method")
    assert response["error"]["code"] == "UNKNOWN_METHOD"


def test_begin_and_end_turn_reset_pending(server):
    _call(server, "begin_turn")
    _search(server)
    _call(server, "end_turn")
    context = _call(server, "context_snapshot")["result"]
    assert context["pending_query"] is None
    assert context["search_count"] == 1


def test_search_capability_serializes_core_dto(server):
    result = _search(server)
    assert result["query"] == "create slides"
    assert result["candidates"][0]["skill_id"] == "slides"
    assert result["retrieval_trace"]["backend"] == "rrf"
    assert result["search_control"]["remaining_search_budget"] == 2


@pytest.mark.parametrize("action,skill_ids", [
    ("CREATE", ["slides"]),
    ("EXTEND", ["ocr"]),
])
def test_apply_capability_actions(server, action, skill_ids):
    if action == "CREATE":
        _call(server, "begin_turn")
        _search(server)
        params = {
            "action": action,
            "skill_ids": skill_ids,
            "reason": "reusable work",
            "purpose": "Presentation work",
            "coverage": [{"need": "slides", "covered_by": "skill:slides"}],
        }
    else:
        created = _create(server)
        bundle_id = created["affected_bundle_id"]
        _call(server, "begin_turn")
        _search(server, "extract scanned text")
        params = {
            "action": action,
            "skill_ids": skill_ids,
            "reason": "add related work",
            "target_bundle_id": bundle_id,
            "coverage": [{"need": "scan", "covered_by": "skill:ocr"}],
        }
    response = _call(server, "apply_capability", params)
    assert response["ok"] is True
    assert response["result"]["action"] == action


def test_apply_capability_direct(server):
    _create(server)
    bundle_id = _call(server, "context_snapshot")["result"][
        "maintained_bundles"][0]["bundle_id"]
    _call(server, "begin_turn")
    _search(server, "extract scanned text")
    response = _call(server, "apply_capability", {
        "action": "DIRECT",
        "skill_ids": ["ocr"],
        "reason": "one-turn operation",
        "target_bundle_id": bundle_id,
        "coverage": [{"need": "scan", "covered_by": "skill:ocr"}],
    })
    assert response["ok"] is True
    assert response["result"]["selected_skill_ids"] == ["ocr"]


def test_apply_requires_pending_candidates(server):
    response = _call(server, "apply_capability", {
        "action": "CREATE",
        "skill_ids": ["slides"],
        "reason": "no search",
        "purpose": "No search",
        "coverage": [{"need": "slides", "covered_by": "skill:slides"}],
    })
    assert response["error"]["code"] == "CORE_VALIDATION_ERROR"
    assert server._diagnostics.getvalue().startswith("sidecar: core validation failed")


def test_apply_rejects_unknown_fields_and_invalid_coverage(server):
    response = _call(server, "apply_capability", {
        "action": "CREATE", "skill_ids": ["slides"], "reason": "x",
        "purpose": "x", "coverage": [], "unknown": True,
    })
    assert response["error"]["code"] == "INVALID_PARAMS"
    response = _call(server, "apply_capability", {
        "action": "CREATE", "skill_ids": ["slides"], "reason": "x",
        "purpose": "x", "coverage": [{"need": "x"}],
    })
    assert response["error"]["code"] == "INVALID_PARAMS"


def test_body_lifecycle_and_audit(server):
    created = _create(server)
    loaded = _call(server, "load_skill_body", {"skill_id": "slides"})["result"]
    assert loaded == {
        "status": "already_resident",
        "skill_id": "slides",
        "bundle_ids": [created["affected_bundle_id"]],
        "body": None,
    }
    evicted = _call(server, "mark_skill_body_evicted", {"skill_id": "slides"})
    assert evicted["result"]["body_state"] == "evicted"
    reloaded = _call(server, "load_skill_body", {"skill_id": "slides"})["result"]
    assert reloaded["status"] == "loaded" and reloaded["body"] == "SLIDE BODY"
    _call(server, "mark_all_skill_bodies_evicted")
    audit = _call(server, "turn_audit")["result"]
    assert audit["body_load_count"] == 1
    assert audit["session"]["apply_count"] == 1


def test_export_state_is_json_and_restore_round_trips(server):
    _create(server)
    _call(server, "mark_skill_body_evicted", {"skill_id": "slides"})
    exported = _call(server, "export_state")["result"]
    json.dumps(exported)
    restored = server.handle_line(_request(
        "restore_state", exported, "restore"
    ))
    assert restored["ok"] is True
    assert _call(server, "export_state")["result"] == exported


def test_restore_rejects_malformed_json_shape(server):
    exported = _call(server, "export_state")["result"]
    malformed = deepcopy(exported)
    malformed["bundles"] = [{"bundle_id": "bad"}]
    response = server.handle_line(_request("restore_state", malformed))
    assert response["error"]["code"] == "INVALID_PARAMS"


def test_failed_restore_does_not_mutate_state(server):
    _create(server)
    before = _call(server, "context_snapshot")["result"]
    exported = _call(server, "export_state")["result"]
    invalid = deepcopy(exported)
    invalid["store_fingerprint"] = "not-a-fingerprint"
    response = server.handle_line(_request("restore_state", invalid))
    assert response["error"]["code"] == "CORE_VALIDATION_ERROR"
    assert _call(server, "context_snapshot")["result"] == before


def test_unready_runtime_methods_fail_closed(diagnostics=StringIO()):
    server = SidecarServer(
        None, startup_error="missing credential", diagnostics=diagnostics
    )
    readiness = _call(server, "handshake")["result"]["readiness"]
    assert readiness["ready"] is False
    assert "missing credential" in readiness["errors"]
    assert readiness["probe_executed"] is False
    for method, params in [
        ("begin_turn", {}),
        ("search_capability", {"need": "slides"}),
        ("export_state", {}),
        (
            "restore_state",
            {
                "version": "state-snapshot-v1",
                "store_fingerprint": "x",
                "bundles": [],
            },
        ),
    ]:
        response = _call(server, method, params)
        assert response["error"]["code"] == "NOT_READY"
    assert "Traceback" not in diagnostics.getvalue()


def test_shutdown_response_and_eof(server):
    responses = []
    code = server.serve([
        _request("handshake", id="1"),
        _request("shutdown", id="2"),
        _request("handshake", id="3"),
    ], responses.append)
    assert code == 0
    assert len(responses) == 2
    assert responses[1] == json.dumps(
        {"id": "2", "ok": True, "result": {"shutdown": True}}, ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    assert json.loads(responses[0])["ok"] is True


def test_serve_eof_and_stdio_contract(server):
    input_stream = StringIO(_request("handshake", id="only") + "\n")
    output_stream = StringIO()
    assert server.serve_stdio(input_stream, output_stream) == 0
    lines = output_stream.getvalue().splitlines()
    assert len(lines) == 1
    response = json.loads(lines[0])
    assert response["ok"] is True
    assert "Traceback" not in output_stream.getvalue()


def test_invalid_utf8_returns_structured_error(server):
    output = BytesIO()
    code = server.serve_binary(
        iter([b'{"id":"\xff","method":"handshake","params":{}}\n']),
        output,
    )
    assert code == 0
    response = json.loads(output.getvalue().decode("utf-8"))
    assert response["error"]["code"] == "INVALID_REQUEST"


def test_unexpected_error_is_internal_and_stdout_is_clean(server, monkeypatch):
    def explode():
        raise RuntimeError("hidden provider details")

    monkeypatch.setattr(server._control_plane, "readiness", explode)
    diagnostics = StringIO()
    server._diagnostics = diagnostics
    response = _call(server, "handshake")
    assert response["error"] == {
        "code": "INTERNAL_ERROR", "message": "internal sidecar error"
    }
    assert "RuntimeError" not in response["error"]["message"]
    assert "Traceback" in diagnostics.getvalue()


def test_readiness_and_handshake_do_not_pollute_turn(server):
    _call(server, "begin_turn")
    _search(server)
    before = _call(server, "context_snapshot")["result"]
    before_audit = _call(server, "turn_audit")["result"]
    _call(server, "handshake")
    assert _call(server, "context_snapshot")["result"] == before
    assert _call(server, "turn_audit")["result"] == before_audit


def _write_cards(store, path):
    path.write_text("\n".join(
        json.dumps({
            "skill_id": skill_id,
            "source_content_hash": card.source_content_hash,
            "purpose": card.purpose,
            "use_when": [*card.use_when, f"when {card.purpose}",
                         f"for {card.purpose}"],
            "capabilities": list(card.capabilities) or [card.purpose],
            "lexical_cues": [*card.lexical_cues, "skill", "capability"],
            "version": card.version,
        }) for skill_id, card in complete_cards(store).items()
    ), encoding="utf-8")


def test_production_startup_constructs_offline_dense_factory(tmp_path):
    for name, description in [
        ("ocr", "extract scanned text"),
        ("slides", "create slides"),
    ]:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nBODY\n",
            encoding="utf-8",
        )
    store = SkillStore.from_tree(tmp_path)
    card_path = tmp_path / "cards.jsonl"
    _write_cards(store, card_path)

    class OfflineEmbedding:
        model = "embedding-3"
        dimensions = 2

        def __call__(self, texts):
            return [[1.0, 0.0] for _ in texts]

    server = build_sidecar_server(
        skill_root=tmp_path,
        retrieval_cards=card_path,
        embedding_client_factory=OfflineEmbedding,
    )
    readiness = _call(server, "handshake")["result"]["readiness"]

    assert readiness["ready"] is True
    assert readiness["dense_ready"] is True
    assert readiness["fusion_backend"] == "rrf"


def test_production_startup_fails_without_credential(tmp_path):
    skill_dir = tmp_path / "slides"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: slides\ndescription: create slides\n---\nBODY\n", encoding="utf-8"
    )
    card_path = tmp_path / "cards.jsonl"
    _write_cards(_store(), card_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("PARATERA_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    try:
        server = build_sidecar_server(
            skill_root=tmp_path, retrieval_cards=card_path
        )
    finally:
        monkeypatch.undo()
    readiness = _call(server, "handshake")["result"]["readiness"]
    assert readiness["ready"] is False
    assert "required for the dense backend" in readiness["errors"][0]
