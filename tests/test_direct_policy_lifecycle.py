import json
from copy import deepcopy

import pytest

from skill_control_plane.integrations.reference_agent import (
    AGENT_INSTRUCTIONS,
    ReferenceSkillAgent,
    dto_payload,
)
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.runtime import (
    CapabilityDecision,
    CoverageClaim,
    RuntimeCapabilityState,
    SkillControlPlane,
)
from tests.support import full_discovery


class NoCallClient:
    def complete_messages(self, messages, *, tools):
        return {"role": "assistant", "content": "done"}


@pytest.fixture
def runtime():
    store = SkillRegistry([
        SkillRecord("pdf", "PDF", "read PDF documents", "PDF BODY", "/pdf"),
        SkillRecord("ocr", "OCR", "extract scanned document text", "OCR BODY", "/ocr"),
        SkillRecord("slides", "Slides", "create presentation slides", "SLIDES BODY", "/slides"),
    ])
    return SkillControlPlane(store, discovery=full_discovery(store))


def decide(action, skill_ids, *, target=None, purpose=None, bundle_coverage=()):
    return CapabilityDecision(
        action=action,
        skill_ids=tuple(skill_ids),
        reason="validated capability organization",
        target_bundle_id=target,
        purpose=purpose,
        coverage=tuple([
            *(CoverageClaim(need, f"bundle:{bundle_id}")
              for need, bundle_id in bundle_coverage),
            *(CoverageClaim(f"use {skill_id}", f"skill:{skill_id}")
              for skill_id in skill_ids),
        ]),
    )


def create_documents(runtime):
    runtime.search_capability("read PDF documents")
    return runtime.apply_capability(decide(
        "CREATE", ("pdf",), purpose="Document work"))


def test_create_builds_bundle_with_maintained_member(runtime):
    result = create_documents(runtime)
    bundle = runtime.context_snapshot().maintained_bundles[0]
    assert result.affected_bundle_id == bundle.bundle_id
    assert [(member.skill_id, member.member_role) for member in bundle.members] == [
        ("pdf", "maintained")]


def test_extend_adds_maintained_member_without_regressing_existing_role(runtime):
    bundle_id = create_documents(runtime).affected_bundle_id
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    runtime.apply_capability(decide("EXTEND", ("ocr",), target=bundle_id))
    members = runtime.context_snapshot().maintained_bundles[0].members
    assert [(member.skill_id, member.member_role) for member in members] == [
        ("ocr", "maintained"), ("pdf", "maintained")]


def test_direct_requires_existing_bundle_and_never_implicitly_creates(runtime):
    runtime.search_capability("extract scanned document text")
    before = deepcopy(runtime.context_snapshot())
    with pytest.raises(ValueError, match="target_bundle_id must be a string"):
        runtime.apply_capability(decide("DIRECT", ("ocr",)))
    after = runtime.context_snapshot()
    assert after.maintained_bundles == ()
    assert after.pending_candidate_skill_ids == before.pending_candidate_skill_ids


def test_direct_member_persists_across_turns_for_coverage_and_dedup(runtime):
    bundle_id = create_documents(runtime).affected_bundle_id
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    applied = runtime.apply_capability(decide(
        "DIRECT", ("ocr",), target=bundle_id,
        bundle_coverage=(("document workflow", bundle_id),),
    ))
    assert applied.affected_bundle_id == bundle_id
    snapshot = runtime.context_snapshot()
    member = next(member for member in snapshot.maintained_bundles[0].members
                  if member.skill_id == "ocr")
    assert member.member_role == "direct"
    assert snapshot.direct_skill_ids == ("ocr",)
    assert any("scanned" in capability
               for capability in snapshot.maintained_bundles[0].capabilities)

    runtime.begin_turn()
    later = runtime.context_snapshot()
    assert later.pending_candidate_skill_ids == ()
    assert [(member.skill_id, member.member_role)
            for member in later.maintained_bundles[0].members] == [
        ("ocr", "direct"), ("pdf", "maintained")]
    runtime.search_capability("create presentation slides")
    extended = runtime.apply_capability(decide(
        "EXTEND", ("slides",), target=bundle_id,
        bundle_coverage=(("reuse scanned extraction", bundle_id),),
    ))
    assert extended.coverage[0] == CoverageClaim(
        "reuse scanned extraction", f"bundle:{bundle_id}")


def test_direct_and_maintained_members_share_body_lifecycle(runtime):
    bundle_id = create_documents(runtime).affected_bundle_id
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    runtime.apply_capability(decide("DIRECT", ("ocr",), target=bundle_id))
    assert runtime.context_snapshot().skill_body_states == {
        "ocr": "resident", "pdf": "resident"}

    runtime.mark_all_skill_bodies_evicted()
    assert runtime.context_snapshot().skill_body_states == {
        "ocr": "evicted", "pdf": "evicted"}
    direct = runtime.load_skill_body("ocr")
    maintained = runtime.load_skill_body("pdf")
    assert (direct.status, direct.body, direct.bundle_ids) == (
        "loaded", "OCR BODY", (bundle_id,))
    assert (maintained.status, maintained.body, maintained.bundle_ids) == (
        "loaded", "PDF BODY", (bundle_id,))


def test_non_pending_direct_is_rejected_and_failed_apply_is_atomic(runtime, monkeypatch):
    bundle_id = create_documents(runtime).affected_bundle_id
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    before = deepcopy(runtime.context_snapshot())
    with pytest.raises(ValueError, match="outside supplied candidates"):
        runtime.apply_capability(decide("DIRECT", ("slides",), target=bundle_id))
    assert runtime.context_snapshot() == before

    runtime.search_capability("create presentation slides")
    pending = deepcopy(runtime.context_snapshot())
    original_get = SkillRegistry.get
    calls = 0

    def fail_during_body_materialization(store, skill_id):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated store failure")
        return original_get(store, skill_id)

    with monkeypatch.context() as patch:
        patch.setattr(SkillRegistry, "get", fail_during_body_materialization)
        with pytest.raises(RuntimeError, match="simulated store failure"):
            runtime.apply_capability(decide(
                "DIRECT", ("ocr", "slides"), target=bundle_id))
    assert runtime.context_snapshot() == pending


def test_snapshot_and_agent_render_compact_member_roles(runtime):
    bundle_id = create_documents(runtime).affected_bundle_id
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    runtime.apply_capability(decide("DIRECT", ("ocr",), target=bundle_id))

    payload = dto_payload(runtime.context_snapshot())
    members = payload["maintained_bundles"][0]["members"]
    assert {member["skill_id"]: member["member_role"] for member in members} == {
        "ocr": "direct", "pdf": "maintained"}
    assert all(set(member) == {"skill_id", "name", "member_role", "body_state"}
               for member in members)

    rendered = ReferenceSkillAgent(runtime, NoCallClient()).render_system_context()
    surface = json.loads(rendered.split("Runtime Bundles (metadata only)\n", 1)[1])
    assert surface["maintained_bundles"][0]["members"] == members
    assert ("CREATE/EXTEND are the default maintained actions. DIRECT is for "
            "clearly one-off or short-lived") in AGENT_INSTRUCTIONS.replace("\n", " ")
    assert "reasonable existing Bundle" in AGENT_INSTRUCTIONS


def test_legacy_detached_direct_ids_are_read_only_compatibility_input():
    store = SkillRegistry([
        SkillRecord("ocr", "OCR", "extract scans", "OCR BODY", "/ocr")])
    runtime = SkillControlPlane(
        store,
        discovery=full_discovery(store),
        state=RuntimeCapabilityState(direct_skills={"ocr"}),
    )
    snapshot = runtime.context_snapshot()
    assert snapshot.direct_skill_ids == ("ocr",)
    assert snapshot.maintained_bundles == ()
    assert snapshot.skill_body_states == {}
