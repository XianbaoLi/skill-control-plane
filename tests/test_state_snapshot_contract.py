from copy import deepcopy
from dataclasses import fields, replace

import pytest

from skill_control_plane import (
    CapabilityDecision,
    CoverageClaim,
    SkillControlPlane,
    StateSnapshotBundleV1,
    StateSnapshotMemberV1,
    StateSnapshotV1,
)
from skill_control_plane.discovery import SkillDiscovery
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore
from tests.support import DeterministicFakeDenseRetriever, complete_cards, full_discovery


def _store() -> SkillStore:
    return SkillStore([
        SkillRecord("ocr", "OCR", "extract scanned text", "OCR BODY", "/ocr"),
        SkillRecord("slides", "Slides", "create slides", "SLIDE BODY", "/slides"),
    ])


@pytest.fixture
def runtime() -> SkillControlPlane:
    store = _store()
    return SkillControlPlane(store, discovery=full_discovery(store))


def _exported_snapshot(runtime: SkillControlPlane) -> StateSnapshotV1:
    runtime.begin_turn()
    runtime.search_capability("create slides")
    runtime.apply_capability(CapabilityDecision(
        action="CREATE",
        skill_ids=("slides",),
        reason="reusable presentation work",
        purpose="Presentation work",
        coverage=(CoverageClaim("slides", "skill:slides"),),
    ))
    runtime.begin_turn()
    runtime.search_capability("extract scanned document text")
    runtime.apply_capability(CapabilityDecision(
        action="DIRECT",
        skill_ids=("ocr",),
        reason="one-turn operation",
        target_bundle_id=runtime.context_snapshot().maintained_bundles[0].bundle_id,
        coverage=(CoverageClaim("scan", "skill:ocr"),),
    ))
    runtime.mark_skill_body_evicted("ocr")
    return runtime.export_state()


def test_state_snapshot_v1_round_trips_persistent_memory(runtime):
    snapshot = _exported_snapshot(runtime)
    store = _store()
    restored = SkillControlPlane(store, discovery=full_discovery(store))
    restored.restore_state(snapshot)

    assert snapshot.version == "state-snapshot-v1"
    assert snapshot == restored.export_state()
    context = restored.context_snapshot()
    assert [(member.skill_id, member.member_role, member.body_state)
            for member in context.maintained_bundles[0].members] == [
        ("ocr", "direct", "evicted"),
        ("slides", "maintained", "resident"),
    ]
    assert context.skill_body_states == {"slides": "resident", "ocr": "evicted"}
    assert restored.load_skill_body("slides").status == "already_resident"
    assert restored.load_skill_body("ocr").body == "OCR BODY"


def test_export_contains_only_persistent_bundle_memory(runtime):
    runtime.begin_turn()
    runtime.search_capability("create slides")
    snapshot = runtime.export_state()

    assert [field.name for field in fields(snapshot)] == [
        "version", "store_fingerprint", "bundles"]
    assert all(isinstance(bundle, StateSnapshotBundleV1) for bundle in snapshot.bundles)
    assert all(
        isinstance(member, StateSnapshotMemberV1)
        for bundle in snapshot.bundles
        for member in bundle.members
    )


def test_restore_replaces_turn_local_state_with_a_clean_turn(runtime):
    snapshot = _exported_snapshot(runtime)
    runtime.begin_turn()
    runtime.search_capability("unrelated need")

    runtime.restore_state(snapshot)
    context = runtime.context_snapshot()

    assert context.pending_query is None
    assert context.pending_candidate_skill_ids == ()
    assert context.search_count == 0
    assert context.remaining_search_budget == 3
    assert context.capability_sufficiency_outcome == "COVERED"
    assert context.maintained_bundles[0].members[0].skill_id == "ocr"


def test_restore_rejects_unsupported_version(runtime):
    snapshot = replace(_exported_snapshot(runtime), version="state-snapshot-v2")
    with pytest.raises(ValueError, match="unsupported state snapshot version"):
        runtime.restore_state(snapshot)


def test_restore_rejects_unknown_skill(runtime):
    snapshot = _exported_snapshot(runtime)
    mismatched = replace(
        snapshot,
        bundles=(
            replace(
                snapshot.bundles[0],
                members=(StateSnapshotMemberV1(
                    "missing", "maintained", "resident"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unknown state snapshot Skill"):
        runtime.restore_state(mismatched)


@pytest.mark.parametrize(("member_role", "body_state"), [
    ("admin", "resident"),
    ("maintained", "loading"),
])
def test_restore_rejects_malformed_member_state(runtime, member_role, body_state):
    snapshot = _exported_snapshot(runtime)
    mismatched = replace(
        snapshot,
        bundles=(replace(
            snapshot.bundles[0],
            members=(StateSnapshotMemberV1(
                "slides", member_role, body_state),),
        ),),
    )
    with pytest.raises(ValueError, match="invalid Bundle member role|invalid Skill body state"):
        runtime.restore_state(mismatched)


def test_restore_rejects_store_mismatch_without_partial_mutation(runtime):
    snapshot = _exported_snapshot(runtime)
    runtime.begin_turn()
    runtime.search_capability("unrelated need")
    before = deepcopy(runtime.context_snapshot())
    before_audit = deepcopy(runtime.turn_audit())
    invalid = replace(
        snapshot,
        bundles=(replace(
            snapshot.bundles[0],
            members=(StateSnapshotMemberV1(
                "missing", "maintained", "resident"),),
        ),),
    )

    with pytest.raises(ValueError, match="unknown state snapshot Skill"):
        runtime.restore_state(invalid)
    assert runtime.context_snapshot() == before
    assert runtime.turn_audit() == before_audit


def test_readiness_reports_full_discovery_without_turn_state_pollution(runtime):
    runtime.begin_turn()
    runtime.search_capability("create slides")
    before = deepcopy(runtime.context_snapshot())
    before_audit = deepcopy(runtime.turn_audit())

    readiness = runtime.readiness()

    assert readiness.ready is True
    assert readiness.skill_count == 2
    assert readiness.retrieval_cards_ready is True
    assert readiness.dense_ready is True
    assert readiness.fusion_backend == "rrf"
    assert readiness.errors == ()
    assert {check.name for check in readiness.checks} == {
        "skill_store", "retrieval_cards", "dense_backend",
        "fusion_backend", "dense_rrf_probe"}
    assert all(check.ready for check in readiness.checks)
    assert runtime.context_snapshot() == before
    assert runtime.turn_audit() == before_audit


def test_readiness_rejects_missing_or_stale_retrieval_cards(runtime):
    runtime._discovery.retrieval_cards.pop("ocr")
    readiness = runtime.readiness()

    assert readiness.ready is False
    assert readiness.retrieval_cards_ready is False
    assert readiness.errors


def test_readiness_rejects_stale_retrieval_cards(runtime):
    stale_cards = dict(runtime._discovery.retrieval_cards)
    stale_cards["ocr"] = replace(
        stale_cards["ocr"], source_content_hash="stale-hash")
    runtime._discovery.retrieval_cards = stale_cards
    readiness = runtime.readiness()

    assert readiness.ready is False
    assert readiness.retrieval_cards_ready is False
    assert "stale=['ocr']" in readiness.errors[0]


def test_readiness_rejects_missing_dense(runtime):
    runtime._discovery.dense = None
    readiness = runtime.readiness()

    assert readiness.ready is False
    assert readiness.dense_ready is False
    assert readiness.fusion_backend == "bm25"


def test_readiness_rejects_non_rrf_production_path(runtime):
    class StaleFusionDiscovery(SkillDiscovery):
        @property
        def fusion_backend(self) -> str:
            return "bm25"

    store = runtime._store
    runtime._discovery = StaleFusionDiscovery(
        store,
        retrieval_cards=complete_cards(store),
        dense_factory=DeterministicFakeDenseRetriever,
    )
    readiness = runtime.readiness()

    assert readiness.ready is False
    assert readiness.fusion_backend == "bm25"
    assert readiness.errors == ()
