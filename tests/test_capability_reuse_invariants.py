import pytest

from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore
from skill_control_plane.runtime import CapabilityDecision, CoverageClaim, SkillControlPlane
from tests.support import full_discovery


def _store() -> SkillStore:
    return SkillStore([
        SkillRecord(
            "api-design", "API Design", "design safe HTTP APIs",
            "API DESIGN BODY", "/api-design"),
        SkillRecord(
            "monitoring", "Monitoring", "monitor deployed services",
            "MONITORING BODY", "/monitoring"),
    ])


@pytest.fixture
def runtime() -> SkillControlPlane:
    store = _store()
    return SkillControlPlane(store, discovery=full_discovery(store))


def decision(action, skill_ids, *, purpose=None, target=None):
    return CapabilityDecision(
        action=action,
        skill_ids=tuple(skill_ids),
        reason="validated capability organization",
        target_bundle_id=target,
        purpose=purpose or "Backend work",
        coverage=tuple(
            CoverageClaim(f"use {skill_id}", f"skill:{skill_id}")
            for skill_id in skill_ids
        ),
    )


def create_api_design(runtime: SkillControlPlane) -> None:
    runtime.begin_turn()
    runtime.search_capability("design safe HTTP APIs")
    runtime.apply_capability(
        decision("CREATE", ("api-design",), purpose="Backend API work"))


def test_duplicate_create_is_reused_without_a_second_member_or_body(runtime):
    create_api_design(runtime)
    before = runtime.context_snapshot().maintained_bundles[0]

    runtime.begin_turn()
    runtime.search_capability("HTTP API contract design guidance")
    duplicate = runtime.apply_capability(
        decision("CREATE", ("api-design",), purpose="A different phrasing"))

    assert duplicate.requested_action == "CREATE"
    assert duplicate.effective_action == "reuse"
    assert duplicate.committed_skill_ids == ()
    assert duplicate.deduplicated_skill_ids == ("api-design",)
    assert duplicate.affected_bundle_id == before.bundle_id
    assert duplicate.skill_bodies == ()
    assert len(runtime.context_snapshot().maintained_bundles) == 1
    assert runtime.context_snapshot().maintained_bundles[0] == before


def test_mixed_create_commits_only_the_new_active_skill(runtime):
    create_api_design(runtime)
    api_bundle = runtime.context_snapshot().maintained_bundles[0]

    runtime.begin_turn()
    runtime.search_capability("design safe HTTP APIs")
    runtime.search_capability("monitor deployed services")
    mixed = runtime.apply_capability(decision(
        "CREATE", ("api-design", "monitoring"), purpose="Backend operations"))

    assert mixed.requested_action == "CREATE"
    assert mixed.effective_action == "CREATE"
    assert mixed.committed_skill_ids == ("monitoring",)
    assert mixed.deduplicated_skill_ids == ("api-design",)
    assert [body.skill_id for body in mixed.skill_bodies] == ["monitoring"]
    assert len(runtime.context_snapshot().maintained_bundles) == 2
    assert api_bundle in runtime.context_snapshot().maintained_bundles
    assert runtime.context_snapshot().skill_body_states == {
        "api-design": "resident", "monitoring": "resident"}


def test_repeated_discovery_scoring_uses_capability_state_not_query_text():
    from skill_control_plane.benchmarks.scorer import (
        duplicate_skill_activation_count,
        redundant_discovery_count,
    )

    events = [
        {"event_seq": 1, "type": "capability_search_result",
         "query": "q1", "skill_ids": ["api-design"], "active_skill_ids": []},
        {"event_seq": 2, "type": "capability_apply_result",
         "skill_ids": ["api-design"], "committed_skill_ids": ["api-design"],
         "deduplicated_skill_ids": []},
        {"event_seq": 3, "type": "capability_search_result",
         "query": "different q2", "skill_ids": ["api-design"],
         "active_skill_ids": ["api-design"]},
        {"event_seq": 4, "type": "capability_apply_result",
         "skill_ids": ["api-design"], "committed_skill_ids": [],
         "deduplicated_skill_ids": ["api-design"]},
        {"event_seq": 5, "type": "capability_search_result",
         "query": "q1", "skill_ids": ["api-design"],
         "active_skill_ids": ["api-design"]},
        {"event_seq": 6, "type": "capability_apply_result",
         "skill_ids": ["api-design"], "committed_skill_ids": [],
         "deduplicated_skill_ids": ["api-design"]},
    ]
    assert duplicate_skill_activation_count(events) == 2
    assert redundant_discovery_count(events) == 2


def test_repeated_query_that_finds_a_new_skill_is_not_redundant():
    from skill_control_plane.benchmarks.scorer import (
        duplicate_skill_activation_count,
        redundant_discovery_count,
    )

    events = [
        {"event_seq": 1, "type": "capability_search_result",
         "query": "backend work", "skill_ids": ["api-design"],
         "active_skill_ids": []},
        {"event_seq": 2, "type": "capability_apply_result",
         "skill_ids": ["api-design"], "committed_skill_ids": ["api-design"]},
        {"event_seq": 3, "type": "capability_search_result",
         "query": "backend work", "skill_ids": ["monitoring"],
         "active_skill_ids": ["api-design"]},
        {"event_seq": 4, "type": "capability_apply_result",
         "skill_ids": ["monitoring"], "committed_skill_ids": ["monitoring"]},
    ]
    assert redundant_discovery_count(events) == 0
    assert duplicate_skill_activation_count(events) == 0
