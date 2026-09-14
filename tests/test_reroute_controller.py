from __future__ import annotations

from dataclasses import dataclass

from skill_control_plane import (
    CapabilityDecision,
    CompletionCapabilityGapDecider,
    CoverageClaim,
    GapDecision,
    RerouteController,
    RerouteState,
    RuntimeEvidence,
    SkillControlPlane,
    StateSnapshotBundleV1,
    StateSnapshotMemberV1,
    StateSnapshotV1,
)
from skill_control_plane.runtime.capability_memory import capability_store_fingerprint
from skill_control_plane.models import SkillRecord
from skill_control_plane.registry import SkillStore
from tests.support import full_discovery


@dataclass
class ScriptedDecider:
    decisions: list[GapDecision]

    def __post_init__(self) -> None:
        self.calls: list[dict] = []

    def decide_gap(self, evidence, active_bundle_cards, current_subgoal_context):
        self.calls.append({
            "evidence": evidence,
            "active_bundle_cards": active_bundle_cards,
            "current_subgoal_context": current_subgoal_context,
        })
        return self.decisions.pop(0)


def evidence(
    evidence_id: str,
    text: str,
    *,
    kind: str = "tool_error",
    source: str = "bash",
    fingerprint: str | None = None,
    metadata: dict | None = None,
) -> RuntimeEvidence:
    return RuntimeEvidence.create(
        evidence_id=evidence_id,
        kind=kind,
        source=source,
        text=text,
        fingerprint=fingerprint,
        metadata=metadata,
    )


def test_ordinary_success_is_ignored_without_gap_decision():
    decider = ScriptedDecider([])
    controller = RerouteController(
        gap_decider=decider,
        discover_capability=lambda _need: (_ for _ in ()).throw(AssertionError()),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )

    outcome = controller.observe(evidence(
        "ok-1", "all tests passed", kind="host_signal",
        metadata={"capability_relevant": False},
    ))

    assert outcome.status == "ignored"
    assert decider.calls == []


def test_completion_gap_decider_receives_only_evidence_cards_and_subgoal():
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return '{"needs_capability":true,"need":"inspect database locks","rationale":"new failure"}'

    decision = CompletionCapabilityGapDecider(complete).decide_gap(
        evidence("db", "deadlock detected"), (), "stabilize checkout",
    )

    assert decision.need == "inspect database locks"
    assert "deadlock detected" in prompts[0]
    assert "stabilize checkout" in prompts[0]
    assert "conversation" not in prompts[0].casefold()
    assert "skill bod" in prompts[0].casefold()


def test_trace_b_failure_no_gap_is_terminal_and_does_not_retrieve():
    decider = ScriptedDecider([GapDecision(False, None, "already covered")])
    searches: list[str] = []
    controller = RerouteController(
        gap_decider=decider,
        discover_capability=lambda need: searches.append(need),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )

    outcome = controller.observe(evidence("a", "test failed"))

    assert outcome.status == "no_gap"
    assert outcome.state is RerouteState.NO_GAP
    assert searches == []
    assert controller.snapshot().evidence[0].state_history == (
        RerouteState.NEW, RerouteState.CHECKING, RerouteState.NO_GAP,
    )


def test_gap_automatically_discovers_and_duplicate_fingerprint_is_ignored():
    decider = ScriptedDecider([GapDecision(True, "debug async tests", "missing")])
    searches: list[str] = []

    @dataclass(frozen=True)
    class Result:
        query: str
        candidates: tuple[str, ...]

    controller = RerouteController(
        gap_decider=decider,
        discover_capability=lambda need: (
            searches.append(need) or Result(need, ("async-debug",))
        ),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )
    first = controller.observe(evidence("a", "test failed", fingerprint="same"))
    duplicate = controller.observe(evidence(
        "a-copy", "test failed again", fingerprint="same",
    ))

    assert first.status == "discovered"
    assert first.state is RerouteState.DISCOVERED
    assert first.need == "debug async tests"
    assert first.candidates == ("async-debug",)
    assert duplicate.status == "duplicate"
    assert searches == ["debug async tests"]
    assert len(decider.calls) == 1
    assert controller.snapshot().evidence[0].state_history == (
        RerouteState.NEW,
        RerouteState.CHECKING,
        RerouteState.GAP_FOUND,
        RerouteState.DISCOVERED,
    )


def test_different_evidence_in_one_turn_is_processed_independently():
    decider = ScriptedDecider([
        GapDecision(False, None),
        GapDecision(True, "configure monitoring"),
    ])

    @dataclass(frozen=True)
    class Result:
        query: str
        candidates: tuple[str, ...]

    controller = RerouteController(
        gap_decider=decider,
        discover_capability=lambda need: Result(need, ("monitoring",)),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )

    assert controller.observe(evidence("a", "assertion failed")).status == "no_gap"
    assert controller.observe(evidence("b", "alert missing")).status == "discovered"
    assert [item.state for item in controller.snapshot().evidence] == [
        RerouteState.NO_GAP, RerouteState.DISCOVERED,
    ]


def test_control_plane_internal_results_are_ineligible():
    decider = ScriptedDecider([])
    controller = RerouteController(
        gap_decider=decider,
        discover_capability=lambda _need: (_ for _ in ()).throw(AssertionError()),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )

    for source in (
        "capability_gap_decision", "discover_capability", "load_capability",
        "apply_capability", "load_skill_body", "telemetry",
    ):
        assert controller.observe(evidence(source, "internal error", source=source)).status == "ignored"
    assert decider.calls == []


def test_discovery_failure_records_failed_state_and_explainable_telemetry():
    controller = RerouteController(
        gap_decider=ScriptedDecider([GapDecision(True, "repair deployment")]),
        discover_capability=lambda _need: (_ for _ in ()).throw(RuntimeError("offline")),
        active_bundle_cards=lambda: (),
        active_skill_ids=lambda: (),
    )

    outcome = controller.observe(evidence("failure", "deploy failed"))

    assert outcome.status == "failed"
    assert outcome.state is RerouteState.FAILED
    discovery = [event for event in controller.telemetry() if event.event == "reroute_discovery"]
    assert discovery[-1].data["status"] == "failed"
    assert discovery[-1].data["error_type"] == "RuntimeError"
    assert controller.snapshot().evidence[0].failure_stage == "discovery"
    assert controller.snapshot().evidence[0].failure_reason == "offline"


def _runtime(decider: ScriptedDecider):
    store = SkillStore([
        SkillRecord("ocr", "OCR", "extract scanned text", "OCR BODY", "/ocr"),
        SkillRecord("slides", "Slides", "create slides", "SLIDE BODY", "/slides"),
    ])
    return store, SkillControlPlane(
        store, discovery=full_discovery(store), gap_decider=decider,
    )


def test_trace_c_failure_gap_commit_is_runtime_driven():
    store, runtime = _runtime(ScriptedDecider([
        GapDecision(True, "create slides"),
    ]))
    runtime.begin_turn()
    outcome = runtime.observe_runtime_evidence(evidence("e-slides", "slides missing"))
    assert outcome.status == "discovered"
    assert "slides" in {candidate.skill_id for candidate in outcome.candidates}

    runtime.apply_capability(CapabilityDecision(
        action="CREATE", skill_ids=("slides",), reason="presentation work",
        purpose="Presentation work",
        coverage=(CoverageClaim("create slides", "skill:slides"),),
    ), reroute_evidence_id="e-slides")
    record = runtime.reroute_snapshot().evidence[0]
    assert record.state is RerouteState.COMMITTED
    assert record.state_history == (
        RerouteState.NEW,
        RerouteState.CHECKING,
        RerouteState.GAP_FOUND,
        RerouteState.DISCOVERED,
        RerouteState.SELECTED,
        RerouteState.COMMITTED,
    )
    active_members = {
        member.skill_id
        for bundle in runtime.context_snapshot().maintained_bundles
        for member in bundle.members
    }
    assert "slides" in active_members
    runtime.load_skill_body("slides")
    events = [event.event for event in runtime.reroute_snapshot().telemetry]
    assert "reroute_candidate_selected" in events
    assert "reroute_apply_result" in events
    assert "reroute_activation" in events
    assert "reroute_body_loaded" in events


def test_trace_d_multi_evidence_same_turn_reaches_independent_terminals():
    decider = ScriptedDecider([
        GapDecision(False, None, "already covered"),
        GapDecision(True, "create slides", "missing presentation capability"),
    ])
    _, runtime = _runtime(decider)
    runtime.begin_turn()

    evidence_a = evidence("A", "tests failed in an already covered endpoint")
    assert runtime.observe_runtime_evidence(evidence_a).status == "no_gap"
    assert runtime.observe_runtime_evidence(evidence(
        "A-repeat", "same normalized content", fingerprint=evidence_a.fingerprint,
    )).status == "duplicate"
    outcome_b = runtime.observe_runtime_evidence(evidence(
        "B", "verifier failed because slide generation is unavailable",
    ))
    assert outcome_b.status == "discovered"
    assert "slides" in {candidate.skill_id for candidate in outcome_b.candidates}

    runtime.apply_capability(CapabilityDecision(
        action="CREATE", skill_ids=("slides",), reason="presentation work",
        purpose="Presentation work",
        coverage=(CoverageClaim("create slides", "skill:slides"),),
    ), reroute_evidence_id="B")

    snapshot = runtime.reroute_snapshot()
    records = {record.evidence.evidence_id: record for record in snapshot.evidence}
    assert records["A"].state is RerouteState.NO_GAP
    assert records["B"].state is RerouteState.COMMITTED
    assert records["A"].state_history == (
        RerouteState.NEW, RerouteState.CHECKING, RerouteState.NO_GAP,
    )
    assert records["B"].state_history == (
        RerouteState.NEW, RerouteState.CHECKING, RerouteState.GAP_FOUND,
        RerouteState.DISCOVERED, RerouteState.SELECTED, RerouteState.COMMITTED,
    )
    assert len(decider.calls) == 2
    assert sum(event.event == "reroute_gap_decision" for event in snapshot.telemetry) == 2
    assert sum(
        event.event == "reroute_discovery"
        and event.data.get("status") == "discovered"
        for event in snapshot.telemetry
    ) == 1

    # Context/tool-loop reads retain the controller; only end_turn clears records.
    runtime.context_snapshot()
    assert len(runtime.reroute_snapshot().evidence) == 2
    runtime.end_turn()
    assert runtime.reroute_snapshot().evidence == ()
    assert runtime.reroute_snapshot().telemetry


def test_failed_records_cover_gap_decision_selection_and_apply_stages():
    class BrokenDecider:
        def decide_gap(self, evidence, active_bundle_cards, current_subgoal_context):
            raise RuntimeError("decider unavailable")

    _, gap_runtime = _runtime(BrokenDecider())
    gap_runtime.begin_turn()
    gap_outcome = gap_runtime.observe_runtime_evidence(evidence("gap", "test failed"))
    assert gap_outcome.failure_stage == "gap_decision"
    assert gap_outcome.failure_reason == "decider unavailable"

    _, selection_runtime = _runtime(ScriptedDecider([
        GapDecision(True, "create slides"),
    ]))
    selection_runtime.begin_turn()
    selection_runtime.observe_runtime_evidence(evidence("selection", "slides missing"))
    try:
        selection_runtime.apply_capability(CapabilityDecision(
            action="CREATE", skill_ids=("not-a-candidate",), reason="invalid",
            purpose="Invalid",
        ), reroute_evidence_id="selection")
    except ValueError:
        pass
    selection_record = selection_runtime.reroute_snapshot().evidence[0]
    assert selection_record.state is RerouteState.FAILED
    assert selection_record.failure_stage == "selection"

    _, apply_runtime = _runtime(ScriptedDecider([
        GapDecision(True, "create slides"),
    ]))
    apply_runtime.begin_turn()
    apply_runtime.observe_runtime_evidence(evidence("apply", "slides missing"))
    try:
        apply_runtime.apply_capability(CapabilityDecision(
            action="CREATE", skill_ids=("slides",), reason="missing purpose",
        ), reroute_evidence_id="apply")
    except ValueError:
        pass
    apply_record = apply_runtime.reroute_snapshot().evidence[0]
    assert apply_record.state is RerouteState.FAILED
    assert apply_record.failure_stage == "apply"
    assert apply_record.state_history[-2:] == (
        RerouteState.SELECTED, RerouteState.FAILED,
    )


def test_turn_lifecycle_retains_tool_loop_state_and_end_turn_clears_it():
    _, runtime = _runtime(ScriptedDecider([GapDecision(False, None)]))
    runtime.begin_turn()
    runtime.observe_runtime_evidence(evidence("a", "test failed"))
    assert len(runtime.reroute_snapshot().evidence) == 1
    assert runtime.context_snapshot().reroute_pending_count == 0
    assert len(runtime.reroute_snapshot().evidence) == 1
    runtime.end_turn()
    assert runtime.reroute_snapshot().evidence == ()
    assert runtime.reroute_snapshot().telemetry


def test_restored_bundle_cards_are_passed_to_gap_decider():
    decider = ScriptedDecider([GapDecision(False, None)])
    store, runtime = _runtime(decider)
    runtime.restore_state(StateSnapshotV1(
        version="state-snapshot-v1",
        store_fingerprint=capability_store_fingerprint(store),
        bundles=(StateSnapshotBundleV1(
            bundle_id="bundle-restored",
            purpose="Presentation work",
            members=(StateSnapshotMemberV1(
                skill_id="slides", member_role="maintained", body_state="resident",
            ),),
        ),),
    ))
    runtime.begin_turn()
    runtime.observe_runtime_evidence(evidence("after-restore", "verifier failed"))

    cards = decider.calls[0]["active_bundle_cards"]
    assert cards[0].bundle_id == "bundle-restored"
    assert cards[0].members[0].skill_id == "slides"
