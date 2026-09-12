from skill_control_plane.models import TaskState
from skill_control_plane.evals.legacy.triggers import RerouteTrigger, detect_reroute_trigger


def test_failure_has_highest_priority() -> None:
    state = TaskState(
        goal="Deploy app",
        subgoal="Build container",
        previous_subgoal="Inspect repository",
        recent_failure="uv sync failed",
        capability_gap=True,
    )
    assert detect_reroute_trigger(state) is RerouteTrigger.EXECUTION_FAILURE


def test_subgoal_transition_triggers_reroute() -> None:
    state = TaskState(
        goal="Deploy app",
        previous_subgoal="Inspect repository",
        subgoal="Build Docker image",
    )
    assert detect_reroute_trigger(state) is RerouteTrigger.SUBGOAL_TRANSITION
