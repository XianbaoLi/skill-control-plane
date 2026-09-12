from __future__ import annotations

from enum import StrEnum

from skill_control_plane.models import TaskState


class RerouteTrigger(StrEnum):
    SUBGOAL_TRANSITION = "subgoal_transition"
    EXECUTION_FAILURE = "execution_failure"
    CAPABILITY_GAP = "capability_gap"


def detect_reroute_trigger(state: TaskState) -> RerouteTrigger | None:
    """Return the highest-priority V0.1 rerouting trigger, if any."""

    if state.recent_failure:
        return RerouteTrigger.EXECUTION_FAILURE
    if state.capability_gap:
        return RerouteTrigger.CAPABILITY_GAP
    if state.subgoal and state.previous_subgoal and state.subgoal != state.previous_subgoal:
        return RerouteTrigger.SUBGOAL_TRANSITION
    return None
