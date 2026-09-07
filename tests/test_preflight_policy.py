import pytest

from skill_control_plane.evolution import action_for_relation, build_decision
from skill_control_plane.models import EvolutionAction, Relation


@pytest.mark.parametrize(
    ("relation", "action"),
    [
        (Relation.COVERED, EvolutionAction.NOOP),
        (Relation.REFINES, EvolutionAction.PATCH_PROPOSAL),
        (Relation.EXTENDS, EvolutionAction.PATCH_PROPOSAL),
        (Relation.NOVEL, EvolutionAction.CREATE_PROPOSAL),
    ],
)
def test_relation_policy(relation: Relation, action: EvolutionAction) -> None:
    assert action_for_relation(relation) is action


def test_patch_requires_target_skill() -> None:
    with pytest.raises(ValueError, match="target_skill"):
        build_decision(
            relation=Relation.EXTENDS,
            target_skill=None,
            confidence=0.9,
            reason="same trigger class",
        )


def test_novel_forces_no_target() -> None:
    decision = build_decision(
        relation=Relation.NOVEL,
        target_skill="should-be-cleared",
        confidence=0.8,
        reason="no existing capability can host this lesson",
    )
    assert decision.action is EvolutionAction.CREATE_PROPOSAL
    assert decision.target_skill is None
