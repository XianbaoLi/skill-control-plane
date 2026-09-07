from __future__ import annotations

from skill_control_plane.models import EvolutionAction, EvolutionDecision, Relation


def action_for_relation(relation: Relation) -> EvolutionAction:
    """V0.1's intentionally conservative mutation policy."""

    if relation is Relation.COVERED:
        return EvolutionAction.NOOP
    if relation in (Relation.REFINES, Relation.EXTENDS):
        return EvolutionAction.PATCH_PROPOSAL
    if relation is Relation.NOVEL:
        return EvolutionAction.CREATE_PROPOSAL
    raise ValueError(f"unsupported relation: {relation}")


def build_decision(
    *,
    relation: Relation,
    target_skill: str | None,
    confidence: float,
    reason: str,
    candidate_evidence: tuple[str, ...] = (),
) -> EvolutionDecision:
    action = action_for_relation(relation)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if action is EvolutionAction.PATCH_PROPOSAL and not target_skill:
        raise ValueError("patch proposals require target_skill")
    if action is EvolutionAction.CREATE_PROPOSAL:
        target_skill = None
    return EvolutionDecision(
        action=action,
        relation=relation,
        target_skill=target_skill,
        confidence=confidence,
        reason=reason,
        candidate_evidence=candidate_evidence,
    )
