from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class Relation(StrEnum):
    COVERED = "covered"
    REFINES = "refines"
    EXTENDS = "extends"
    NOVEL = "novel"


class EvolutionAction(StrEnum):
    NOOP = "noop"
    PATCH_PROPOSAL = "patch_proposal"
    CREATE_PROPOSAL = "create_proposal"


@dataclass(frozen=True, slots=True)
class SkillRecord:
    skill_id: str
    name: str
    description: str
    body: str
    source_path: str
    tags: tuple[str, ...] = ()
    content_hash: str = ""
    category: str = ""
    retrieval_representation: str = ""

    @property
    def search_text(self) -> str:
        parts = [self.name, self.description, " ".join(self.tags), self.body]
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    skill_id: str
    score: float
    rank: int
    source_scores: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskState:
    goal: str
    subgoal: str | None = None
    previous_subgoal: str | None = None
    recent_observation: str | None = None
    recent_failure: str | None = None
    capability_gap: bool = False
    active_skills: tuple[str, ...] = ()

    def retrieval_query(self) -> str:
        fields = [self.goal, self.subgoal, self.recent_observation, self.recent_failure]
        return "\n".join(value for value in fields if value)


@dataclass(frozen=True, slots=True)
class Experience:
    task: str
    lesson: str
    evidence: tuple[str, ...] = ()

    def retrieval_query(self) -> str:
        return "\n".join([self.task, self.lesson, *self.evidence])


@dataclass(frozen=True, slots=True)
class EvolutionDecision:
    action: EvolutionAction
    relation: Relation
    target_skill: str | None
    confidence: float
    reason: str
    candidate_evidence: tuple[str, ...] = ()
