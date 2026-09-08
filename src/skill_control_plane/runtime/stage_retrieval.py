from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from skill_control_plane.models import RetrievalCandidate
from skill_control_plane.retrieval import Retriever, reciprocal_rank_fusion
from skill_control_plane.runtime.evidence import EvidencePool


class RetrievalPhase(StrEnum):
    RAW = "raw"
    AGENT_AUGMENTED = "agent_augmented"
    SRC_ENHANCED = "src_enhanced"


@dataclass(frozen=True, slots=True)
class StageRetrievalContext:
    """Compact semantic context used only to improve Skill retrieval."""

    text: str
    source: str = "enhancer"


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    phase: RetrievalPhase
    query: str
    candidates: tuple[RetrievalCandidate, ...]
    sufficient: bool


@dataclass(frozen=True, slots=True)
class StageRetrievalResult:
    """Result of selective semantic escalation for one stage transition."""

    candidates: tuple[RetrievalCandidate, ...]
    phase: RetrievalPhase
    traces: tuple[RetrievalTrace, ...]
    src: StageRetrievalContext | None = None

    @property
    def used_extra_context(self) -> bool:
        return self.phase is RetrievalPhase.SRC_ENHANCED


class StageContextEnhancer(Protocol):
    def build_context(self, evidence: EvidencePool) -> StageRetrievalContext:
        """Return a retrieval-oriented semantic context, not a task solution."""
        ...


RetrievalSufficiency = Callable[[Sequence[RetrievalCandidate]], bool]


def retrieve_for_stage(
    evidence: EvidencePool,
    retriever: Retriever,
    *,
    sufficiency: RetrievalSufficiency,
    enhancer: StageContextEnhancer | None = None,
    k: int = 5,
    rrf_k: int = 60,
) -> StageRetrievalResult:
    """Retrieve the next capability with selective semantic escalation.

    Escalation order:
    1. raw + structured runtime evidence;
    2. add main-Agent interpretation already present in the EvidencePool;
    3. if still insufficient and an enhancer exists, retrieve from SRC and
       fuse it with the evidence-only ranking.

    No universal sufficiency threshold is hard-coded here. The caller must
    supply a policy calibrated for its corpus/evaluation setting.
    """

    if evidence.is_empty:
        return StageRetrievalResult(
            candidates=(),
            phase=RetrievalPhase.RAW,
            traces=(),
        )

    traces: list[RetrievalTrace] = []

    raw_query = evidence.raw_retrieval_text()
    raw_candidates = tuple(retriever.search(raw_query, k=k)) if raw_query else ()
    raw_ok = bool(raw_candidates) and sufficiency(raw_candidates)
    traces.append(
        RetrievalTrace(
            phase=RetrievalPhase.RAW,
            query=raw_query,
            candidates=raw_candidates,
            sufficient=raw_ok,
        )
    )
    if raw_ok:
        return StageRetrievalResult(
            candidates=raw_candidates,
            phase=RetrievalPhase.RAW,
            traces=tuple(traces),
        )

    current_candidates = raw_candidates

    if evidence.agent_interpretation:
        agent_query = evidence.direct_retrieval_text()
        agent_candidates = tuple(retriever.search(agent_query, k=k))
        agent_ok = bool(agent_candidates) and sufficiency(agent_candidates)
        traces.append(
            RetrievalTrace(
                phase=RetrievalPhase.AGENT_AUGMENTED,
                query=agent_query,
                candidates=agent_candidates,
                sufficient=agent_ok,
            )
        )
        current_candidates = agent_candidates
        if agent_ok:
            return StageRetrievalResult(
                candidates=agent_candidates,
                phase=RetrievalPhase.AGENT_AUGMENTED,
                traces=tuple(traces),
            )

    if enhancer is None:
        phase = (
            RetrievalPhase.AGENT_AUGMENTED
            if evidence.agent_interpretation
            else RetrievalPhase.RAW
        )
        return StageRetrievalResult(
            candidates=current_candidates,
            phase=phase,
            traces=tuple(traces),
        )

    src = enhancer.build_context(evidence)
    src_query = src.text.strip()
    if not src_query:
        phase = (
            RetrievalPhase.AGENT_AUGMENTED
            if evidence.agent_interpretation
            else RetrievalPhase.RAW
        )
        return StageRetrievalResult(
            candidates=current_candidates,
            phase=phase,
            traces=tuple(traces),
            src=src,
        )

    src_candidates = tuple(retriever.search(src_query, k=k))

    rankings: dict[str, Sequence[RetrievalCandidate]] = {"src": src_candidates}
    if raw_candidates:
        rankings["raw"] = raw_candidates
    if evidence.agent_interpretation and current_candidates:
        rankings["agent"] = current_candidates

    fused = tuple(
        reciprocal_rank_fusion(
            rankings,
            k=rrf_k,
            limit=k,
        )
    )
    src_ok = bool(fused) and sufficiency(fused)
    traces.append(
        RetrievalTrace(
            phase=RetrievalPhase.SRC_ENHANCED,
            query=src_query,
            candidates=fused,
            sufficient=src_ok,
        )
    )

    return StageRetrievalResult(
        candidates=fused,
        phase=RetrievalPhase.SRC_ENHANCED,
        traces=tuple(traces),
        src=src,
    )
