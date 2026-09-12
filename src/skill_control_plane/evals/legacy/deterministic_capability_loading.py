"""V0.7 capability loading: discovery followed by deterministic composition resolution."""
from dataclasses import dataclass
from enum import StrEnum

from skill_control_plane.models import SkillRecord
from skill_control_plane.evals.legacy.bundle_registry import Bundle, BundleRegistry
from skill_control_plane.discovery.discovery import SkillDiscovery, SkillDiscoveryResult


class Decision(StrEnum):
    DIRECT = "DIRECT"
    REUSE = "REUSE"
    EXTEND = "EXTEND"
    CREATE = "CREATE"


@dataclass(frozen=True)
class ResolutionPolicy:
    min_coverage: float = 2 / 3
    max_additions: int = 1

    def __post_init__(self):
        if not 0 < self.min_coverage <= 1 or self.max_additions < 1:
            raise ValueError("invalid resolution policy")


@dataclass(frozen=True)
class BundleMatch:
    bundle_id: str
    coverage: float
    missing: tuple[str, ...]
    purpose_compatible: bool
    eligible: bool


@dataclass(frozen=True)
class BundleResolution:
    decision: Decision
    skill_ids: tuple[str, ...]
    purpose: str
    existing_bundle: Bundle | None = None
    add_skills: tuple[str, ...] = ()
    ambiguity: tuple[str, ...] = ()
    matches: tuple[BundleMatch, ...] = ()


def resolve_bundle(need: str, candidate_skill_ids: tuple[str, ...],
                   registry: BundleRegistry, *,
                   policy: ResolutionPolicy = ResolutionPolicy()) -> BundleResolution:
    ids = tuple(dict.fromkeys(candidate_skill_ids))
    if not need.strip() or not ids:
        raise ValueError("resolution requires a need and non-empty candidate composition")
    if set(ids) - registry.skill_ids:
        raise ValueError("unknown candidate skills")
    if len(ids) == 1:
        return BundleResolution(Decision.DIRECT, ids, need)
    touched = set().union(*(registry.skill_to_bundles.get(s, ()) for s in ids))
    matches = []
    # Deliberately conservative lexical purpose gate, not semantic equivalence.
    normalize = lambda text: " ".join(text.casefold().split())
    for bid in sorted(touched):
        bundle = registry.get(bid)
        missing = tuple(s for s in ids if s not in bundle.skill_ids)
        matches.append(BundleMatch(bid, (len(ids) - len(missing)) / len(ids), missing,
                                   normalize(need) == normalize(bundle.purpose),
                                   bundle.status == "active" and bundle.validated))
    reusable = [m for m in matches if m.eligible and not m.missing and m.purpose_compatible]
    extendable = [m for m in matches if m.eligible and m.purpose_compatible and
                  m.coverage >= policy.min_coverage and 0 < len(m.missing) <= policy.max_additions]
    pool = reusable or extendable
    if len(pool) == 1:
        match = pool[0]
        bundle = registry.get(match.bundle_id)
        composition = tuple(dict.fromkeys((*bundle.skill_ids, *match.missing)))
        return BundleResolution(Decision.REUSE if reusable else Decision.EXTEND,
                                composition, need, bundle, match.missing, matches=tuple(matches))
    return BundleResolution(Decision.CREATE, ids, need,
                            ambiguity=tuple(m.bundle_id for m in pool), matches=tuple(matches))


@dataclass(frozen=True)
class CapabilityResult:
    need: str
    discovery: SkillDiscoveryResult
    resolution: BundleResolution
    skills: tuple[SkillRecord, ...]
    warnings: tuple[str, ...]

    @property
    def decision(self) -> Decision:
        return self.resolution.decision


class CapabilityLoader:
    def __init__(self, discovery: SkillDiscovery, bundles: BundleRegistry, *,
                 policy: ResolutionPolicy = ResolutionPolicy()):
        self.discovery, self.bundles, self.policy = discovery, bundles, policy
        if set(discovery.records) != bundles.skill_ids:
            raise ValueError("discovery and bundles must share a skill corpus")

    def load_capability(self, need: str, *, k: int = 5) -> CapabilityResult:
        found = self.discovery.discover_skills(need, k)
        if not found.candidates:
            raise ValueError("no skills discovered for capability need")
        resolution = resolve_bundle(need, tuple(c.skill_id for c in found.candidates),
                                    self.bundles, policy=self.policy)
        warnings = ["Candidate composition is a retrieval assumption, not proven sufficiency."]
        if found.truncated:
            warnings.append("Candidate composition truncated by k.")
        if resolution.ambiguity:
            warnings.append("Multiple eligible bundles; CREATE is an unregistered proposal requiring review.")
        return CapabilityResult(need, found, resolution,
            tuple(self.discovery.records[s] for s in resolution.skill_ids), tuple(warnings))


def load_capability(need: str, *, loader: CapabilityLoader, k: int = 5) -> CapabilityResult:
    return loader.load_capability(need, k=k)
