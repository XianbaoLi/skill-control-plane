# Capability Loading V0.7

LLM requests capability;
Skill is the atomic capability unit;
Bundle is a reusable capability composition;
Control Plane resolves how the capability is provided.

The runtime supplies a natural-language need to `loader.load_capability(need)`.
It does not choose between skill discovery and bundle discovery. Loading here
means resolving records and a composition proposal, not executing skill code.

## Architecture

```text
SKILL.md → canonical SkillRecord + content_hash
         → offline RetrievalCard → prepared registry representation
                                      ↓
need → discover_skills → candidates → Bundle Resolver → CapabilityResult
                                      ↑
                       Bundle Registry / skill_to_bundles
```

### Skill Registry

The existing `SkillRecord` and `SkillRegistry` are extended, not replaced.
`source_path` is the skill path; canonical name, description, body and content
hash remain intact. `retrieval_representation` stores prepared retrieval text.
`SkillRegistry.from_tree(root, cards=cards)` reuses the existing loader and
`apply_retrieval_cards`, including corpus/hash validation. Without cards,
registration explicitly falls back to legacy name/description/tags metadata.

Generate cards offline using the existing `skill-control-plane corpus retrieval-cards` CLI workflow
and load its JSONL with `load_retrieval_cards`. No query generates a card or
rereads SKILL.md. `SkillDiscovery` is an index snapshot; rebuild it after registry
changes. Registries are in-memory; durable storage and automatic refresh are
outside V0.7.

### Skill Discovery

`SkillDiscovery.discover_skills(query, k=5)` returns ranked existing
`RetrievalCandidate` objects, scores, source evidence, matched representation
text, backend and truncation information. Existing BM25 is the dependency-free
default. Supplying a dense factory enables existing Dense + BM25 + existing
reciprocal rank fusion (RRF constant 60). Source depth defaults to 10 and grows
to at least k+1 to expose truncation. Both retrievers index identical prepared
metadata/card text. Existing local Dense and BigModel backends can be injected;
credentials and model configuration remain outside this API.

Representations are the evidence; there is no separate evidence store. Retrieval
scores are ranking signals, not calibrated confidence or proof of sufficiency.

### Bundle Registry

`Bundle(bundle_id, purpose, skill_ids, status, validated)` represents a purpose
and immutable composition. Membership must contain unique known skill IDs.
`BundleRegistry` maintains a read-only `skill_to_bundles` inverted view; multiple
bundles may share each skill. Duplicate bundle IDs are rejected. Status is one
of `active`, `draft`, or `inactive`; validation defaults to false. Only active,
validated bundles are resolution candidates.

The earlier `runtime.bundles.CapabilityBundle` is a category-grouped stage
working set with a rank, descriptor and session activation state. It remains
unchanged. It is not silently promoted to a validated purpose composition.
This separate record is intentional; category grouping is not purpose matching.

### Bundle Resolver

`resolve_bundle(need, candidate_skill_ids, registry)` deduplicates the candidate
composition and uses inverted membership lookup. It exposes coverage, missing
skills, purpose compatibility and eligibility for every overlapping bundle.

| Decision | Deterministic rule | Result |
| --- | --- | --- |
| DIRECT | Exactly one candidate | Atomic record, no bundle |
| REUSE | Exactly one eligible bundle covers all candidates and matches purpose | Existing bundle, including its full composition |
| EXTEND | No REUSE; exactly one purpose-compatible eligible bundle covers at least 2/3 of candidates and lacks at most one skill | Existing bundle plus `add_skills` |
| CREATE | No eligible match or multiple eligible matches | Unregistered purpose/composition proposal |

Purpose compatibility deliberately means equality after case folding and
whitespace normalization. It is conservative and does not infer synonymy from
one shared word. REUSE therefore also requires purpose compatibility: identical
membership alone does not prove that two compositions serve the same purpose.
`ResolutionPolicy` exposes minimum coverage and maximum additions. There is no
hidden overlap ranking or arbitrary ID tie-break: multiple qualifying bundles
are reported in `ambiguity`, and the CREATE proposal requires review. REUSE has
priority over EXTEND. Draft or inactive entries cannot be reused or extended.

CREATE and EXTEND never mutate the registry. CREATE has no invented permanent
bundle ID; purpose and skill_ids in the resolution are its proposed definition.
Empty retrieval raises `ValueError` instead of proposing an empty capability.

### High-level API

```python
from skill_control_plane import CapabilityLoader, SkillDiscovery, BundleRegistry
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.retrieval.dense import DenseRetriever

skills = SkillRegistry.from_tree(
    'fixtures/skills', cards=load_retrieval_cards('cards.jsonl')
)
discovery = SkillDiscovery(
    skills,
    dense_factory=lambda records: DenseRetriever(records),
)
bundles = BundleRegistry(skills)  # Add explicitly validated compositions here.
loader = CapabilityLoader(discovery, bundles)
result = loader.load_capability('Diagnose a Python environment failure')
```

Setup belongs to the host. Once configured, the runtime only invokes
`load_capability(need)`. Module-level convenience functions are also exported;
`load_capability(need, loader=loader)` avoids process-global mutable state.
`CapabilityResult` includes discovery, resolution, canonical selected skill
records (including original paths/body/hash), and explicit warnings. No new CLI
is required for the Python runtime boundary; historical CLI experiments remain
unchanged.

## Existing architecture mapping

| Existing implementation | V0.7 responsibility |
| --- | --- |
| registry/loader.py, registry/store.py, models.SkillRecord | Canonical registration and cached representation |
| retrieval/cards.py | Offline card generation/cache, hash validation, enhanced projection |
| retrieval/bm25.py, dense.py, bigmodel.py, fusion.py | Existing retrieval engines, wrapped by retrieval/discovery.py |
| runtime/bundles.py, hierarchical.py, soft_bundle.py | Preserved experimental stage/shelf routing; not the V0.7 resolver |
| runtime/capability_need.py, capability_facets.py, stage_retrieval.py | Preserved upstream need/facet extraction and reroute contracts |
| evals/retrieval_ablation.py, representation_ablation.py, query_robustness.py | Preserved frozen retrieval evaluation and field ablation |

Repeated retriever assembly exists in historical CLI/evaluation paths. V0.7
centralizes new API assembly without migrating those experiments or changing
frozen 87-skill, 65-query and 39-query assets. No retrieval algorithm is copied.

## Acceptance and limits

`tests/test_capability_loading.py` exercises real BM25 discovery through the
high-level API for all four decisions, plus card registration/hash rejection,
shared membership, ambiguous bundles, purpose/status gates, coverage boundaries,
input errors, truncation and existing Dense/RRF composition using an offline
encoder double. The existing retrieval and stage/reroute suite remains required.

The candidate set is treated as a proposed composition. Top-k retrieval can
include alternatives or omit necessary skills; even DIRECT does not prove that
one candidate fully satisfies a free-form need. Every result discloses this
assumption, with a further warning when truncated. Dense retrieval may return
weak matches; no calibrated relevance filter or semantic sufficiency judge is
claimed. The legacy BM25 tokenizer primarily supports Latin tokens; use the
existing multilingual dense backend for appropriate corpora. Exact purpose
matching favors safe CREATE proposals over potentially incorrect reuse.

V0.7 excludes OpenPI integration, automatic triggers, execution, automatic
permanent registration, long-term learning, bundle embeddings, merge/refinement,
complex lifecycle and self-evolution. Later hosts may adapt runtime evidence to
need strings and consume these result records without changing the entry point.
