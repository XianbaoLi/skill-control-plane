# V1 Runtime Architecture

> Status: current and authoritative. This is the only architecture fact source
> for the V1 production runtime. Documents under `docs/archive`, `docs/experiments`
> and `docs/handoffs` are historical or eval-only evidence.

## Scope and dependency direction

V1 narrows the production path to five layers and one sidecar:

```text
integrations
    ├── tool schemas and native tool loop
    ├── Bundle/context injection
    └── History Projection
              ↓
runtime/discovery_session ──→ discovery ──→ registry
              │                              ↑
              └── coordinates ──→ runtime/capability_memory

evals ──→ production modules       (sidecar only)
```

Allowed dependencies are:

- `integrations -> runtime -> discovery -> registry`;
- `runtime/capability_memory -> registry`;
- `evals -> production modules`.

The following are forbidden and enforced by tests:

- `registry -> runtime`;
- `discovery -> capability_memory`;
- `runtime -> integrations`;
- production Registry, Discovery, Runtime or Integrations code importing
  `evals`, experiments or demos.

The CLI exposes corpus and evaluation utilities. It is not an additional runtime
layer. OpenPI/Hermes migration and a unified Agent API are outside V1.

## Canonical entities

### Skill

The atomic stored capability. A Skill has an exact `skill_id`, name, description,
tags/category metadata, source path, content hash/version identity and the parsed
`SKILL.md` body. `SkillStore` owns it; retrieval ranking and Bundle lifecycle do not.

### Retrieval Card

Offline, hash-bound metadata describing purpose, use conditions, capabilities and
lexical cues for one Skill. It is a Discovery index artifact, not part of the
canonical Skill Store and never replaces the Skill Body.

### Candidate

A Skill reference returned by one search, with rank, source scores and minimal
retrieval evidence. A Candidate is evidence, not proof of capability sufficiency
and not a loaded Skill Body.

### Discovery Session

All mutable search state for one user turn: pending Candidates, Candidate Closure,
search count/budget, repeated queries, no-progress detection, remaining gaps,
sufficiency transitions, and validation that applied Skills came from the pending
Candidate set. A new turn resets this state.

### Bundle

A cross-turn maintained capability composition with a stable ID, reusable purpose
and exact member `skill_id`s. A Bundle is not a filesystem category, a retrieval
result group, or an ordinary Skill grouping.

### Bundle Card

The compact, compression-resistant model surface for a Bundle: ID, purpose,
bounded capability phrases, member identity and each member's latest body state.
It contains no full Skill Body.

### Skill Body

The instruction content parsed from `SKILL.md`. It is delivered only after an
eligible apply or exact reload. The current body state controls model visibility.

## Canonical policies

### Retrieval First

Search narrows the Store before semantic selection. Discovery performs one BM25
search and, when configured, one Dense search, combining them with RRF. It returns
a Compact Candidate Surface and retrieval trace without mutating turn or Bundle
state.

### Candidate Closure

Multiple searches in a turn accumulate a deduplicated pending Candidate pool.
Application may select only from that pool. Search failure or rejected application
must not partially commit Bundle memory. Successful application consumes the pool.

### Bundle-first Trigger

At every user turn, inspect current Bundle Cards first. Search only for an explicit
gap not covered by current Bundles. Continuing, revising or retrying the same
capability does not itself justify rediscovery.

### Metadata-first Reload

For an evicted Bundle member, decide from its Bundle Card and Skill metadata whether
the current step needs detailed instructions. If so, call `load_skill_body` by exact
known `skill_id`; do not run retrieval. If metadata is sufficient, continue without
reloading. Evicted means body text is unavailable to the model, not merely flagged.

### History Projection

Integrations retain canonical tool-call history, then derive model-visible history
for each request. Projection preserves native assistant/tool pairing and other
fields, shows only the newest resident body occurrence, and replaces evicted or
superseded bodies with tombstones. Canonical history is never mutated by projection.

## States, outcomes and actions

Skill Body states:

- `resident`: the latest applicable body may be visible in projected history;
- `evicted`: every historical occurrence is tombstoned and exact reload is required
  when detailed instructions are needed.

Discovery Session outcomes:

- `COVERED`: Bundle coverage plus eligible pending Candidates cover the current need;
- `SEARCH_MORE`: a concrete remaining gap and search budget remain;
- `UNSATISFIED`: a gap remains after available candidates/budget cannot cover it.

Application actions:

- `CREATE`: create a maintained Bundle with purpose and selected Candidate members;
- `EXTEND`: add at least one new selected Candidate to an existing Bundle;
- `DIRECT`: activate selected bodies temporarily in current model context without
  creating, extending or writing a long-term Bundle.

There is no current `REUSE` apply action. Bundle reuse is the zero-discovery path.

## Layer contracts

### Registry — Skill Store

`registry/loader.py` parses Skill frontmatter/body and computes the normalized
content hash. `registry/store.py` stores canonical Skills, rejects duplicate IDs,
and supports exact lookup/body reads. Registry does not import or construct
Retrieval Cards, rankings, Bundles, turn state or model history.

### Discovery — Capability Discovery

`discovery/cards.py`, retrievers and fusion own Retrieval Cards, BM25, Dense and
RRF. `SkillDiscovery.discover_skills` is one immutable-index search. It emits the
compact candidate payload and trace, but owns no pending pool, budget, sufficiency
or Bundle commit.

### Runtime — Discovery Session

`runtime/discovery_session.py` is the authority for per-turn search control,
Candidate Closure, remaining gaps, `COVERED / SEARCH_MORE / UNSATISFIED`, and apply
eligibility. Model-provided sufficiency fields are rejected; Core derives outcomes
from validated behavior.

### Runtime — Capability Memory

`runtime/capability_memory.py` is the authority for maintained Bundles, Bundle
Cards, member body residency, eviction and metadata-first exact reload. It reads
canonical records only through Skill Store. DIRECT activations may appear in the
compatibility state/context surface, but never in maintained Bundle Cards.

### Integrations

`integrations/reference_agent.py` supplies the current three native tools:
`load_capability`, `apply_capability`, and `load_skill_body`. It owns the provider
loop, canonical history, context injection, tombstones and model-visible History
Projection, while forwarding all three tools to Core. `ReferenceSkillAgent` is a
reference/example integration; its historical `ExperimentalSkillAgent` alias is
retained for experiment reproducibility and is not a Core state owner.

`RuntimeCapabilityHarness` is a thin forwarding facade over `DiscoverySession`
and `CapabilityMemory`; it does not duplicate their state.

## Runtime sequence

```text
new user turn
  1. Integration asks Core to begin a Discovery Session.
  2. Integration injects Bundle Cards and projected canonical history.
  3. Model follows Bundle-first Trigger.
  4. If a gap exists, load_capability delegates one search to Discovery.
  5. Discovery Session accumulates Candidates and derives SEARCH_MORE.
  6. Model either searches a distinct residual gap, reports UNSATISFIED, or calls
     apply_capability with coverage and remaining gaps.
  7. Discovery Session validates Candidate eligibility and action fields.
  8. Capability Memory atomically CREATEs/EXTENDs a Bundle or returns DIRECT bodies.
  9. Later compression may evict bodies while Bundle Cards survive.
 10. A needed known body is reloaded exactly; History Projection exposes only the
     latest resident occurrence.
```

There is no independent Resolver LLM in the V1 path. Capability sufficiency and
organization are expressed through the same model's native calls and Core validation.

## Eval isolation and cleanup inventory

Classification used for the V1 cleanup:

- **KEEP**: `registry/{loader,store}.py`, `discovery/**`,
  `runtime/{discovery_session,capability_memory,capability_harness}.py`,
  `integrations/**`, current corpus helpers, CLI, tests and fixtures.
- **MERGE**: registry-owned Retrieval Card/index preparation moved into Discovery;
  Harness pending/search/sufficiency moved into Discovery Session; Harness Bundle
  and body state moved into Capability Memory; experimental agent tool/history
  logic moved into the reference Integration.
- **EVAL-ONLY**: `skill_control_plane/evals/**`, top-level `evals/**`, `scripts/**`
  and experiment fixtures/artifacts.
- **LEGACY**: `skill_control_plane/evals/legacy/**`, `skill_control_plane/evolution/**`,
  archived V0.x architecture documents and historical handoffs.
- **DELETE-CANDIDATE**: none. Uncertain historical material was retained and labeled
  rather than deleted.

Production modules do not import legacy/eval-only code. Evals may import production
modules to validate behavior.

## Verification

```bash
PYTHONPATH=src .venv/bin/pytest -q
python -m compileall -q src tests scripts
git diff --check
```

Architecture boundary tests live in `tests/test_v1_architecture_boundaries.py`.
