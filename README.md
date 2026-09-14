# Skill Control Plane

Skill Control Plane is a runtime control layer for large Agent Skill libraries. It keeps capability discovery, runtime rerouting, and cross-turn capability memory outside the main Agent loop, so an existing Agent can gain on-demand Skills without loading the whole Skill corpus into every prompt.

The current v0.1 release integrates with Pi through a TypeScript adapter and a Python NDJSON sidecar. The Core keeps retrieval and lifecycle state explicit and inspectable: Retrieval Card + BM25/Dense/RRF for discovery, `CapabilityMemory` for cross-turn Bundles, and an evidence-driven `RerouteController` for discovering new capabilities when runtime evidence shows that the current capability set is insufficient.

## Why this exists

Large Skill libraries create two practical problems:

1. exposing too many Skills increases prompt/context cost and makes capability choice harder;
2. a long-running Agent can discover a new subgoal or failure after the initial plan, so one-shot Skill selection is not enough.

Skill Control Plane addresses both by narrowing the Skill surface before selection, retaining only compact capability state across turns, and rerouting on structured runtime evidence instead of relying on the Main Agent to remember to search again.

## Architecture

```text
Pi Agent Loop
    │
    │ tool calls / tool results / lifecycle events
    ▼
TypeScript Pi Adapter
    │
    │ private stdio NDJSON
    ▼
Python Sidecar
    │
    ▼
SkillControlPlane
    ├── RerouteController
    │     RuntimeEvidence
    │       → CapabilityGapDecider
    │       → automatic capability discovery
    │       → candidate projection
    │       → SELECTED / COMMITTED telemetry
    │
    ├── DiscoverySession
    │     per-turn Candidate Closure / search budget / apply eligibility
    │
    ├── CapabilityMemory
    │     cross-turn Bundles / resident-evicted Skill Body lifecycle
    │
    └── Discovery
          Retrieval Cards + BM25 + Dense + RRF
                 │
                 ▼
              Skill Store
```

The Agent loop itself is not forked or replaced. Pi remains the host Agent; the adapter translates host events into the Control Plane protocol and projects compact capability state back to the model.

## Pi Adapter + Python sidecar

`adapters/pi/` is the production Pi integration. It owns host-specific concerns such as tool schemas, lifecycle mapping, context injection, History Projection, and conversion of tool/runtime failures into structured `RuntimeEvidence`.

The adapter talks to the Python runtime through a private NDJSON stdio sidecar:

```bash
python -m skill_control_plane.sidecar \
  --skill-root /path/to/skills \
  --retrieval-cards /path/to/retrieval-cards-v0.1.jsonl \
  --dense-index /path/to/dense-index-v1.json
```

Core semantics remain provider- and host-independent. A future Agent integration only needs an adapter that maps host events and tool calls onto the same public runtime contract.

## Evidence-driven rerouting

Dynamic capability discovery is handled by `RerouteController`, not by an ad-hoc prompt checkpoint.

```text
RuntimeEvidence
   ↓
deterministic eligibility + per-evidence dedup
   ↓
CapabilityGapDecider
   ├── NO_GAP → continue with current capabilities
   └── GAP_FOUND
          ↓
     automatic search_capability()
          ↓
     BM25 + Dense + RRF candidate pool
          ↓
     Main Agent selects Skill(s)
          ↓
     apply_capability(reroute_evidence_id=...)
          ↓
     SELECTED → COMMITTED
```

The responsibility split is deliberate:

- Runtime decides **when** capability sufficiency must be reconsidered.
- `CapabilityGapDecider` decides **whether** a capability is missing and emits a concise capability need.
- Discovery finds candidate Skills.
- The Main Agent decides **which** candidates to activate and whether to `CREATE`, `EXTEND`, or use scoped `DIRECT`.
- `CapabilityMemory` owns persistent Bundle and Skill-body lifecycle.

Reroute telemetry is keyed by evidence ID, so `NO_GAP`, `DISCOVERED`, `SELECTED`, `COMMITTED`, and typed failures are runtime facts rather than reporter-side inference.

See [RerouteController v0.1](docs/architecture/reroute-controller-v0.1.md).

## Benchmark results

The v0.1 benchmark compares Pi Native Skills with the Control Plane under frozen Skill corpora and shared task/verifier definitions.

### Scaling benchmark

At the largest tested corpus size (`S128`, 16 tasks per arm):

| Metric | Pi Native Skills | Skill Control Plane |
|---|---:|---:|
| Task success | 68.75% | 68.75% |
| Required-Skill recall | 12.50% | 56.25% |
| Mean total LLM tokens | 49.4k | 20.6k |

That is about **58% fewer total LLM tokens at S128** while task success remains equal in this benchmark slice. Across the full S32/S64/S128 scaling run, task success was approximately equal overall while the Control Plane showed substantially higher explicit Skill recall and materially lower token growth as the corpus expanded.

These results support the current v0.1 claim: the Control Plane's clearest benefit is **capability recall and context/token scaling**, not a universal task-success improvement.

### Reroute component evaluation

The frozen v0.1 component evaluation isolates gap detection from final candidate selection:

| Component | Result |
|---|---:|
| GapDecider accuracy / precision / recall / F1 | 1.000 / 1.000 / 1.000 / 1.000 |
| Generated-need target recall @5 | 1.000 |
| Candidate selection micro precision | 0.842 |
| Candidate selection micro recall | 0.941 |
| Candidate selection micro F1 | 0.889 |
| Exact required-set coverage | 10 / 11 |
| Retrieval misses in selector set | 0 |

The sample is intentionally small and one-repeat. It is diagnostic evidence, not a claim that gap detection or retrieval is universally solved. Candidate selection remains the main known quality-optimization area.

See [Reroute component evaluation v0.1](docs/experiments/reroute-component-eval-v0.1.md) and [benchmark design](docs/experiments/benchmark-v0.1-design.md).

## Core runtime model

The production runtime has five layers:

- `registry`: canonical Skill Store with exact `skill_id`, source path, content hash, metadata, and `SKILL.md` body.
- `discovery`: Retrieval Cards, BM25, Dense, RRF, compact candidate surfaces, and retrieval traces.
- `runtime/discovery_session`: per-turn pending Candidates, Candidate Closure, search budget, no-progress handling, and apply eligibility.
- `runtime/capability_memory`: cross-turn Bundles, Bundle Cards, and resident/evicted Skill Body state.
- `integrations`: host-specific adapters, tool schemas, context injection, serialization, and History Projection.

`SkillControlPlane` is the supported Agent-facing Core facade. Production construction fails closed unless Retrieval Cards are complete/current, Dense is configured, and the fusion path is RRF. The low-level BM25-only path remains available only for explicit evaluation ablations.

## Capability lifecycle

A successful application uses one of three actions:

- `CREATE`: create a maintained Bundle from selected candidates.
- `EXTEND`: add maintained members to an existing Bundle.
- `DIRECT`: attach a clearly one-off or short-lived capability to a reasonable existing Bundle.

Skill bodies have `resident` and `evicted` states. Bundle metadata survives body eviction, so a known evicted Skill can be reloaded exactly by `skill_id` without running retrieval again.

`StateSnapshotV1` persists `CapabilityMemory` only. Turn-local `DiscoverySession` state and per-evidence reroute state are intentionally not persisted.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Pi adapter validation:

```bash
npm --prefix adapters/pi install
npm --prefix adapters/pi test
npm --prefix adapters/pi run typecheck
```

Supported production construction:

```python
from skill_control_plane import SkillControlPlane
from skill_control_plane.discovery.bigmodel import (
    BigModelDenseRetriever,
    BigModelEmbeddingClient,
)

embedding = BigModelEmbeddingClient()
control_plane = SkillControlPlane.from_tree(
    "/path/to/skills",
    retrieval_cards="/path/to/retrieval-cards-v0.1.jsonl",
    dense_factory=lambda records: BigModelDenseRetriever(
        records,
        model_name=embedding.model,
        dimensions=embedding.dimensions,
        embed_batch=embedding,
    ),
)

control_plane.begin_turn()
snapshot = control_plane.context_snapshot()
```

Provider credentials are read from environment variables; `.env.example` contains placeholders only.

## Repository map

```text
src/skill_control_plane/
  registry/           canonical Skill Store
  discovery/          Retrieval Cards + BM25/Dense/RRF
  runtime/            SkillControlPlane, DiscoverySession,
                      CapabilityMemory, RerouteController
  sidecar/            NDJSON runtime server
  integrations/       reference Agent integration

adapters/pi/           TypeScript Pi adapter
evals/                 frozen corpora, gold, benchmark definitions
scripts/               benchmark/evaluation runners
docs/architecture/     current architecture contracts
docs/experiments/      experiment reports and diagnostics
```

The authoritative architecture reference is [V1 Runtime Architecture](docs/architecture/v1-runtime-architecture.md).

## Release status

`v0.1.0` freezes the core architecture and benchmark methodology. The core runtime, Pi adapter/sidecar integration, CapabilityMemory lifecycle, hybrid discovery, and evidence-driven rerouting are complete for this release. Further candidate-selection robustness and quality tuning are intentionally deferred.

License: MIT.
