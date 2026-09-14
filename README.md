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

The v0.1 evaluation compares Pi Native Skills with the Control Plane under frozen Skill corpora and shared task/verifier definitions. The results are best read as four separate system claims: **scaling efficiency, retrieval quality, cross-turn capability reuse, and dynamic rerouting**.

### 1. Scaling efficiency — keep model cost nearly flat as the Skill corpus grows

The main scaling experiment runs the same 16 tasks against `S32`, `S64`, and `S128` Skill corpora.

| Corpus | Native tokens | Control Plane tokens | Token reduction | Native task success | Control Plane task success |
|---|---:|---:|---:|---:|---:|
| S32 | 24.5k | 20.7k | 15.7% | 81.25% | 75.00% |
| S64 | 34.9k | 17.8k | 49.1% | 75.00% | 81.25% |
| S128 | 49.4k | 20.6k | 58.4% | 68.75% | 68.75% |

As the Skill corpus grows **4× from 32 to 128 Skills**, Native token usage rises from 24.5k to 49.4k (**+101.6%**), while Control Plane usage stays essentially flat at 20.7k to 20.6k (**-0.5%**). Averaged across the three corpus sizes, the Control Plane uses about **19.7k vs 36.3k tokens (-45.8%)**.

Task success is approximately equal overall across the scaling suite (~75% for both arms). The strongest scaling claim is therefore not universal outcome improvement; it is that retrieval decouples the Main Agent's context cost from Skill-library size while preserving comparable task completion.

### 2. Retrieval quality — improve required-Skill recall instead of merely shrinking context

The token reduction does not come from blindly hiding Skills. The Control Plane retrieves the required capability more reliably as the corpus gets harder:

| Corpus | Native required-Skill recall | Control Plane recall | Relative gain |
|---|---:|---:|---:|
| S32 | 50.00% | 68.75% | 1.38× |
| S64 | 31.25% | 50.00% | 1.60× |
| S128 | 12.50% | 56.25% | **4.50×** |
| Overall | 31.25% | 58.33% | **1.87×** |

Across the scaling run, Skill-selection precision remains high as well: approximately **87.5% for the Control Plane vs 83.3% for Native**. At S128, the headline result is therefore **4.5× required-Skill recall with 58.4% fewer LLM tokens and equal task success**.

The capability representation contributes to that retrieval quality. On the frozen representation ablation, replacing metadata-only Dense retrieval with Retrieval Card v0.1 raises Dense `Recall@5` from **84.6% to 92.3% (+7.7 percentage points)**; BM25 + Dense RRF also reaches **92.3% Recall@5**. Retrieval Cards encode `purpose`, `use_when`, `capabilities`, and `lexical_cues` specifically for capability discovery rather than treating the raw Skill body as the only search representation.

### 3. Cross-turn CapabilityMemory — avoid paying for the same capability twice

Bundles are not only organizational metadata. `CapabilityMemory` persists which capabilities have already been acquired, allowing a later turn to reuse an active Bundle instead of rediscovering and reactivating the same Skill.

A targeted reuse ablation (`RT-R04`, second turn) shows the mechanism clearly:

| T2 metric | Memory ON | Memory OFF |
|---|---:|---:|
| Rediscovery calls | 0 | 1 |
| Reactivations | 0 | 1 |
| Skill-body loads | 0 | 0 |
| LLM tokens | **6,481** | **33,822** |

That is an **80.8% reduction in second-turn tokens** while eliminating both rediscovery and reactivation. This is mechanism/cost evidence rather than an outcome claim: both arms had the same task outcome in this case.

The benefit is also intentionally described as case-dependent. In `RT-R03`, Pi's existing conversation history already preserved enough context that neither arm needed rediscovery; memory reduced second-turn tokens only from 9,233 to 7,886 (**14.6%**). The intended role of `CapabilityMemory` is therefore to provide an explicit, compression-resistant capability state when host history alone is insufficient, not to claim a fixed saving on every turn.

### 4. Dynamic rerouting — recover capability gaps discovered after execution starts

A one-shot Skill choice at the beginning of a task cannot cover every failure or newly exposed subgoal. `RerouteController` turns runtime failures/subgoals into structured `RuntimeEvidence`, decides whether they reveal a capability gap, and automatically invokes the existing retrieval path when needed.

The frozen reroute component evaluation separates gap detection, retrieval, and final candidate selection:

| Component metric | Result |
|---|---:|
| Eligible GapDecider cases | 11 |
| TP / TN / FP / FN | 7 / 4 / 0 / 0 |
| GapDecider precision / recall / F1 | **1.000 / 1.000 / 1.000** |
| Generated-need target Recall@5 | **1.000** |
| Generated-need target Recall@10 | **1.000** |
| Candidate selection micro precision | 0.842 |
| Candidate selection micro recall | **0.941** |
| Candidate selection micro F1 | 0.889 |
| Exact required-set coverage | **10 / 11** |
| Retrieval misses in selector set | **0** |

All seven positive GapDecider cases progressed through `NEW → CHECKING → GAP_FOUND → DISCOVERED`; all four eligible negatives terminated as `NO_GAP`. Every positive generated capability need recovered its required Skill in the top five candidates. The remaining error source in this small one-repeat diagnostic is final candidate selection, not retrieval: one required Skill was missed and three extra Skills were selected across the selector set.

This supports the runtime claim that the system can **reconsider and acquire capabilities after execution has already started**, while keeping the responsibilities separated: Runtime decides *when* to reconsider, GapDecider decides *whether/what* is missing, retrieval finds candidates, and the Main Agent decides *which* candidates to apply.

See [Reroute component evaluation v0.1](docs/experiments/reroute-component-eval-v0.1.md), [RerouteController v0.1](docs/architecture/reroute-controller-v0.1.md), and [benchmark design](docs/experiments/benchmark-v0.1-design.md).

Taken together, the benchmark supports a deliberately narrow v0.1 claim: **the Control Plane scales capability access more efficiently than exposing a growing Skill surface directly to the model, improves required-Skill recall, can reuse previously acquired capabilities across turns, and can rediscover missing capabilities from runtime evidence.** It does not claim universal task-success improvement.

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
