# Skill Control Plane

Skill Control Plane is a small, inspectable control layer for discovering Agent
Skills, closing capability gaps within a user turn, and retaining compact
capability memory across turns. V1 keeps retrieval deterministic and keeps model
protocol concerns outside Core.

## V1 architecture

The current runtime has five layers:

```text
integrations
    ↓
runtime/discovery_session ──→ discovery ──→ registry
runtime/capability_memory ─────────────────→ registry
```

- `registry`: the Skill Store; parses and retains Skill metadata, source path,
  content hash and `SKILL.md` body, with exact `skill_id` lookup.
- `discovery`: Retrieval Cards, BM25/Dense/RRF, one search, compact candidates
  and retrieval trace.
- `runtime/discovery_session`: per-turn Candidate Closure, search budget,
  no-progress/repeated-query handling, sufficiency and candidate eligibility.
- `runtime/capability_memory`: cross-turn Bundles, Bundle Cards and resident or
  evicted Skill Body state.
- `integrations`: native tool schemas and loop, structured-context injection,
  JSON serialization, and canonical-to-model-visible History Projection.

`SkillControlPlane` is the only supported Agent-facing entry point. It composes
the Store, Discovery, Discovery Session, and Capability Memory behind structured
dataclass inputs and outputs; Core does not build prompts, tool schemas, provider
messages, or tool-call JSON. The production entry point fails closed unless every
Skill has a current Retrieval Card and Dense is configured, so its search backend
is always Retrieval Card v0.1 + BM25 + Dense + RRF. BM25-only construction is
retained solely for explicit low-level eval ablations.

`evals` is a sidecar validation system and is not part of the production runtime
chain. Historical V0.x implementations live under `skill_control_plane.evals.legacy`.

## Core flow

```text
user turn
  → Bundle-first check
  → uncovered gap: load_capability
  → one Discovery search; accumulate pending Candidates in Discovery Session
  → COVERED, SEARCH_MORE, or UNSATISFIED
  → apply_capability with DIRECT, CREATE, or EXTEND
  → exact Skill Body delivery
  → Bundle memory update for CREATE/EXTEND only
  → History Projection hides evicted or superseded body text from the model
```

DIRECT is a temporary activation in the current context; it does not create or
extend long-term Bundle memory.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Supported production construction (the embedding client reads its provider
credentials from the environment):

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

Missing, extra, content-hash-stale or wrong-version Cards, or a missing Dense
backend, are configuration errors. There is no production BM25-only fallback.

`RuntimeCapabilityHarness` remains importable only from
`skill_control_plane.runtime.capability_harness` for historical experiments. It
is deprecated, retains the old combined resolver loader where needed, and is not
used by the Reference Agent or exported as public API.

Corpus and evaluation commands are available through `skill-control-plane --help`.

The single authoritative architecture and terminology reference is
[V1 Runtime Architecture](docs/architecture/v1-runtime-architecture.md).
