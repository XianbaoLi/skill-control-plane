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
- `integrations`: native tool schemas and loop, Bundle/context injection, and
  canonical-to-model-visible History Projection.

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

Minimal Core construction:

```python
from skill_control_plane import (
    CapabilityMemory, DiscoverySession, SkillDiscovery, SkillStore,
)

store = SkillStore.from_tree("/path/to/skills")
discovery = SkillDiscovery(store)
memory = CapabilityMemory(store)
session = DiscoverySession(discovery, memory)
```

Corpus and evaluation commands are available through `skill-control-plane --help`.

The single authoritative architecture and terminology reference is
[V1 Runtime Architecture](docs/architecture/v1-runtime-architecture.md).
