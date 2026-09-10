# Runtime Capability Harness

For the single-model, same-history experiment with Skill bodies delivered in history,
see [Experimental Skill Agent](experimental-skill-agent.md). It uses separate
search/apply stages; the combined resolver API documented below remains available.

The harness maintains the current capability state and renders the capability
context for every LLM turn. The task LLM decides when its next step requires
missing capabilities and calls `load_capability(need)`. Retrieval searches only
the Skill library; the resolver organizes selected skills against **all** bundles
maintained in the current runtime.

```text
RuntimeCapabilityState
  → RuntimeCapabilityHarness.render_context()
  → task LLM: direct skills + all maintained bundles + load_capability interface
  → normal task execution; if next-step capabilities are missing:
      load_capability(need)
        → RuntimeCapabilityLoader
        → SkillDiscovery: Retrieval Card / BM25 / optional Dense + RRF
        → LLMCapabilityResolver:
            need + skill candidates + representations/evidence
            + all maintained bundles from current state
        → strict validation: DIRECT / EXTEND / CREATE
        → new RuntimeCapabilityState
        → harness adopts successful result
  → render_context() for the next task LLM turn
```

## Entry point and host contract

```python
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.runtime.capability_loading import RuntimeCapabilityLoader
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness

registry = SkillRegistry.from_tree(
    'local_artifacts/v0.5/hermes-current87',
    cards=load_retrieval_cards('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl'),
)
harness = RuntimeCapabilityHarness(RuntimeCapabilityLoader(SkillDiscovery(registry)))
context = harness.render_context()
# Host includes context each turn and binds the tool to harness.load_capability.
# The task LLM supplies this need only when the next step requires it:
result = harness.load_capability('Extract text from a PDF and make presentation slides')
next_context = harness.render_context()
assert harness.state is result.resulting_state
```

`render_context()` returns short instructions followed by compact JSON containing:

- `direct_skills`: current direct skill IDs;
- `maintained_bundles`: every current bundle's `bundle_id`, `purpose`, `skill_ids`;
- `tools`: the `load_capability` interface with a required string `need` parameter.

The self-trigger rule asks the task LLM to call only when current capabilities
cannot support the next step. `need` describes missing capability without guessing
Skill or Bundle names. The harness never applies a deterministic trigger. The
host is responsible for passing each rendered context to its task LLM and binding
its tool calls; this layer does not implement a provider-specific task agent loop.

Rendering sorts direct IDs, bundles by ID, and member IDs, so equivalent states
produce identical context irrespective of collection order. It performs no search
or model call and does not expose registry contents, retrieval cards, or skill
bodies. Capability JSON is labelled as data rather than instructions. The host's
skill execution mechanism remains responsible for consuming loaded skills.

Use the modules under `skill_control_plane.runtime.capability_*` for this design.
The older `skill_control_plane.capability_loading`, package-root exports, and
Shelf / `discovery_surface()` experiments remain for compatibility and are not
used in this runtime call chain. Historical experiment reports remain unchanged.

## Retrieval and resolver boundary

`SkillDiscovery` is unchanged: its index uses Retrieval Card representations (or
existing metadata fallback), BM25, and optional Dense with RRF when a dense factory
is supplied. Default skill candidate depth is 10. Loading performs one global
Skill search, with no bundle search, bundle index, bundle ranking, or bundle Top-K.
`BundleCandidate`, `discover_active_bundles`, and the `bundle_k` argument have been
removed from this API.

`resolve_capability(need, skill_candidates, runtime_state)` receives the discovery
result, including candidate representation/evidence, and current state. Its prompt
contains all maintained bundles as plain ID/purpose/member records, even when they
have no lexical overlap with the need. Only retrieved Skill candidates and their
representations are exposed, never the complete Skill library. Existing bundle
members need not appear among candidates; newly selected skills must appear there.

The default resolver uses the existing `BigModelChatClient` and `BIGMODEL_*`
configuration. Tests inject the existing `TextCompleter` or a stub resolver. There
is no new provider abstraction, sufficiency pass, or required-skill pruning.

## State, actions, and atomic application

`RuntimeCapabilityState` contains `active_bundles` (the maintained runtime bundles)
and `direct_skills`. Immutable `ActiveBundle` records contain ID, purpose and skill
IDs. Bundles may coexist and share skills. State validation requires real registry
skill IDs, unique bundle IDs, and unique member IDs.

- DIRECT adds one or more candidate skills to direct skills without maintaining
  them in a bundle. Skill count does not decide the action.
- EXTEND adds selected candidate skills to any existing maintained bundle. Its
  target must exist in current state and it must add at least one new member.
- CREATE allocates a runtime UUID bundle ID and records a non-empty reusable
  purpose and selected candidate skills.

All actions preserve other bundles and direct skills. Duplicate selections are
deduplicated. Unknown actions, extra fields, duplicate JSON keys, invented or
out-of-candidate skill IDs, invalid targets, empty selections/purposes/reasons,
and no-op EXTEND fail validation. Custom resolver decisions are revalidated before
application. Application constructs a new state without mutating the supplied
state; the harness adopts it only after a successful load. Direct loader callers
can still explicitly adopt `result.resulting_state`.

Malformed JSON/schema permits one repair call. A second failure raises
`ResolverError` with attempts. Provider/transport exceptions propagate; empty
candidate results fail explicitly. These failures preserve the state and next
rendered context. Load results retain need, skill retrieval/evidence, the full
input `maintained_bundles` snapshot, resolver attempts/decision, affected bundle
ID, and resulting state. The existing adapter may canonicalize provider JSON;
recorded responses are adapter output, not original HTTP response bytes.

## Verification and boundaries

Network-free pytest covers stable and library-scoped context, empty initial state,
all maintained bundles beyond the former Top-K, evidence in resolver input,
non-lexical EXTEND targets, multi-skill DIRECT, CREATE, EXTEND, turn-to-turn context
updates, strict validation, one repair, and unchanged state/context on failure.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_runtime_capability_loading.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The separate `scripts/capability_loading_e2e.py` remains an opt-in live resolver
acceptance script using the frozen current87 registry/cards and configured Dense
provider plus RRF. It aborts on Dense preflight failure and reports all maintained
bundles; live calls are not part of ordinary pytest.

Self-trigger behavior is specified in the context; actual task-model judgment is
not guaranteed by schema validation. This minimal harness does not execute skills
or external actions. Bundle retrieval, sufficiency re-retrieval, deterministic
hard triggers, Pi/OpenPI integration, bundle eviction/merge/long-term lifecycle,
persistence and concurrent state coordination are outside this implementation.
