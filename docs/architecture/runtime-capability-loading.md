# Minimal runtime capability loading

The runtime expresses a capability need. Deterministic retrieval narrows skill
and current active-bundle candidates; the existing API LLM decides how to organize
the selected capabilities. This API has no historical REUSE action.

```text
need ──→ SkillDiscovery (existing RetrievalCard / BM25 / optional Dense + RRF)
     └─→ current active bundles (existing BM25 over purpose + member representations)
                       ↓
             LLMCapabilityResolver
                       ↓ strict validation
               DIRECT / EXTEND / CREATE
                       ↓ atomic application
              resulting runtime state
```

## Entry point and compatibility

Use `skill_control_plane.runtime.capability_loading`. The older
`skill_control_plane.capability_loading` and package-root exports remain the
previous V0.7 deterministic experiment, unchanged to preserve existing work and
tests. They are not the entry point for this runtime design. Historical experiment
reports have not been rewritten.

```python
from skill_control_plane.registry import SkillRegistry
from skill_control_plane.retrieval.discovery import SkillDiscovery
from skill_control_plane.retrieval.cards import load_retrieval_cards
from skill_control_plane.runtime.capability_loading import (
    RuntimeCapabilityState, RuntimeCapabilityLoader, load_capability,
)

registry = SkillRegistry.from_tree(
    'local_artifacts/v0.5/hermes-current87',
    cards=load_retrieval_cards('local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl'),
)
loader = RuntimeCapabilityLoader(SkillDiscovery(registry))
state = RuntimeCapabilityState()
result = load_capability('Read a PDF and make slides for this task', state, loader=loader)
state = result.resulting_state
```

The default resolver directly instantiates the existing `BigModelChatClient`,
using its existing `BIGMODEL_*` configuration. Tests inject the existing
`TextCompleter` callable or a stub resolver; there is no new provider abstraction.
SkillDiscovery builds its index once; callers can inject its existing dense
factory. Default skill candidate depth is 10, active bundle depth 3. There is no
additional required-skill pruning, new retrieval algorithm or coverage threshold.

## Runtime state and actions

`RuntimeCapabilityState` contains a list of immutable `ActiveBundle` records and
a set of direct skill IDs. An ActiveBundle has an ID, purpose and skill IDs.
Multiple bundles may coexist and share skills. Input state must refer to real
registry skill IDs and contain unique bundle IDs and member IDs.

- DIRECT adds one or more candidate skills to `direct_skills`, without changing
  bundles. Cardinality does not determine the action.
- EXTEND adds deduplicated candidate skills to one candidate active bundle.
  It must add at least one new member. Other bundles and direct skills remain
  unchanged.
- CREATE allocates a runtime UUID bundle ID and adds a new bundle with the LLM's
  purpose and selected skills. Existing bundles/direct skills remain unchanged.

Application returns a new state; the host explicitly adopts
`result.resulting_state`. The supplied state is not mutated, including on errors.
Results preserve need, retrieval candidates and representation evidence, bundle
candidates, LLM attempts/decision/reason, affected bundle ID and resulting state.
This resolves capability organization; it does not execute SKILL.md or contact
email/GitHub on the user's behalf.

## Resolver boundary and validation

The model receives only need, Top-k skill candidates with their prepared
representations/evidence and Top-n active bundles with purpose/current membership.
It does not receive the entire registry, direct-skill surface, unrelated bundles
or historical library. Input text is labelled untrusted data in the prompt.

The model chooses a one-off direct capability surface, an extension of current
context, or a new cluster worth maintaining. It may reject weak/alternative
retrieval matches. The decision requires `action`, a non-empty array of candidate
skill IDs and a non-empty `reason`; EXTEND additionally requires a candidate
`target_bundle_id`; CREATE additionally requires a non-empty `purpose`. Unknown
fields, unknown actions (including REUSE), invented IDs, out-of-candidate IDs,
empty selections and no-op EXTEND are invalid. Duplicate skill selections are
deduplicated before application. Stub resolver decisions are revalidated too.

Malformed JSON/schema permits one repair call with the validation error. A
second failure raises `ResolverError` with attempts and leaves state unchanged.
Transport/provider errors propagate without being mistaken for schema repair.
The existing chat adapter canonicalizes provider JSON before returning it; audit
records label this as `adapter_response`, not original HTTP response bytes.
If the adapter rejects malformed JSON before returning text, an attempt records
that validation error without inventing missing raw text.

## Validation and current limits

Ordinary pytest is network-free. Tests cover multi-skill DIRECT, CREATE alongside
existing bundles, target-only EXTEND, multiple bundles, deduplication, scope-limited
prompts, strict schema, invented/out-of-candidate IDs, one repair and provider
failure with unchanged state.

Run the separate live acceptance with the existing environment:

```bash
set -a
source .env
set +a
PYTHONPATH=src .venv/bin/python scripts/capability_loading_e2e.py
```

It validates the frozen current87 manifest/cards, attempts existing Dense/RRF
with copied embedding cache, falls back explicitly to real BM25 if Dense fails,
and makes three actual chat requests. Unique local output directories preserve
previous runs. A non-successful case makes the runner exit nonzero. No fixture
injects candidates or decisions. The current recorded run had HTTP 429 for all
three resolver requests and Dense, so live resolver semantics remain unverified.

Lexical bundle narrowing may admit unrelated bundles through common words or
miss paraphrases. The LLM must assess relevance, but a valid schema does not prove
semantic correctness. The distinction between a one-off task and a maintained
cluster can be ambiguous unless the need supplies duration/context. Persistent
storage, historical reuse, templates, lifecycle, learning, embeddings for bundles,
automatic merging, execution, OpenPI integration and triggers are out of scope.
