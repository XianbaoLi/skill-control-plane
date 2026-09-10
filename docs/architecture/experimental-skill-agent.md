# Experimental Skill Agent v0.2

This bounded experiment tests dynamic Skill management using one conversation
client and one growing history. It is not an execution agent or a general agent
framework. No resolver model runs inside capability loading.

```text
User task + retained history
  → same client.complete_messages([current system context, *history])
  → assistant JSON load_capability(need)
  → Harness.search_capability → existing SkillDiscovery
  → candidates + representations/evidence appended to history as tool result
  → same client.complete_messages([current system context, *history])
  → assistant JSON apply_capability(DIRECT / EXTEND / CREATE)
  → Harness.apply_capability → validate_decision → apply_decision
  → new RuntimeCapabilityState + apply result with selected bodies appended to history
  → same client, retained history + regenerated system containing Bundle metadata only
  → continue or final
```

## Minimal API

```python
from skill_control_plane.corpus.bigmodel_chat import BigModelChatClient
from skill_control_plane.runtime.capability_harness import RuntimeCapabilityHarness
from skill_control_plane.runtime.experimental_agent import ExperimentalSkillAgent

# discovery is an existing SkillDiscovery, with optional dense_factory.
harness = RuntimeCapabilityHarness(discovery=discovery)
agent = ExperimentalSkillAgent(harness, BigModelChatClient(), max_steps=8)
answer = agent.run('A task requiring the installed capabilities')
trace = agent.trace
# A subsequent run retains previous conversation and runtime state.
follow_up = agent.run('A follow-up task')
```

`BigModelChatClient.complete_messages(messages)` uses the existing HTTP transport
and configured model, sends the supplied messages unchanged, and returns raw
assistant text. It inserts no extraction instructions and does not canonicalize
JSON (so duplicate action keys remain detectable). The original `__call__(prompt)`
still adds its extraction system/user pair and returns canonical JSON, preserving
existing callers and tests.

The Agent never calls `RuntimeCapabilityLoader.load_capability` or instantiates
`LLMCapabilityResolver`. The harness accepts either a legacy loader or a discovery
instance; the latter needs no resolver credentials. Legacy combined loading and
metadata-only `render_context()` remain compatible. The experimental Agent uses
`render_bundle_context()` instead; it does not use the legacy direct-skill surface.

## Action protocol and history

Each completion is exactly one JSON object with one of three `type` values:

- `load_capability`: only `type` and a nonblank string `need`.
- `apply_capability`: `type` plus the existing strict DIRECT/EXTEND/CREATE schema
  (`action`, `skill_ids`, `reason`, and action-specific target or purpose).
- `final`: only `type` and nonblank string `content`.

A load action calls `harness.search_capability`, performing one Skill search and
storing the returned candidates without changing runtime state. It returns the
complete `SkillDiscoveryResult` as a JSON `capability_tool_result` envelope in a
user-role message. Apply results use the same envelope and include `action`,
`affected_bundle_id`, `selected_skill_ids` and `skill_bodies` (ID/full body pairs). This is a provider-neutral
JSON protocol, not native tool calling; the system prompt explicitly identifies
these user-role envelopes as tool data. Every assistant action, original user
message and result remains in `agent.history` for subsequent model calls.

Each model call has exactly one regenerated system message followed by a snapshot
of the complete history. State updates never clear history. A new `run()` adds a
user task, resets the per-run trace and pending search, and keeps prior history and
state. There is a maximum number of model calls per run. Invalid actions, retrieval
or provider errors fail explicitly and remain in the trace; no hidden repair model
or automatic sufficiency search is invoked. Previously successful applications
remain committed if a later step fails.

## Loaded context and validation

The system context contains only the short self-trigger rule, action interfaces,
and `Runtime Bundles`. `render_bundle_context()` reads each maintained bundle from
state and renders its `bundle_id`, `purpose`, and `members`. Each member has
`skill_id`, `name`, and `short_description` from its `SkillRecord`. Descriptions
have whitespace collapsed and are capped at 240 characters; bundles and members
are sorted by ID. No Retrieval Card fields or global catalog are included.

Full Skill bodies are never placed in the system context, including on the first
call with a nonempty initial state. DIRECT state semantics remain unchanged, but
DIRECT skills are not a permanent system display surface. Initial state alone
does not deliver instructions: bodies enter this conversation only after an
explicit successful application.

`harness.apply_capability()` prepares a result containing the validated selected
IDs and their exact `SkillRecord.body` values before committing the new state.
The Agent appends those bodies as the apply tool result. It tracks delivered IDs
for this conversation, so selecting the same skill again (including DIRECT then
CREATE/EXTEND, or a later `run()`) retains selected IDs in the result but omits
already-delivered bodies. Deduplication belongs to conversation history, not to
runtime lifecycle. Other consumers of the harness receive all selected bodies.

The original apply result remains in history and is included in later requests;
no new body copy is appended per turn. This still transmits that history text on
every full-history request; it is not provider caching or context truncation.
A separate Agent with a new history has a separate delivered-ID set. External
history editing/restoration is not supported in this experiment.

Retrieval results contain only candidate representations and evidence, not Skill
bodies. Unselected bodies and the full library are never supplied to the Agent.
Skill instructions provide task guidance while the action protocol remains
authoritative. Supporting files referenced by SKILL.md are not read or executed.

Apply reuses `validate_decision` and `apply_decision`: candidates must come from
the latest search, the target must exist in current state, and EXTEND must add at
least one member. Multi-skill DIRECT is allowed. Successful application consumes
the pending search; replay without a fresh load fails. A later load replaces the
pending candidates, and a failed new search invalidates stale candidates. Invalid
application preserves both state and the pending candidates for explicit caller
handling. There is no bundle retrieval or bundle Top-K.

## Trace and live acceptance

Each step records the model response/action, need, retrieved and selected IDs,
organization action, state before/after, `system_skill_body_ids` (always empty),
`history_skill_body_ids` at the call, `appended_skill_body_ids` for that step,
`history_skill_body_ids_after`, and an error type if applicable. State snapshots are independent of
later mutation. Provider error bodies are excluded from traces. The live script
also records each full model request history, response, model configuration
(excluding credentials), registry snapshot ID and final answer.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_experimental_agent.py tests/test_runtime_capability_loading.py tests/test_bigmodel_chat.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check

set -a
source .env
set +a
PYTHONPATH=src .venv/bin/python scripts/experimental_skill_agent_e2e.py
```

The live script uses the current87 manifest/cards, real configured Dense embeddings
and BM25/RRF. There is no fallback to lexical-only retrieval and no injected model
action/candidate fixture. It checks that a model-triggered search and successful
application are followed by another model call with the selected instructions in
history. An independent audit of actual model messages checks no registry bodies
appear in system, unselected bodies stay hidden, each delivered body matches its
record and occurs once in tool results, Bundle members use only the documented
metadata, and the earlier history/assistant responses are retained. Per-call
`context_audit` entries expose system bundles and history body IDs separately.
Traces are written to unique local directories even on failure. Ordinary pytest
is network-free.

Not included: command/file execution, provider-native tools, OpenPI/Pi, hard
triggers, automatic candidate sufficiency/retrieval policies, bundle lifecycle,
persistence, concurrent mutation handling or context truncation. The action loop
already permits another explicit model load with a new need, but no sufficiency
loop or retry strategy is implemented. Self-trigger quality and task correctness
require further experiments beyond schema and transport checks.
