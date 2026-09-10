# Experimental Skill Agent v0.1

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
  → new RuntimeCapabilityState + application result appended to history
  → same client, retained history + regenerated system context with selected bodies
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
metadata-only `render_context()` remain compatible.

## Action protocol and history

Each completion is exactly one JSON object with one of three `type` values:

- `load_capability`: only `type` and a nonblank string `need`.
- `apply_capability`: `type` plus the existing strict DIRECT/EXTEND/CREATE schema
  (`action`, `skill_ids`, `reason`, and action-specific target or purpose).
- `final`: only `type` and nonblank string `content`.

A load action calls `harness.search_capability`, performing one Skill search and
storing the returned candidates without changing runtime state. It returns the
complete `SkillDiscoveryResult` as a JSON `capability_tool_result` envelope in a
user-role message. Apply results use the same envelope. This is a provider-neutral
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

The system context contains the short self-trigger rule, action schemas, runtime
metadata and `Loaded Skill Instructions`. Metadata lists direct skills and every
maintained bundle. The instruction section reads `SkillRecord.body` for exactly:

```text
direct_skills ∪ all maintained bundle member skill_ids
```

Each selected body is included once in stable ID order. Initial empty state injects
no bodies. Retrieval result messages include only candidate representations and
evidence; they contain no Skill bodies. The complete registry and Retrieval Card
library are never rendered. Selected instructions provide task guidance while the
system protocol remains authoritative. Supporting files referenced by SKILL.md
are not automatically read or executed.

Apply reuses `validate_decision` and `apply_decision`: candidates must come from
the latest search, the target must exist in current state, and EXTEND must add at
least one member. Multi-skill DIRECT is allowed. Successful application consumes
the pending search; replay without a fresh load fails. A later load replaces the
pending candidates, and a failed new search invalidates stale candidates. Invalid
application preserves both state and the pending candidates for explicit caller
handling. There is no bundle retrieval or bundle Top-K.

## Trace and live acceptance

Each step records the model response/action, need, retrieved and selected IDs,
organization action, state before/after, injected IDs at the model call and after
the action, and an error type if applicable. State snapshots are independent of
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
application are followed by another model call with the selected instructions.
Traces are written to unique local directories even on failure. Ordinary pytest
is network-free.

Not included: command/file execution, provider-native tools, OpenPI/Pi, hard
triggers, automatic candidate sufficiency/retrieval policies, bundle lifecycle,
persistence, concurrent mutation handling or context truncation. The action loop
already permits another explicit model load with a new need, but no sufficiency
loop or retry strategy is implemented. Self-trigger quality and task correctness
require further experiments beyond schema and transport checks.
