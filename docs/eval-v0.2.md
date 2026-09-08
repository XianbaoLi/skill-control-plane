# Evaluation Plan V0.2 — Stage Reroute L1

V0.2 currently focuses on the first clean stage-transition experiment:

```text
raw runtime evidence
        |
        v
BM25 + Dense retrieval
        |
        v
candidate union
        |
        v
compare with Gold required_now / new_required
```

There is no LLM state interpreter and no runtime activation judge in this level.

Multi-Skill evaluation is parked for a later round so the stage-reroute
methodology can be validated independently.

## Gold

`evals/gold/stage-transition-v0.2.jsonl`

Each case contains:

- `initial_task`: the original user task;
- `stages`: the task trajectory.

Each stage contains:

- `runtime_evidence`: raw facts/messages/tool outputs available at runtime;
- `required_now`: Gold Skills required at this stage;
- `new_required`: Skills newly required at this stage;
- `useful` and `hard_negative`: annotation-only labels.

Gold interpretations must never appear in `runtime_evidence`.

Examples of valid runtime evidence:

- a real traceback or failed test line;
- an email message body;
- a user follow-up request;
- a tool error;
- a concrete CI status.

Examples of invalid runtime evidence:

- "this requires root-cause debugging";
- "the task has transitioned to calendar mutation";
- any annotation written because the labeler already knows the target Skill.

## Retrieval query

### S1

S1 has no added runtime evidence:

```text
query = initial_task
```

Therefore the S1 reroute candidate set must be identical to the one-shot
candidate set. The evaluator asserts this invariant.

### S2+

Later stages use:

```text
query =
initial_task
+
current stage raw runtime evidence
```

This still gives the retriever the persistent task goal, but adds no artificial
LLM or human semantic interpretation.

## Baselines

### One-shot

Retrieve once from `initial_task` and hold that candidate set fixed.

### Raw-evidence reroute

Re-run retrieval at every stage using the query above.

## Metrics

### one_shot_stage_full_coverage

Fraction of stages where the initial candidate set contains every
`required_now` Skill.

### reroute_stage_full_coverage

Fraction of stages where the current raw-evidence reroute candidate set contains
every `required_now` Skill.

### reroute_gain

```text
reroute_stage_full_coverage - one_shot_stage_full_coverage
```

Because S1 uses the exact same query under both methods, this gain can no longer
come from giving S1 extra human-written state text.

### transition_new_skill_recall

Among `new_required` Skills introduced after S1, how many appear in the current
reroute candidate set.

This asks whether the new stage information points retrieval toward the newly
needed capability.

### incremental_recovery

A stricter metric.

It considers only transition Skills that were **absent from the original
one-shot candidate set**, and asks how many are recovered after rerouting.

A Skill that was already present at task start is not counted as "recovered."

### average_reroute_candidate_set_size

Context/cost proxy for the downstream judge.

## Not measured yet

Candidate presence is not Skill activation.

L1 does not measure:

- active-Skill precision;
- hard-negative activation;
- premature activation;
- LLM state interpretation quality.

Those belong to later levels:

```text
L1: raw evidence -> Retriever
L2: raw evidence -> LLM state interpreter -> Retriever
L3: raw evidence -> interpreter -> Retriever -> runtime judge
```

## Current calibration cases

- `ST-01`: issue-to-PR -> reproducible Python failure -> step-through debugging -> PR review;
- `ST-03`: terminal inbox triage -> email requests meeting -> user approves calendar action.

These are calibration cases, not a statistically strong benchmark.


## Failure diagnosis: A/B/C query ablation

When raw-evidence rerouting misses a required Skill, run the diagnostic evaluator
before changing K, metadata, or retrieval algorithms.

The same target Skill is ranked under three query constructions:

```text
A = initial_task
B = initial_task + raw runtime evidence
C = raw runtime evidence only
```

For every failed target, record:

- BM25 full-corpus rank;
- Dense full-corpus rank;
- whether the target is present in the BM25@K ∪ Dense@K candidate union.

Interpretation:

- **ANCHORING_SIGNAL** — B misses the target but C brings it into the candidate
  union. Keeping the initial goal is likely suppressing the current-stage signal.
- **TOPK_BUDGET_SIGNAL** — B and C both miss at K, but the best target rank is
  between K+1 and 2K. The representation is finding the Skill, but the candidate
  budget may be too tight.
- **REPRESENTATION_OR_RETRIEVER_SIGNAL** — the target remains farther away.
  Investigate compact Skill metadata and retriever semantics before increasing K.

These labels are diagnostic heuristics, not final causal claims. Always inspect
the actual BM25 and Dense ranks.

The default diagnostic output includes only required Skills missed by B, because
those are the failures of the current L1 reroute path.
