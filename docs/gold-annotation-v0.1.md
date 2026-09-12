# Gold Annotation V0.1

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


## Purpose

The runtime retrieval Gold set evaluates whether the retrieval layer preserves the
Skills needed for a task state before the LLM judge makes the semantic decision.

The retrieval layer is intentionally recall-oriented:

```text
BM25 Top-K ─┐
            ├─ union + dedupe -> candidate set -> LLM judge
Dense Top-K ┘
```

Gold labels therefore describe **task-stage capability need**, not retriever rank.

## Unit of annotation

One case represents one task state or stage.

Do not label an entire long conversation with one fixed Skill set if the required
capabilities change during execution.

## Labels

### required

A Skill is `required` when omitting it is expected to materially block or degrade
the current task stage.

A case may have more than one required Skill.

### useful

A Skill is `useful` when it is genuinely relevant and could help, but the task
stage can still be completed correctly without it.

Useful Skills are not retrieval failures if absent and should not be treated as
hard negatives.

### hard_negative

A Skill is a `hard_negative` when it is plausibly confusable with the required
Skill because of shared vocabulary, domain, tool family, or workflow shape, but
should not be activated for this task stage.

Do not use random unrelated Skills as hard negatives.

## Annotation rules

1. **No forced single-Gold assumption.** Multiple required Skills are allowed.
2. **Inspect the Skill definition, not the retriever output.** Retrieval rank must
   not determine the Gold label.
3. **Natural task wording.** Queries should resemble user intent and should not
   copy the target Skill name or description unless the tool/product name is
   naturally part of the request.
4. **Separate relevance from necessity.** Put supporting capabilities in
   `useful`, not `required`.
5. **Hard negatives must be defensible.** Add them only when the wrong activation
   would be a realistic semantic mistake.
6. **Ambiguous cases stay out of scoring.** If capability boundaries are unclear,
   mark the case for review instead of forcing a label.
7. **Freeze corpus identity.** Every Gold set records the Skill snapshot it was
   annotated against.

## Case schema

```json
{
  "case_id": "HRT-001",
  "query": "Natural task-state query",
  "required": ["skill-a"],
  "useful": ["skill-b"],
  "hard_negative": ["skill-c"],
  "rationale": "Why these labels are correct.",
  "snapshot_id": "..."
}
```

## Retrieval metrics

For the default V0.1 candidate generator:

```text
candidate_set = BM25 Top-K union Dense Top-K
```

Primary metrics:

- required Skill recall over the union candidate set
- full required-set coverage per case
- candidate-set size

Retriever rank is diagnostic only. It is not the final semantic decision metric.

Runtime judge evaluation is separate and measures activation precision,
hard-negative activation, and required Skill coverage after judging.
