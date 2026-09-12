# Evaluation Plan V0.1

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


## Unit of evaluation

Runtime Gold is labeled at the **task-state/stage** level, not only once per conversation.

A long conversation can produce multiple cases:

```text
Conversation
  -> Stage 1: inspect repository
  -> Stage 2: Docker discovered
  -> Stage 3: dependency failure
```

Each stage receives its own Skill labels.

## Runtime labels

- **required** — omission is expected to materially hurt this stage.
- **useful** — relevant but not required.
- **hard_negative** — close in domain or vocabulary but should not be activated.

See `docs/gold-annotation-v0.1.md` for the annotation rules.

## Runtime candidate generation

V0.1 uses two compact metadata retrievers:

```text
BM25 Top-K ─┐
            ├─ union + dedupe -> candidate set -> LLM judge
Dense Top-K ┘
```

The retrieval layer is optimized for high recall. It does not own the final
semantic ranking decision.

The default searchable surface for both retrievers is compact Skill metadata:
name, description, and tags. Full Skill bodies are loaded later when richer
evidence is needed.

RRF remains available as an experimental baseline, not the default candidate
generator.

## Runtime baselines

- **R0:** harness-native / prompt-only Skill routing
- **R1:** one-shot union retrieval at task start
- **R2:** union retrieval plus dynamic rerouting

## Runtime retrieval metrics

Primary retrieval metrics:

- **union candidate recall** — fraction of required Skill labels present in
  `BM25@K ∪ Dense@K`
- **full-case coverage** — fraction of task stages for which every required Skill
  is present in the union candidate set
- **average candidate-set size** — context/cost proxy for the downstream judge

Retriever rank is diagnostic only.

Downstream runtime-judge metrics remain separate:

- active-Skill precision
- hard-negative activation rate
- stage Skill coverage
- reroute recovery rate
- Skill-related prompt/context tokens

## Evolution labels

For each experience:

- gold relation: COVERED / REFINES / EXTENDS / NOVEL
- gold action: NOOP / PATCH_PROPOSAL / CREATE_PROPOSAL
- gold target Skill when a patch is expected

## Evolution baselines

- **E0:** prompt-only learning decision
- **E1:** nearest-neighbor / threshold baseline
- **E2:** high-recall retrieval + evidence + LLM relation judge

## Evolution metrics

- candidate Recall@K
- action Macro-F1
- target-Skill accuracy
- duplicate-create rate
- wrong-patch rate

Wrong patch is more dangerous than duplicate creation because it contaminates an existing Skill.

## Initial acceptance targets

| Metric | Initial target |
|---|---:|
| Runtime union candidate recall | >= 90% |
| Evolution candidate Recall@5 | >= 95% |
| Active-Skill precision | >= 75% |
| Dynamic stage-coverage improvement | >= +15 percentage points vs one-shot |
| Duplicate-create reduction | >= 50% vs prompt-only |
| Wrong-patch rate | <= 10% |
| Explainable decision payload | 100% |

Average candidate-set size is recorded during the first calibration round before
setting a hard target.

## First corpus size

Start small enough to audit:

- 50-100 Skills
- ~30 conversations/tasks
- ~75 stage-level runtime cases
- ~60 evolution experiences

Only scale after the harness and labels are stable.
