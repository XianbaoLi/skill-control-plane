# Retrieval explainability v0.1 — field ablation + query robustness

## Scope

This experiment extends RetrievalCard v0.1 without changing the runtime query
generator or the retrieval stack.

Frozen conditions:

- corpus: `current87`;
- query source: frozen old `capability_need` replay;
- Dense backend: BigModel `embedding-3`, 2048 dimensions;
- BM25/Dense source depth: Top-10;
- RRF: `k=60`;
- Gold semantics: `new_required` is the target capability introduced by a Stage
  transition.

The experiment asks two questions:

1. Which RetrievalCard fields actually contribute to retrieval?
2. Does the richer Skill representation make retrieval less sensitive to wording
   changes in the same capability need?

The current audited `current87` Gold has only 3 positive target transitions.
That is enough for a smoke test, not for an explanatory conclusion.

The CLI therefore requires at least 20 target transitions by default.
`--allow-small-sample` exists only to verify the harness on the current 3 cases.

For architecture-adoption decisions, keep the existing project target of 30–50
audited transitions.

---

## Experiment A — RetrievalCard field ablation

### Terminology

A **leave-one-field-out ablation** starts from the complete RetrievalCard and removes
exactly one field at a time. If retrieval quality drops after removing a field, that
field was contributing useful retrieval information.

RetrievalCard v0.1 fields:

- `purpose`: one-sentence core job of the Skill;
- `use_when`: situations or signals that should trigger retrieval;
- `capabilities`: concrete actions the Skill enables;
- `lexical_cues`: short user-facing words or phrases likely to express those
  situations.

The legacy Skill metadata `name + description + tags` is held fixed in every card
variant.

### Variants

| Variant | Indexed text |
|---|---|
| `metadata-v0.1` | name + description + tags |
| `full` | metadata + all four RetrievalCard fields |
| `minus-purpose` | full card without `purpose` |
| `minus-use-when` | full card without `use_when` |
| `minus-capabilities` | full card without `capabilities` |
| `minus-lexical-cues` | full card without `lexical_cues` |

The report computes:

`drop_from_full = full_metric - ablated_metric`

A positive value means removing the field hurt retrieval, so the field was useful.

### Main metrics

- Dense Recall@5 / Recall@10
- BM25 Recall@5 / Recall@10
- RRF Recall@5 / Recall@10
- Union candidate recall
- Union full-transition coverage

### Formal run

After the Stage Gold contains at least 20 target transitions:

```bash
cd ~/workspace/skill-control-plane

set -a
source .env
set +a

skill-control-plane eval retrieval-field-ablation \
  local_artifacts/v0.5/hermes-current87 \
  --gold local_artifacts/v0.5/stage-transition-v0.2-current87.jsonl \
  --manifest local_artifacts/v0.5/hermes-current87-manifest.json \
  --dense-backend bigmodel \
  --bigmodel-embedding-model embedding-3 \
  --bigmodel-embedding-dimensions 2048 \
  --old-rewrite-command "python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions" \
  --retrieval-cards local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl \
  --per-retriever-k 10 \
  --rrf-k 60 \
  --json \
  | tee local_artifacts/v0.5/current87-field-ablation.json
```

### Current 3-transition smoke test

The same command can be exercised now by adding:

```bash
--allow-small-sample
```

Do not interpret those 3-case deltas as field importance.

---

## Experiment B — query-paraphrase robustness

### Terminology

A **paraphrase query** expresses the same capability need using different wording.
For each target Stage transition the harness freezes:

- 1 original old `capability_need` query;
- 4 semantically equivalent paraphrases.

That gives 5 fixed query variants per transition.

The paraphrase model receives only the original query. It does **not** receive:

- Gold Skill IDs;
- RetrievalCard contents;
- retrieval rankings;
- other Skills;
- benchmark answers.

This prevents the paraphrase step from leaking the correct Skill into the query.

The generated variants are stored in JSONL and reused on later runs. Metadata and
RetrievalCard therefore see exactly the same query variants.

### Metrics

**Recall@K**

Fraction of all required-Skill occurrences across all query variants that appear
within Top-K.

**OriginalRecall@K**

Recall on only the original frozen `capability_need` query.

**ParaphraseRecall@K**

Recall on only the four paraphrases.

**MRR@K**

Mean Reciprocal Rank within K. A required Skill at rank 1 contributes 1, rank 2
contributes 1/2, rank 3 contributes 1/3, and a miss beyond K contributes 0.
This captures both successful retrieval and how high the Skill ranks.

**AllVariantsHit@K**

For one Stage target, every one of its 5 query variants must retrieve the required
Skill within Top-K. This is the strongest direct robustness metric in this
experiment.

If RetrievalCard improves both Recall@K and AllVariantsHit@K relative to metadata,
the result supports the claim that richer Skill representation reduces sensitivity
to runtime query wording.

### Formal run

```bash
cd ~/workspace/skill-control-plane

set -a
source .env
set +a

skill-control-plane eval retrieval-robustness \
  local_artifacts/v0.5/hermes-current87 \
  --gold local_artifacts/v0.5/stage-transition-v0.2-current87.jsonl \
  --manifest local_artifacts/v0.5/hermes-current87-manifest.json \
  --dense-backend bigmodel \
  --bigmodel-embedding-model embedding-3 \
  --bigmodel-embedding-dimensions 2048 \
  --old-rewrite-command "python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions" \
  --paraphrase-command "python scripts/bigmodel_completion_adapter.py --model glm-5.2" \
  --query-variants local_artifacts/v0.5/current87-query-variants-v0.1.jsonl \
  --retrieval-cards local_artifacts/v0.5/retrieval-cards-v0.1-current87.jsonl \
  --paraphrases-per-query 4 \
  --per-retriever-k 10 \
  --rrf-k 60 \
  --json \
  | tee local_artifacts/v0.5/current87-query-robustness.json
```

The first run generates the paraphrase JSONL. Later runs reuse it. Use
`--force-paraphrases` only when intentionally creating a new frozen query-variant
set.

For a harness-only smoke test on the current 3 targets, add
`--allow-small-sample`.

---

## Interpretation order

Do not start from the best number. Read the results in this order:

1. Confirm the target-transition count and Gold snapshot.
2. Compare `full` RetrievalCard with `metadata-v0.1`.
3. Inspect `drop_from_full` to identify which card fields matter.
4. In robustness, compare original recall and paraphrase recall separately.
5. Use `AllVariantsHit@5` as the clearest robustness indicator.
6. Only after the Gold reaches the required sample size treat these differences as
   explanatory evidence.

The intended claims are deliberately narrower than “more text is always better”:

- some capability-oriented RetrievalCard fields should contribute more than others;
- a useful representation should retrieve the same Skill under multiple equivalent
  runtime phrasings;
- these effects should be measured on the complete audited Stage-transition set, not
  cherry-picked examples.
