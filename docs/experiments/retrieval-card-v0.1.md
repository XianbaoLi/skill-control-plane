# RetrievalCard v0.1 — frozen-query retrieval ablation

## Goal

Improve Skill candidate recall without changing the runtime query generator.

This experiment freezes:

- the audited old `capability_need` rewrites;
- the Hermes 87-Skill corpus snapshot;
- BigModel `embedding-3`, 2048 dimensions;
- BM25/Dense Top-10 source depth;
- RRF `k=60`;
- A/B/C/D retrieval definitions.

The only retrieval-side treatment is the Skill representation.

## Gold audit

`ST-01/S2 -> systematic-debugging` was reclassified.

A reproducible Python test failure makes `systematic-debugging` useful methodology,
but does not uniquely require that Skill: normal code inspection/debugging can still
make progress without loading it. Therefore:

- S2: `systematic-debugging` moves from `new_required/required_now` to `useful`.
- S3: it remains `useful`, while `python-debugpy` remains `new_required`.

After this audit there are 3 positive runtime `new_required` occurrences:
`google-workspace`, `python-debugpy`, and `github-code-review`.

## RetrievalCard v0.1

Each card is extracted offline from exactly one Skill. The LLM never sees Gold,
queries, retrieval outputs, or other Skills.

Fields:

- `purpose`
- `use_when`
- `capabilities`
- `lexical_cues`

The stored card also contains `skill_id`, source `content_hash`, and card version.
A stale or incomplete card corpus is rejected before evaluation.

The search text is:

```text
name
description
tags
purpose
use_when
capabilities
lexical_cues
```

Full `SKILL.md` bodies are still excluded from runtime retrieval. They are loaded
only after candidate selection.

## Build cards

RetrievalCard extraction calls ZAI/BigModel directly. Hermes is not part of this
offline indexing path.

Expected environment:

```bash
BIGMODEL_API_KEY=...
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
BIGMODEL_CHAT_MODEL=glm-5.2
```

First run a one-call provider smoke test:

```bash
cd ~/workspace/skill-control-plane
set -a
source .env
set +a

printf '%s' 'Return ONLY this JSON object: {"status":"ok"}' \
  | python scripts/retrieval_card_bigmodel.py \
      --model "$BIGMODEL_CHAT_MODEL" --reasoning-effort none
```

Then build all cards:

```bash
mkdir -p local_artifacts/v0.5/retrieval-card-audits

skill-control-plane corpus retrieval-cards \
  /mnt/d/Hermes/skills \
  --output local_artifacts/v0.5/retrieval-cards-v0.1.jsonl \
  --extract-command "python scripts/retrieval_card_bigmodel.py --model glm-5.3-flash --temperature 0.1 --max-tokens 1200 --reasoning-effort none --audit-dir local_artifacts/v0.5/retrieval-card-audits"
```

The direct adapter uses the OpenAI-compatible BigModel `/chat/completions`
endpoint, validates the returned content as exactly one JSON object, and writes only
canonical JSON to stdout. API keys are never written to audit files.

The extraction cache checkpoints after every successful Skill and reuses cards whose
source `content_hash` still matches.

## Run tests

```bash
pytest -q tests/test_bigmodel_chat.py \
          tests/test_retrieval_cards.py \
          tests/test_stage_gold_audit.py \
          tests/test_retrieval_ablation.py
```

## Re-run corrected metadata baseline

This must be rerun because the Gold denominator changed from 4 to 3 positive
`new_required` occurrences.

```bash
set -a
source .env
set +a

mkdir -p local_artifacts/v0.5

skill-control-plane eval retrieval-ablation \
  /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-backend bigmodel \
  --bigmodel-embedding-model embedding-3 \
  --bigmodel-embedding-dimensions 2048 \
  --old-rewrite-command "python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions" \
  --skill-representation metadata \
  --per-retriever-k 10 \
  --rrf-k 60 \
  --json \
  | tee local_artifacts/v0.5/frozen-old-query-metadata-gold-audited.json
```

## Run RetrievalCard treatment

```bash
skill-control-plane eval retrieval-ablation \
  /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-backend bigmodel \
  --bigmodel-embedding-model embedding-3 \
  --bigmodel-embedding-dimensions 2048 \
  --old-rewrite-command "python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions" \
  --skill-representation retrieval-card \
  --retrieval-cards local_artifacts/v0.5/retrieval-cards-v0.1.jsonl \
  --per-retriever-k 10 \
  --rrf-k 60 \
  --json \
  | tee local_artifacts/v0.5/frozen-old-query-retrieval-card-v0.1.json
```

## Primary decision rule

Do not optimize for 100% on this tiny diagnostic set. On the current 3 positive
targets, RetrievalCard v0.1 is directionally successful if it recovers
`python-debugpy` without losing `google-workspace` or `github-code-review`, while
candidate-set size remains bounded.

Before architectural adoption, expand to at least 30-50 audited transitions and target:

- Union candidate Recall@10 >= 0.90;
- RRF Recall@5 >= 0.80;
- average final candidate budget about 5-10;
- hard-negative hit rate < 0.10.
