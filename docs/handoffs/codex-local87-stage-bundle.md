# Codex Handoff — Exact 87-Skill Stage Bundle Experiment

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


## Objective

Run the Stage Capability Bundle experiment against the user's exact local
87-Skill Hermes snapshot, then implement the smallest evidence-backed next step.

Do **not** redesign the architecture before observing the exact-snapshot result.

## Repository / environment

Repository:

```text
~/workspace/skill-control-plane
```

Target branch:

```text
feature/stage-capability-bundles-v0.2
```

Hermes Skill tree:

```text
/mnt/d/Hermes/skills
```

Gold:

```text
evals/gold/stage-transition-v0.2.jsonl
```

Expected manifest:

```text
local_artifacts/corpora/hermes-local-v0.1/manifest.json
```

Expected Gold snapshot id:

```text
1b7e524ea167fb9057423e999d94ba920e6f11817f7ce06fede3a46722bd4bf6
```

Dense model:

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

The model is expected to already be cached locally. Prefer offline mode.

## Architecture boundary

Current runtime architecture is:

```text
Initial Task
    ↓
Global retrieval
    ↓
Candidate Pool
    ↓
Active Bundle + Capability Shelf
    ↓
Work
    ↓
Evidence Pool
    ↓
selective retrieval representation
RAW → reuse Agent interpretation → optional SRC
    ↓
candidate delta
    ↓
expand existing Bundle or create a new Bundle
```

Important separation:

- retrieval/SRC answers **which Skills may be relevant**;
- Bundle/Shelf answers **how retrieved Skills are organized, exposed and loaded**;
- SRC must not be redesigned in this task.

## Existing exploratory finding

The public 81-Skill exploratory run suggested:

1. exact future-Skill prefetch is low;
2. future capability-Bundle foresight can still be useful;
3. filesystem category is too coarse as a final Bundle boundary;
4. most importantly, a known Shelf Bundle must become operational:
   when new evidence matches an existing Bundle, search/expand inside that Bundle
   before doing another full global discovery pass.

Treat these as hypotheses until the exact 87-Skill run is inspected.

## Step 1 — synchronize and validate

Run:

```bash
cd ~/workspace/skill-control-plane
git status
git fetch origin
git switch feature/stage-capability-bundles-v0.2
git pull --ff-only

python -m pip install -e ".[dev,dense]"
pytest -q
```

Do not discard local changes if the working tree is dirty. Report them first and
work around them safely.

## Step 2 — verify exact snapshot

Inspect:

```bash
cat local_artifacts/corpora/hermes-local-v0.1/manifest.json
```

Confirm:

- skill_count = 87;
- snapshot_id equals the Gold snapshot id above.

If snapshot ids do not match, stop the benchmark interpretation. Do not bypass
the check with an exploratory flag.

## Step 3 — run exact Stage Bundle experiment

Run:

```bash
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

skill-control-plane eval stage-bundle \
  /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 \
  --max-bundles 4 \
  --max-skills-per-bundle 4
```

Also capture JSON output:

```bash
mkdir -p local_artifacts/experiments

skill-control-plane eval stage-bundle \
  /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 \
  --max-bundles 4 \
  --max-skills-per-bundle 4 \
  --json \
  > local_artifacts/experiments/stage-bundle-local87-v0.2.json
```

## Step 4 — analyze before coding

Report at minimum:

- initial_shelf_future_skill_recall;
- initial_shelf_future_bundle_recall;
- shelf_reuse_rate;
- new_bundle_rate;
- mean_active_required_recall;
- mean_shelf_required_recall;
- per-stage Active Bundle ids;
- per-stage Shelf Bundle ids;
- per-stage candidate ids;
- every required Skill missed by the Shelf.

For each miss, classify it as one of:

```text
A. retrieval miss
   target Skill never entered candidate pool

B. bundle-budget loss
   target entered candidates but its Bundle was dropped

C. known-bundle expansion miss
   relevant Bundle was already on Shelf, but exact Skill was not recovered

D. activation miss
   exact Skill / Bundle was available but not activated correctly

E. grouping problem
   deterministic category/tag boundary is semantically wrong
```

Do not infer a fix until the miss is assigned one of these categories.

## Step 5 — implementation gate

Only implement **Shelf-aware hierarchical retrieval** if the exact run still
shows meaningful type-C misses.

Target behavior:

```text
New Evidence
    ↓
match against existing Shelf Bundle descriptors/members
    ↓
match?
 ├─ yes → retrieve within that Bundle's Skill subset
 │         → expand / activate matching Skills
 └─ no  → global Skill retrieval
          → discover/append new Bundle
```

### Constraints

- Existing Shelf/registered Skills remain monotonic.
- Do not delete or replace existing Bundles.
- Bundle-local retrieval must reuse the existing Retriever interface where
  practical.
- Do not introduce a new LLM call for Bundle matching in the first version.
- Prefer a deterministic semantic/retrieval baseline first.
- Do not change SRC behavior in this task.
- Do not build a learned ontology or clustering system yet.
- Preserve old evaluation baselines.

## Step 6 — required evaluation for the new implementation

Add an A/B evaluator:

### A — global-only delta retrieval

Current behavior.

### B — Shelf-aware hierarchical retrieval

Known Bundle first, global fallback only when needed.

Measure:

- required-Skill recall;
- Shelf required recall;
- global retrieval rate;
- bundle-local retrieval rate;
- mean search-corpus size;
- wrong-Bundle match rate;
- registered/active surface growth.

Add unit tests covering:

1. evidence matches an existing Shelf Bundle;
2. local search recovers a previously absent Skill inside that Bundle;
3. no Bundle match falls back to global retrieval;
4. global fallback creates a new Bundle;
5. registered Skill set remains monotonic.

Run full:

```bash
pytest -q
```

## Deliverables

1. Exact 87-Skill experiment result and interpretation.
2. A short Markdown experiment note under `docs/experiments/`.
3. If type-C misses justify it, the minimal hierarchical retrieval implementation.
4. A/B metrics and tests.
5. A concise summary of:
   - what failed before;
   - what changed;
   - whether the change improved the exact Gold cases;
   - remaining failure types.

Do not merge the feature branch into main.
