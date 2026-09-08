# Skill Control Plane

An evaluation-first control layer for large Agent Skill libraries.

## V0.1 hypothesis

> Deterministic retrieval narrows a large Skill library into a high-recall candidate set with inspectable evidence; an LLM judge makes the semantic decision; runtime routing can re-run when the task state changes instead of locking the initial Skill set.

V0.1 validates two loops. V0.2 keeps those baselines and adds a stage-batched runtime working set:

1. **Runtime routing** — retrieve candidate Skills from the current task state and support re-routing on subgoal transitions, execution failures, and capability gaps.
2. **Evolution preflight** — retrieve existing Skills before learning from a new experience, then decide whether the experience is already covered, refines/extends an existing Skill, or is novel.\n3. **Stage capability bundles (V0.2)** — one retrieval pass forms an Active Bundle plus a compact Capability Shelf; later evidence expands an existing bundle or creates a new one without automatically discarding prior capabilities.

The first release is deliberately proposal-first. It does **not** automatically merge, archive, delete, supersede, or mutate a user's live Skill library.

## V0.1 decision surface

Evolution relations:

- `COVERED` → `NOOP`
- `REFINES` → `PATCH_PROPOSAL`
- `EXTENDS` → `PATCH_PROPOSAL`
- `NOVEL` → `CREATE_PROPOSAL`

Runtime re-routing triggers:

- subgoal transition
- execution failure
- explicit capability gap

## Architecture

```text
                   Skill Registry
                         |
                    Skill Index
                         |
                 Retrieval / Evidence
                  /                \
                 /                  \
        Runtime Judge          Evolution Judge
             |                      |
       Active Skill Set      NOOP / PATCH / CREATE
             |
         Agent runtime
             |
     subgoal / failure / gap
             |
          re-route
```

Runtime and evolution intentionally share the same registry and retrieval layer.

## Development

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

See `docs/architecture-v0.1.md`, `docs/architecture-v0.2.md`, and `docs/eval-v0.2.md`.

## Non-goals for V0.1

No automatic lifecycle manager, Skill graph, hierarchy builder, multi-Skill DAG composer, marketplace, learned retriever, or automatic destructive mutation.


## Stage Bundle experiment (V0.2)

After creating or restoring the exact Skill snapshot referenced by
`evals/gold/stage-transition-v0.2.jsonl`, run:

```bash
export HF_HUB_OFFLINE=1

skill-control-plane eval stage-bundle \
  /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 \
  --max-bundles 4 \
  --max-skills-per-bundle 4
```

The first experiment intentionally uses:

- S1 query = `initial_task`;
- S2+ query = current raw runtime evidence only;
- BM25 Top-K union Dense Top-K as the candidate pool;
- deterministic category/tag grouping as the Bundle baseline;
- no SRC/extra LLM call yet.

Primary metrics:

- `initial_shelf_future_skill_recall`: whether the initial Shelf already contains future required Skills;
- `initial_shelf_future_bundle_recall`: whether it at least predicts the future capability group;
- `shelf_reuse_rate`: later transitions that can reuse a previously discovered Bundle;
- `new_bundle_rate`: later transitions that require a newly discovered capability group;
- Active-vs-Shelf required-Skill recall.

This experiment tests the value of retaining a compact Capability Shelf before
adding retrieval-sufficiency thresholds or SRC semantic expansion.
