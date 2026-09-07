# Skill Control Plane

An evaluation-first control layer for large Agent Skill libraries.

## V0.1 hypothesis

> Deterministic retrieval narrows a large Skill library into a high-recall candidate set with inspectable evidence; an LLM judge makes the semantic decision; runtime routing can re-run when the task state changes instead of locking the initial Skill set.

V0.1 validates two loops:

1. **Runtime routing** — retrieve candidate Skills from the current task state and support re-routing on subgoal transitions, execution failures, and capability gaps.
2. **Evolution preflight** — retrieve existing Skills before learning from a new experience, then decide whether the experience is already covered, refines/extends an existing Skill, or is novel.

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

See `docs/architecture-v0.1.md` and `docs/eval-v0.1.md`.

## Non-goals for V0.1

No automatic lifecycle manager, Skill graph, hierarchy builder, multi-Skill DAG composer, marketplace, learned retriever, or automatic destructive mutation.
