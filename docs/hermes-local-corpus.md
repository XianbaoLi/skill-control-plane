# Hermes Local Corpus Snapshot

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


The first real V0.1 corpus comes from the local Hermes Skill library.

Current reference environment:

- Hermes home: `D:\Hermes`
- Hermes source: `D:\Hermes\hermes-agent`
- Hermes Skill root: `D:\Hermes\skills`
- Hermes source commit: `d5fd2e93666d20616dca932a488c64ddace2b917`
- observed Skill count: `87`

The Hermes source working tree is not treated as an integration target yet. V0.1 reads the Skill library only.

## WSL snapshot command

From the Skill Control Plane repository:

```bash
cd ~/workspace/skill-control-plane
git pull
python -m pip install -e ".[dev]"

skill-control-plane corpus snapshot \
  /mnt/d/Hermes/skills \
  --output local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --source hermes-local \
  --harness-commit d5fd2e93666d20616dca932a488c64ddace2b917
```

The output path is ignored by Git.

Expected first-run properties:

- `skill_count` should be `87`
- `snapshot_id` should be stable if the Skill corpus is unchanged
- `duplicate_skill_ids` is reported explicitly
- the manifest contains no SKILL.md body text
- local absolute paths are not persisted

## Why the live library is not the benchmark

`D:\Hermes\skills` may change through user edits, Background Review, hub updates, or curator activity. Therefore evaluation must bind to a snapshot id.

The first manifest is a provenance checkpoint, not yet the public benchmark corpus.

After inspecting provenance and possible local/private Skills, a fixed public V0.1 corpus can be selected separately for reproducible repository-level evaluation.
