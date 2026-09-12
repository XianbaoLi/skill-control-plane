# Benchmark corpus v0.1

> **Frozen, pending review.** This record covers corpus freezing only. No benchmark,
> Retrieval Cards, DenseIndex, embeddings, or retrieval changes were run or made.

## Source

- Repository: `https://github.com/Emmraan/agent-skills`
- Frozen commit: `1124ce406f47ba7c5c54e18c5f86f70ebd51c33d`
- Source license: MIT (`LICENSE`)
- Source definition: one package is a top-level `skills/<category>/<skill>/` directory.
- Raw audit: 172 `SKILL.md` files across the package trees.
- Package audit: 152 top-level packages, 146 eligible, 6 excluded.

The nested `pixijs-2d` and `webflow` router packages were excluded because they
contain descendant `SKILL.md` files. Four packages were excluded because their
YAML frontmatter was unparseable. The exact exclusions and reasons are recorded
in the tracked manifest. No third-party script was executed.

## Selection

Selection uses `deterministic-stratified-largest-remainder` v1:

1. Order eligible packages within each category by
   `sha256(source_commit + NUL + source_relative_path)`, with the source path as
   the final lexical tie-breaker.
2. Allocate S32/S64/S128 category quotas by largest remainder.
3. Take the first allocated rank from each category.

This covers all 11 source categories and produces nested subsets. For the same
source commit, rerunning the freeze produces the same IDs and hashes.

| Category | Eligible | S32 | S64 | S128 |
|---|---:|---:|---:|---:|
| agent-meta | 25 | 5 | 11 | 22 |
| ai-ml | 4 | 1 | 2 | 4 |
| animation-webgl | 21 | 5 | 9 | 18 |
| backend-apis | 18 | 4 | 8 | 16 |
| databases-data | 8 | 2 | 3 | 7 |
| design-ux | 9 | 2 | 4 | 8 |
| devops-cloud | 12 | 3 | 5 | 11 |
| frontend-ui | 16 | 3 | 7 | 14 |
| languages | 13 | 3 | 6 | 11 |
| marketing-growth | 13 | 3 | 6 | 11 |
| testing-quality | 7 | 1 | 3 | 6 |
| Total | 146 | 32 | 64 | 128 |

## Package integrity

The complete package hash is SHA-256 over all package files in sorted relative
path order, updating:

```text
relative_path_bytes + 0x00 + file_bytes
```

Thus `references/`, `scripts/`, and `assets/` affect the hash. The S128 corpus
contains 1,185 files and 8,011,767 content bytes: 93 packages include
references, 4 include scripts, and 3 include assets. The freeze script copies
bytes without following symlinks and verifies each copied package hash.

## Artifacts

- Tracked manifest: `evals/corpora/benchmark-corpus-v0.1.json`
- Freeze script: `scripts/freeze_benchmark_corpus.py`
- Local corpus: `local_artifacts/corpora/benchmark-corpus-v0.1/`
- Local manifests: `local_artifacts/corpora/benchmark-corpus-v0.1/manifests/`

The local corpus materializes independent `S32/skills/`, `S64/skills/`, and
`S128/skills/` trees plus `source-metadata.json`. `local_artifacts/` remains
ignored.

## Validation

The real freeze used `SkillStore.from_tree` on each materialized subset:

- S32: 32 unique Skill IDs, valid
- S64: 64 unique Skill IDs, valid
- S128: 128 unique Skill IDs, valid
- `S32` IDs are a strict subset of `S64`; `S64` IDs are a strict subset of `S128`
- Every selected `SKILL.md` parsed, had a nonempty body, and resolved under its
  materialized subset
