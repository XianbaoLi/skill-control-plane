# Historical Hermes 87-skill snapshot rebuild audit

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


Completed locally without modifying the live tree, Hermes Git history, or the official incomplete frozen directory. No representation was regenerated or relabelled.

## Root cause

The requested `registry.py` has been split in this checkout: the implementation is `src/skill_control_plane/registry/loader.py`. The original loader computes:

```python
raw = skill_md.read_text(encoding="utf-8-sig", errors="replace")
content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

`read_text` uses universal newlines: CRLF and bare CR become LF. UTF-8 BOM is removed; invalid UTF-8 is replaced. The hash covers the complete decoded text, including frontmatter and whitespace, before body `.strip()`. It is not a raw-byte hash, nor a body-only hash. Shared byte-decoding/hash helpers now preserve this exact behavior for both the registry and Git blobs.

Actual live scan: 88 skill IDs, all historical 87 present, only `github` extra. Registry hashes match 78; raw hashes match 72. The six below contain valid UTF-8, no BOM, no bare CR. Replacing only CRLF with LF makes each hash equal its manifest hash. Thus the six are recoverable manifest-equivalent historical text, not six additional content changes.

| Skill | CRLF count | Raw / normalized bytes | Old frozen match | Live match | Git history match |
|---|---:|---:|---|---|---|
| codex-quota | 105 | 4792 / 4687 | yes | yes | not found |
| hermes-memory-system | 138 | 7610 / 7472 | yes | yes | not found |
| windows-disk-space-audit | 58 | 4735 / 4677 | absent | yes | not found |
| quiz-bank-ocr-audit | 29 | 3876 / 3847 | absent | yes | not found |
| hermes-desktop-plugin-authoring | 113 | 6110 / 5997 | absent | yes | not found |
| windows-system-maintenance | 42 | 3982 / 3940 | absent | yes | not found |

Git search inspected **1220 unique SKILL.md blobs** across all local refs, including both sides of changes, deletions and merge diffs. Repository is not shallow. No corresponding pair was found in that history for these six. This does not claim absence from remote-only refs or dangling objects. Their matching text is already present in live; the first two also exist byte-identically in the old frozen tree. Full raw/normalized hashes and paths are recorded per skill in `rebuild-audit.json`.

## Rebuild and acceptance

Identity is exactly `(skill_id, content_hash)`. Matching live bytes are preferred even if paths moved; otherwise a hash- and ID-verified Git blob is restored. Manifest paths are reconstructed to retain historical categories and snapshot ID. The output uses an exclusive `mkdtemp` directory. Missing content fails before any output tree is created. No checkout, source write, or official-frozen replacement is performed.

| Audit item | Result |
|---|---|
| Old frozen skill count / missing / mismatches | 31 / 56 / 0 |
| Recovery source | live 78; Git 9 |
| Final skill count | 87 |
| Missing skill IDs | 0 |
| Extra skill IDs | 0 (`github` excluded) |
| Duplicate skill IDs | 0 |
| Content hash mismatches | 0 |
| Historical relative path mismatches | 0 |
| Existing CLI `_validate_root_snapshot` | PASS |
| Snapshot ID | `1b7e524ea167fb9057423e999d94ba920e6f11817f7ce06fede3a46722bd4bf6` |
| Protected source checks | live SKILL.md inventory, old frozen inventory, manifest, Git HEAD and refs unchanged |

The Git-recovered skills are:

| Skill | Blob ID | Historical Git path |
|---|---|---|
| codebase-inspection | `d42b9a2292a243b14f30d25ee1e1021fd736eef4` | `skills/github/codebase-inspection/SKILL.md` |
| document-to-action-items | `1c32c2459fc6b1570272b0cf21e5cfaaef2c50eb` | `optional-skills/productivity/document-to-action-items/SKILL.md` |
| pdf | `a10b25fdca78c23f7f153ab5de7347e1a670ea83` | `skills/productivity/pdf/SKILL.md` |
| teams-meeting-pipeline | `dc2f88611e245956df1f03d689a56a1089f6420f` | `skills/productivity/teams-meeting-pipeline/SKILL.md` |
| arxiv | `e3e6ac738f7b56344241a7ebff7dfc500d77b4c5` | `skills/research/arxiv/SKILL.md` |
| blocked-page-recovery | `64d19dc967efd5f9b3080fa0028438dd9d56306d` | `skills/research/blocked-page-recovery/SKILL.md` |
| competitor-news-monitor | `c480f2b18b33dea9ecfadd7734814fd3c0dd0c08` | `skills/research/competitor-news-monitor/SKILL.md` |
| grounded-citations | `4c767369f4d97960925c7c2e73ffa192f95fe583` | `skills/research/grounded-citations/SKILL.md` |
| requesting-code-review | `0a543cee2b074b270f030cbd4c1a708ef2721d6d` | `skills/software-development/requesting-code-review/SKILL.md` |

The moved `codebase-inspection` and `blocked-page-recovery` are written at manifest paths, not their current live paths. A `github/` category directory can legitimately exist; the excluded item is the skill whose ID is exactly `github`.

## Representation provenance

| Artifact (under local_artifacts/v0.5) | Rows | IDs / source hashes | Missing / extra | Historical hash matches / mismatches | Historical 1:1 |
|---|---:|---|---|---|---|
| `retrieval-cards-v0.1-current87.jsonl` | 87 | both recorded | 0 / 0 | 78 / 9 | no |
| `retrieval-cards-v0.1-frozen87.jsonl` | 87 | both recorded | 0 / 0 | 78 / 9 | no |
| `retrieval-cards-v0.1.jsonl` | 88 | both recorded | 0 / 1 | 78 / 9 | no |

`retrieval-cards-v0.1-frozen87.jsonl` and `retrieval-cards-v0.1-current87.jsonl` are byte-identical. Both correspond to current87, not the historical manifest. The 9 mismatches are exactly the 9 Git-recovered skills above. The existing `validate_retrieval_cards()` rejects the historical snapshot with these cards; that guard remains intact.

Enhanced representation fields are `purpose`, `use_when`, `capabilities`, and `lexical_cues`, with `skill_id`, `source_content_hash`, and `version` on every card. There is provenance, but it points to a different source version for nine skills.

The retrieval-card audit cache contains 88 successful prompt/completion records covering 88 IDs. It stores `prompt_sha256` and a structured skill payload inside the prompt, but no explicit `source_content_hash`. 80 historical reconstructed prompts match cached prompts exactly; 7 do not. Prompt equality does not certify original file-hash identity, because prompts omit some source material/formatting. No cache completion was promoted into a new historical card set.

- `local_artifacts/v0.4/hermes-facets/`: 4 runtime-stage capability extraction records, no per-skill ID/source hash; not an 87-skill representation corpus.
- `local_artifacts/v0.6/capability-need-v03-summary.jsonl` and `capability-need-v03-refreeze.jsonl`: 13 target-stage queries each; contain Gold IDs in `new_required`, but no per-skill source hash; not 87-skill abstractions.
- `local_artifacts/v0.6/robustness-current87-13target-65query-indexed-texts.json`: 6 × 87 indexed representations, with skill IDs but no per-skill source hashes in the file. The companion experiment report binds it via file hashes to current87 manifest/cards, not this historical snapshot.

## OpenPI Bundle E2E readiness

**Do not start the historical87 + enhanced-representation E2E run yet.** The historical SKILL.md corpus is now fully verified and can be used for corpus/retrieval audits. The available full 87-card representation sets still have 9 stale source hashes, so the requested historical treatment is not ready. Obtain/explicitly prepare a hash-aligned representation set and validate it before that experiment. This task does not regenerate representations.

This manifest hashes only SKILL.md. Rebuilding it cannot establish historical provenance for sibling scripts/assets or certify runtime dependencies. The temporary tree intentionally contains SKILL.md files and audit reports only; it must not be described as a fully restored executable Hermes installation. Runtime skill support files and the OpenPI harness must be validated before activation/E2E.

## Commands, files and tests

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/rebuild_historical_snapshot.py \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --live /mnt/d/Hermes/skills \
  --git-repo /mnt/d/Hermes/hermes-agent \
  --old-frozen local_artifacts/v0.5/hermes-frozen-87 \
  --output-parent local_artifacts/v0.5
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The completed run used the same default arguments. Each rerun creates a new temporary directory.

**Tests: 125 passed in 0.55s.** Added tests cover LF/CRLF/bare-CR/BOM/invalid UTF-8 hash parity with the original loader, historical Git recovery, moved paths, unchanged source bytes, exclusion of extra IDs, repeated runs without overwriting, missing-version failure, manifest path/count/duplicate rejection, source-tree output protection, and stale representation detection.

Changed files:

- `src/skill_control_plane/registry/loader.py`: shared canonical text decoding and hash helpers; unchanged hash semantics.
- `src/skill_control_plane/corpus/rebuild.py`: validated live/Git recovery, protected sources, per-skill provenance and final audit.
- `scripts/rebuild_historical_snapshot.py`: reproducible CLI with temporary-only output.
- `tests/test_historical_rebuild.py`: regression and recovery tests.
- `docs/experiments/historical-hermes87-rebuild-audit.md`: this report.

Local output artifacts:

- `/home/lixianbao/workspace/skill-control-plane/local_artifacts/v0.5/hermes-frozen-87-rebuild-4kpj849b/`
- `/home/lixianbao/workspace/skill-control-plane/local_artifacts/v0.5/hermes-frozen-87-rebuild-4kpj849b/rebuild-audit.json`
- `/home/lixianbao/workspace/skill-control-plane/local_artifacts/v0.5/hermes-frozen-87-rebuild-4kpj849b/representation-audit.json`
- `local_artifacts/v0.5/historical-rebuild-run.log`
- `local_artifacts/v0.5/historical-rebuild-tests.log`

Formal `local_artifacts/v0.5/hermes-frozen-87` remains untouched at 31 skills. Use the validated temporary path explicitly; no promotion was performed.
