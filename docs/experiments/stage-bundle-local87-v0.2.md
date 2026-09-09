# Exact Hermes 87-Skill Stage Bundle V0.2

## Baseline and implementation gate (recorded before algorithm changes)

Branch: `feature/stage-capability-bundles-v0.2`. Baseline tests: **39 passed**.
Real tree: `/mnt/d/Hermes/skills`; manifest: `local_artifacts/corpora/hermes-local-v0.1/manifest.json`.
Rebuilt manifest from the actual tree: **87 Skills**, no snapshot bypass.
All Gold cases, stored manifest and rebuilt snapshot agree on
`1b7e524ea167fb9057423e999d94ba920e6f11817f7ce06fede3a46722bd4bf6`.

Official CLI, offline cached `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`,
K=5 **per retriever**, max_bundles=4, max_skills_per_bundle=4.
Raw evidence only; metadata BM25 + Dense, existing ordered candidate union.
JSON: `local_artifacts/experiments/stage-bundle-local87-v0.2.json` (local, ignored).
Python environment: `/home/lixianbao/miniforge3/envs/skill-control-plane/bin`.

| Metric | Baseline |
|---|---:|
| case_count | 2 |
| stage_count | 6 |
| initial_shelf_future_skill_recall | 0.0 |
| initial_shelf_future_bundle_recall | 0.3333333333333333 |
| shelf_reuse_rate | 0.5 |
| new_bundle_rate | 0.5 |
| mean_active_required_recall | 0.38888888888888884 |
| mean_shelf_required_recall | 0.4444444444444444 |

### Every missed required Skill, classified

Classification is stage-local: C takes precedence over A when a relevant Bundle
existed **before** the evidence pass. Initial retrieval has no pre-existing Shelf.
B covers candidate loss to a Bundle/member budget; D covers Skills on Shelf but
inactive. Historical contributing causes are distinguished from the current failure.

| Case/stage | Required Skill | Class | Evidence |
|---|---|---|---|
| ST-03/S1 | himalaya | A | Absent from both candidate lists; no prior Shelf. Email Bundle is first discovered in this stage. |
| ST-03/S2 | google-workspace | B | Dense rank 1, union rank 6; productivity is the sixth delta group and is dropped by the four-Bundle budget. It was not on the prior Shelf. |
| ST-01/S2 | systematic-debugging | A | Absent from candidates; software-development was absent before this stage. |
| ST-01/S3 | systematic-debugging | C | Still absent from candidates, but software-development is now on Shelf with python-debugpy and other members. Existing domain is never searched locally. |
| ST-01/S3 | python-debugpy | D | Registered since S2 and retrieved again at S3, but software-development is not activated. This is an Active miss, not a Shelf miss. |
| ST-01/S4 | github-code-review | C | GitHub Bundle exists since S1; target absent from S4 candidates. It entered S1 candidates but was truncated as the fifth GitHub member (historical B), so local expansion must recover it at the review stage. |

Five Shelf misses and one additional Active-only miss. No E is assigned as a
primary cause: category boundaries are broad, but these observed omissions have
concrete A/B/C/D mechanisms. The broad software-development grouping is a risk
for semantic matching, not proof by itself of an E failure.

The two C occurrences justify a minimal experimental local-retrieval path. This
is a gate to test a hypothesis, not evidence of an improvement yet. No SRC/LLM,
learned clustering or ontology change is warranted.

### Complete stage trace

#### ST-03/S1

- Active: `email`
- Shelf: `email`, `github`, `note-taking`, `apple`
- BM25 candidates: `email-inbox-triage`, `github-issues`, `obsidian`, `imessage`, `windows-disk-space-audit`
- Dense candidates: `email-inbox-triage`, `github-issues`, `product-price-monitor`, `weekly-review-planning`, `requesting-code-review`
- Union candidates: `email-inbox-triage`, `github-issues`, `obsidian`, `imessage`, `windows-disk-space-audit`, `product-price-monitor`, `weekly-review-planning`, `requesting-code-review`
- Registered Skills: `email-inbox-triage`, `github-issues`, `obsidian`, `imessage`

#### ST-03/S2

- Active: `email`, `apple`
- Shelf: `email`, `github`, `note-taking`, `apple`, `autonomous-ai-agents`
- BM25 candidates: `imessage`, `email-inbox-triage`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`
- Dense candidates: `google-workspace`, `himalaya`, `apple-reminders`, `imessage`, `email-inbox-triage`
- Union candidates: `imessage`, `email-inbox-triage`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`, `google-workspace`, `himalaya`, `apple-reminders`
- Registered Skills: `email-inbox-triage`, `himalaya`, `github-issues`, `obsidian`, `imessage`, `apple-reminders`, `computer-use`

#### ST-01/S1

- Active: `github`
- Shelf: `github`, `devops`
- BM25 candidates: `github-issue-to-pr`, `github-pr-workflow`, `sdlc-review`, `github-issues`, `github-auth`
- Dense candidates: `github-issue-to-pr`, `github-code-review`, `github-pr-workflow`, `github-auth`, `github-issues`
- Union candidates: `github-issue-to-pr`, `github-pr-workflow`, `sdlc-review`, `github-issues`, `github-auth`, `github-code-review`
- Registered Skills: `github-issue-to-pr`, `github-pr-workflow`, `github-issues`, `github-auth`, `sdlc-review`

#### ST-01/S2

- Active: `github`
- Shelf: `github`, `devops`, `autonomous-ai-agents`, `note-taking`, `software-development`
- BM25 candidates: `github-issue-to-pr`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`, `imessage`
- Dense candidates: `github-issue-to-pr`, `python-debugpy`, `github-auth`, `hermes-desktop-plugin-authoring`, `claude-code`
- Union candidates: `github-issue-to-pr`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`, `imessage`, `python-debugpy`, `github-auth`, `hermes-desktop-plugin-authoring`, `claude-code`
- Registered Skills: `github-issue-to-pr`, `github-pr-workflow`, `github-issues`, `github-auth`, `sdlc-review`, `computer-use`, `claude-code`, `obsidian`, `inspecting-hermes-desktop-dom`, `python-debugpy`, `hermes-desktop-plugin-authoring`

#### ST-01/S3

- Active: `github`, `note-taking`
- Shelf: `github`, `devops`, `autonomous-ai-agents`, `note-taking`, `software-development`, `apple`
- BM25 candidates: `obsidian`, `imessage`, `computer-use`, `inspecting-hermes-desktop-dom`, `windows-disk-space-audit`
- Dense candidates: `sdlc-review`, `python-debugpy`, `blocked-page-recovery`, `quiz-bank-ocr-audit`, `test-driven-development`
- Union candidates: `obsidian`, `imessage`, `computer-use`, `inspecting-hermes-desktop-dom`, `windows-disk-space-audit`, `sdlc-review`, `python-debugpy`, `blocked-page-recovery`, `quiz-bank-ocr-audit`, `test-driven-development`
- Registered Skills: `github-issue-to-pr`, `github-pr-workflow`, `github-issues`, `github-auth`, `sdlc-review`, `computer-use`, `claude-code`, `obsidian`, `inspecting-hermes-desktop-dom`, `python-debugpy`, `hermes-desktop-plugin-authoring`, `test-driven-development`, `imessage`

#### ST-01/S4

- Active: `github`, `note-taking`
- Shelf: `github`, `devops`, `autonomous-ai-agents`, `note-taking`, `software-development`, `apple`
- BM25 candidates: `github-pr-workflow`, `imessage`, `github-issue-to-pr`, `computer-use`, `obsidian`
- Dense candidates: `github-issue-to-pr`, `github-pr-workflow`, `requesting-code-review`, `merge-reconciler`, `pdf`
- Union candidates: `github-pr-workflow`, `imessage`, `github-issue-to-pr`, `computer-use`, `obsidian`, `requesting-code-review`, `merge-reconciler`, `pdf`
- Registered Skills: `github-issue-to-pr`, `github-pr-workflow`, `github-issues`, `github-auth`, `sdlc-review`, `computer-use`, `claude-code`, `merge-reconciler`, `obsidian`, `inspecting-hermes-desktop-dom`, `python-debugpy`, `hermes-desktop-plugin-authoring`, `test-driven-development`, `imessage`

## Minimal hierarchical experiment

Implemented as opt-in `eval stage-bundle --hierarchical-ab`; the original CLI
baseline and SRC behavior are preserved. The A arm reproduces every baseline
stage's candidate order, Shelf, Active set and recall (checked against saved JSON).

The matcher uses the same cached Dense encoder over one synthetic metadata
record per existing Shelf Bundle: descriptor + use_when + registered members'
name/description/tags. No future/unregistered member is used for matching and
no Gold label is used in routing. Fixed first-pass cosine threshold **0.35** and
best-versus-second margin **0.05** were chosen before the A/B run, with no Gold
parameter sweep. On a match, BM25 metadata and Dense retrieval search all corpus
Skills with that Bundle's existing deterministic group key, including absent
members. K and Bundle budgets remain 5/4/4. No match uses the original global
retrievers. Local indices reuse the loaded encoder and are cached per Bundle.
Existing integration appends Skills and activates the best delta Bundle.

### Reproduction

```bash
export PATH=/home/lixianbao/miniforge3/envs/skill-control-plane/bin:$PATH
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
skill-control-plane eval stage-bundle /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 --max-bundles 4 --max-skills-per-bundle 4 --json \
  > local_artifacts/experiments/stage-bundle-local87-v0.2.json
# Add --hierarchical-ab to the same command and save to:
# local_artifacts/experiments/stage-bundle-local87-v0.2-ab.json
pytest -q
```

### A/B results

Required-Skill recall below measures the **current candidate pool**, macro-averaged
over six stages; Shelf and Active recall use required_now and the same stage
averaging. Route rates, mean corpus size and surface growth use the **four evidence
transitions**, excluding initial retrieval. Corpus size counts unique Skill
records in the search domain (both retrievers search that domain). Descriptor
search is reported separately; these counts are not wall-clock speedups and omit
embedding/index construction costs.

Wrong-Bundle match is measured against **new_required**, falling back to
required_now only when no new requirement exists. A match to an old still-required
capability does not count as a correct route for new evidence. The denominator
is accepted Bundle matches: B has 1 wrong / 2 matches; A has no matches (the
serialized 0 is not an estimate of matcher accuracy). This definition uses Gold
only after routing and is a group-level proxy, not a semantic correctness oracle.

| Metric | A: global-only | B: hierarchical |
|---|---:|---:|
| required_skill_recall | 0.555556 | 0.722222 |
| mean_active_required_recall | 0.388889 | 0.555556 |
| mean_shelf_required_recall | 0.444444 | 0.611111 |
| global_retrieval_rate | 1.000000 | 0.500000 |
| bundle_local_retrieval_rate | 0.000000 | 0.500000 |
| mean_search_corpus_size | 87.000000 | 47.000000 |
| mean_matcher_corpus_size | 0.000000 | 3.500000 |
| wrong_bundle_match_rate | 0.000000 | 0.500000 |
| mean_registered_surface_growth | 3.000000 | 2.750000 |
| mean_active_surface_growth | 1.000000 | 1.500000 |
| mean_final_registered_skill_count | 10.500000 | 10.000000 |
| mean_final_active_skill_count | 4.500000 | 5.500000 |

### B stage trace and remaining failures

#### ST-03/S1

Route: global; matched Bundle: None; Skill corpus: 87; score: None; margin: None.

- Active: `email`
- Shelf: `email`, `github`, `note-taking`, `apple`
- Candidates: `email-inbox-triage`, `github-issues`, `obsidian`, `imessage`, `windows-disk-space-audit`, `product-price-monitor`, `weekly-review-planning`, `requesting-code-review`
- Shelf misses: `himalaya`
- Active misses: `himalaya`

#### ST-03/S2

Route: global; matched Bundle: None; Skill corpus: 87; score: 0.1947766131260857; margin: 0.0001561585878298022.

- Active: `email`, `apple`
- Shelf: `email`, `github`, `note-taking`, `apple`, `autonomous-ai-agents`
- Candidates: `imessage`, `email-inbox-triage`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`, `google-workspace`, `himalaya`, `apple-reminders`
- Shelf misses: `google-workspace`
- Active misses: `google-workspace`

#### ST-01/S1

Route: global; matched Bundle: None; Skill corpus: 87; score: None; margin: None.

- Active: `github`
- Shelf: `github`, `devops`
- Candidates: `github-issue-to-pr`, `github-pr-workflow`, `sdlc-review`, `github-issues`, `github-auth`, `github-code-review`
- Shelf misses: (none)
- Active misses: (none)

#### ST-01/S2

Route: bundle-local; matched Bundle: github; Skill corpus: 7; score: 0.3587628558066717; margin: 0.2977112306598526.

- Active: `github`
- Shelf: `github`, `devops`
- Candidates: `github-issue-to-pr`, `github-pr-workflow`, `github-auth`, `codebase-inspection`, `github-issues`
- Shelf misses: `systematic-debugging`
- Active misses: `systematic-debugging`

#### ST-01/S3

Route: global; matched Bundle: None; Skill corpus: 87; score: 0.18655741294892328; margin: 0.02625623034054622.

- Active: `github`, `note-taking`
- Shelf: `github`, `devops`, `note-taking`, `apple`, `autonomous-ai-agents`, `software-development`
- Candidates: `obsidian`, `imessage`, `computer-use`, `inspecting-hermes-desktop-dom`, `windows-disk-space-audit`, `sdlc-review`, `python-debugpy`, `blocked-page-recovery`, `quiz-bank-ocr-audit`, `test-driven-development`
- Shelf misses: `systematic-debugging`
- Active misses: `python-debugpy`, `systematic-debugging`

#### ST-01/S4

Route: bundle-local; matched Bundle: github; Skill corpus: 7; score: 0.41203487355953705; margin: 0.20450463353383458.

- Active: `github`, `note-taking`
- Shelf: `github`, `devops`, `note-taking`, `apple`, `autonomous-ai-agents`, `software-development`
- Candidates: `github-pr-workflow`, `github-issue-to-pr`, `github-code-review`, `github-issues`, `github-repo-management`
- Shelf misses: (none)
- Active misses: (none)

| Remaining required miss in B | Class | Explanation |
|---|---|---|
| ST-03/S1 himalaya | A | Initial retrieval unchanged. |
| ST-03/S2 google-workspace | B | Global fallback still loses the productivity group to the delta budget. |
| ST-01/S2 systematic-debugging | A | Wrong GitHub match prevents discovery of software-development; target never enters this stage's candidates. Routing error is counted separately. |
| ST-01/S3 systematic-debugging | A | Now no software-development Bundle exists before S3 because B routed S2 locally. Global fallback still misses the exact Skill. Baseline's C has changed provenance, not been solved. |
| ST-01/S3 python-debugpy | D | Global retrieval registers the debugger, but the best delta activates note-taking. |

The meaningful success is ST-01/S4: GitHub-local search (7 of 87 Skills) finds
`github-code-review` and expands the already-active GitHub Bundle. This fixes
one of two baseline C occurrences. Shelf and Active recalls each rise by
**16.67 percentage points**, entirely from this one stage. The other local route,
ST-01/S2, is wrong for the newly required debugging capability and prevents early
software-development discovery. Baseline S3's C persists as an A failure in B.

The experiment supports **retaining an opt-in bundle-local retrieval baseline**,
not making this matcher the default. Two cases / six stages do not establish
robustness. Biggest remaining issues are matcher confusion between an ongoing
task and its new capability need, raw-evidence retrieval misses, BM25-first union
ordering plus Bundle/member budgets, and activation based only on best delta.
Broad category boundaries remain a limitation, but no missing target here
requires an E classification to explain its immediate cause.

Surface counts are Skill counts rather than tokens: B slightly reduces final
registration (10.5 → 10 mean Skills) but increases active exposure (4.5 → 5.5).
No claim of net latency or prompt-token savings is made.

### Validation

Full `pytest -q`: **48 passed**. Tests cover confident matching, absent-member
recovery, activation and monotonic/idempotent integration, low-score/ambiguous/
empty matches falling back globally, new-Bundle creation, global policy bypass,
A/B denominators and wrong matches to old capabilities, exact preservation of
old evaluator stage output, shared Dense encoder with registered-only matching
metadata, and opt-in CLI parsing. The real official baseline CLI was also run
in text mode after changes. No merge to main and no SRC/LLM changes.
