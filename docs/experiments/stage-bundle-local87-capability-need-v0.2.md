# Capability-need representation vs raw evidence — Exact Hermes 87 Skills

## Finding

The joint hypothesis is **not satisfied** in this run. Capability-need extraction
avoids the wrong GitHub-local route at ST-01/S2, but does not recover
`systematic-debugging`. At ST-01/S4 it still recovers `github-code-review`, but
**loses GitHub-local routing** and searches all 87 Skills. R1 routes globally on
all four transitions. Higher Shelf recall is accompanied by greater search and
exposure costs; zero wrong matches with zero accepted matches is not evidence
of a more accurate matcher. Keep this experiment opt-in.

## Fixed setup and input isolation

- Branch: `feature/stage-capability-bundles-v0.2`; no merge to main.
- Real tree: `/mnt/d/Hermes/skills`; rebuilt count **87**.
- Manifest: `local_artifacts/corpora/hermes-local-v0.1/manifest.json`.
- Gold: `evals/gold/stage-transition-v0.2.jsonl` (unchanged).
- Real tree, manifest and every Gold case have snapshot
  `1b7e524ea167fb9057423e999d94ba920e6f11817f7ce06fede3a46722bd4bf6`.
- Dense: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, offline cached.
- K=5 per retriever; Bundle/member budgets 4/4; matcher cosine/margin **0.35/0.05**.
- BM25 metadata, Dense metadata, candidate union order, category/group key,
  local universe, monotonic integration and activation policy unchanged.
- Initial stage uses the initial task exactly as before; no extraction at S1.
- R0 uses raw current evidence. R1 uses the extracted need for **both** Bundle
  matching and downstream local/global retrieval. Initial task never becomes
  part of the runtime retrieval query.

Reused `TextCompleter` from `runtime/llm_context.py`, with an independent narrow
`CapabilityNeedExtractor` protocol and `LLMCapabilityNeedExtractor` adapter.
No SRC prompt/behavior changed, no framework dependency, parser, ontology or
Gold-specific rule was added.

The prompt explicitly treats initial_task as background, permits a new domain,
and asks for the action newly required to progress. The only variable inputs
are initial_task and runtime_evidence; no case ID, rationale, snapshot, labels,
Skill/Bundle lists, ranks, prior route or matcher output reaches the extractor.
The complete prompt is in `runtime/capability_need.py`; exact submitted prompts
are saved in the local input audit and provider audits.

For invalid/empty/failed extraction or self-reported confidence < **0.5**, R1
uses raw evidence in **global-only** retrieval, bypassing the matcher. Null
confidence means unknown and permits a non-empty valid need. The extraction
confidence gate was fixed before the run and does not modify matcher thresholds.
All four final extractions succeeded with confidence 0.98–0.99; no fallback due
to extraction failure occurred in the analyzed run.

## Real completion and reproducibility

The real extractor used **gpt-6-astra** through the already authenticated local
Codex CLI, as a text completer in a fresh empty temporary working directory.
User config, project docs, host Skill discovery, plugins, apps, tools and memory
were disabled for extraction; each request was an independent ephemeral session.
Four accepted provider audits contain only message events, no tool calls.
No credential is stored in the repository. The script's isolation settings follow
the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and the installed CLI's help/features output.

A transport bring-up run was discarded: the CLI encoded an experimental-feature
warning as an `error` item, which the conservative adapter rejected. Its JSON is
preserved as `stage-bundle-local87-v0.2-capability-need-transport-fallback.json`;
it is **not** an R1 representation result. The warning was suppressed without
changing the extraction prompt or routing parameters. Diagnostic responses were
not used to choose phrasing or tune thresholds.

The accepted live run is saved as
`stage-bundle-local87-v0.2-capability-need-live.json`. Its outer execution
environment produced matcher-score differences around 1e-7 from the old local
baseline. To avoid resampling the LLM, the **same four exact accepted completions**
were replayed by SHA-256 of the entire prompt in the original local environment
with CPU retrieval (`CUDA_VISIBLE_DEVICES=''`). The reported JSON is this replay.
Live and replay candidate IDs, routes, registered sets and Active sets all agree.

The original default CLI and original `--hierarchical-ab` were rerun into new
`*-recheck.json` files. Both are **exact full-JSON matches** to their saved prior
outputs. R0's every prior stage field—including floating-point matcher scores—
is also exactly reproduced by the replay. Only the new experiment's aggregate
recall denominator differs intentionally, as explained below.

## Metrics

All three recall metrics below are macro means over the **four runtime evidence
transitions**, excluding S1. `required_skill_recall` scores current candidate IDs
against required_now; Shelf/Active recall score their respective retained sets.
This differs from the previous hierarchical report's six-stage recall means and
must not be compared without accounting for the denominator. Initial-stage
results remain in the trace to audit the unchanged initial retrieval.

Route rates, corpus sizes, matcher sizes and surface growth also use the four
transitions. Final surface counts average the final states of the two cases.
Search-corpus size counts unique Skill records, excluding separately reported
Bundle descriptors and embedding/index construction costs; it is not a latency
measurement. Wrong-Bundle matches are scored after routing using new_required
(required_now only if new_required is empty), divided by accepted matches.

| Metric | R0: raw evidence | R1: capability need |
|---|---:|---:|
| required_skill_recall | 0.708333 | 0.708333 |
| mean_active_required_recall | 0.458333 | 0.541667 |
| mean_shelf_required_recall | 0.541667 | 0.791667 |
| global_retrieval_rate | 0.500000 | 1.000000 |
| bundle_local_retrieval_rate | 0.500000 | 0.000000 |
| wrong_bundle_match_rate | 0.500000 | 0.000000 |
| mean_search_corpus_size | 47.000000 | 87.000000 |
| mean_matcher_corpus_size | 3.500000 | 4.750000 |
| mean_registered_surface_growth | 2.750000 | 3.750000 |
| mean_active_surface_growth | 1.500000 | 2.750000 |
| mean_final_registered_skill_count | 10.000000 | 12.000000 |
| mean_final_active_skill_count | 5.500000 | 8.000000 |

R0 wrong matches: **1/2**. R1: **0/0**, serialized as 0 by the existing evaluator;
interpret as **not estimable**, not 100% matching precision. Candidate recall
is unchanged; Shelf recall rises 25 percentage points and Active recall rises
8.33 points. All transitions now pay for global search, and active/registered
surfaces increase.

## All four evidence transitions

### ST-03/S2

Raw evidence:

> The latest important email says: "Can we meet tomorrow at 3 PM? Please send me an invite."
> The user says: "Yes, add it to my calendar."

**Extracted need (verbatim):**

> Create a calendar event for tomorrow at 3 PM and send an invitation to the email sender.

Confidence: 0.99. Evidence basis: The email requests a meeting invitation, and the user explicitly authorizes adding it to their calendar.

| Observation | R0 | R1 |
|---|---|---|
| Route | global | global |
| Accepted Bundle | None | None |
| Top score | 0.1947766131260857 | 0.24412388800372997 |
| Margin | 0.0001561585878298022 | 0.0001333602617513141 |
| Skill corpus size | 87 | 87 |
| Matcher descriptors | 4 | 4 |

R0 has no confident match and uses global retrieval; google-workspace is a
candidate but its Bundle loses to the budget. R1 also uses global retrieval,
but changes candidate ordering: google-workspace moves to union rank 2 and
productivity is retained. It reaches the Shelf but remains inactive because
imessage/Apple is the best delta. The failure moves from B to D.

**R0 trace:**

- before_bundle_ids: `email`, `github`, `note-taking`, `apple`
- candidate_ids: `imessage`, `email-inbox-triage`, `computer-use`, `obsidian`, `inspecting-hermes-desktop-dom`, `google-workspace`, `himalaya`, `apple-reminders`
- bundle_ids: `email`, `github`, `note-taking`, `apple`, `autonomous-ai-agents`
- active_bundle_ids: `email`, `apple`
- missing_candidate_ids: (none)
- missing_shelf_ids: `google-workspace`
- missing_active_ids: `google-workspace`

**R1 trace:**

- before_bundle_ids: `email`, `github`, `note-taking`, `apple`
- candidate_ids: `imessage`, `google-workspace`, `email-inbox-triage`, `obsidian`, `github-issue-to-pr`, `himalaya`, `apple-reminders`
- bundle_ids: `email`, `github`, `note-taking`, `apple`, `productivity`
- active_bundle_ids: `email`, `apple`
- missing_candidate_ids: (none)
- missing_shelf_ids: (none)
- missing_active_ids: `google-workspace`

### ST-01/S2

Raw evidence:

> CI output: FAILED tests/test_auth.py::test_login
> Python traceback ends with KeyError: 'token' at src/auth.py:87.
> Rerunning the same test reproduces the same failure.

**Extracted need (verbatim):**

> Diagnose and fix a reproducible Python test failure caused by KeyError: 'token' at src/auth.py:87.

Confidence: 0.98. Evidence basis: CI and a repeated test run both fail tests/test_auth.py::test_login with the same missing-key error.

| Observation | R0 | R1 |
|---|---|---|
| Route | bundle-local | global |
| Accepted Bundle | github | None |
| Top score | 0.3587628558066717 | 0.2237890404977848 |
| Margin | 0.2977112306598526 | 0.16338450826102635 |
| Skill corpus size | 7 | 87 |
| Matcher descriptors | 2 | 2 |

R0 confidently selects github and searches seven Skills, so a new debugging
capability cannot be discovered. R1 extracts Python failure diagnosis, rejects
the existing-Bundle match and searches all 87. **Wrong GitHub routing is avoided.**
python-debugpy leads the candidates; software-development is registered and
activated. However, **systematic-debugging is absent from candidate, Shelf and
Active**. The remaining failure is A at retrieval: the need still embeds concrete
error/path details, and this run does not establish that generic systematic
root-cause diagnosis is recovered. No hardcoded enrichment was added.

**R0 trace:**

- before_bundle_ids: `github`, `devops`
- candidate_ids: `github-issue-to-pr`, `github-pr-workflow`, `github-auth`, `codebase-inspection`, `github-issues`
- bundle_ids: `github`, `devops`
- active_bundle_ids: `github`
- missing_candidate_ids: `systematic-debugging`
- missing_shelf_ids: `systematic-debugging`
- missing_active_ids: `systematic-debugging`

**R1 trace:**

- before_bundle_ids: `github`, `devops`
- candidate_ids: `python-debugpy`, `github-issue-to-pr`, `session-librarian`, `design-md`, `arxiv`, `github-auth`, `quiz-bank-ocr-audit`, `hermes-desktop-plugin-authoring`
- bundle_ids: `github`, `devops`, `software-development`, `productivity`, `creative`
- active_bundle_ids: `github`, `software-development`
- missing_candidate_ids: `systematic-debugging`
- missing_shelf_ids: `systematic-debugging`
- missing_active_ids: `systematic-debugging`

### ST-01/S3

Raw evidence:

> The failing test still reproduces after reading the full traceback and showing locals.
> The final traceback does not reveal where the session dictionary loses the token field.
> The user asks to step through build_session() and watch the dictionary change while the test runs.

**Extracted need (verbatim):**

> Step through build_session() during the failing test and watch the session dictionary to identify when the token field disappears.

Confidence: 0.99. Evidence basis: The traceback and locals did not locate the mutation, and the user explicitly requested stepping through execution while watching the dictionary.

| Observation | R0 | R1 |
|---|---|---|
| Route | global | global |
| Accepted Bundle | None | None |
| Top score | 0.18655741294892328 | 0.25564433457157676 |
| Margin | 0.02625623034054622 | 0.06324269859044979 |
| Skill corpus size | 87 | 87 |
| Matcher descriptors | 2 | 5 |

R0 has not discovered software-development at S2 and falls back globally.
It registers python-debugpy but does not activate its Bundle (D), and still
misses systematic-debugging (A). R1 already has software-development active
from S2. Its extracted stepping/watch description still does not meet the
matcher score threshold, so it retrieves globally. python-debugpy remains
active; systematic-debugging is absent. With a relevant Bundle now existing
before this stage, the remaining failure is C (known-Bundle expansion miss).
A missing current candidate github-issue-to-pr is not a Shelf/Active failure:
that Skill remains registered and active from the initial stage.

**R0 trace:**

- before_bundle_ids: `github`, `devops`
- candidate_ids: `obsidian`, `imessage`, `computer-use`, `inspecting-hermes-desktop-dom`, `windows-disk-space-audit`, `sdlc-review`, `python-debugpy`, `blocked-page-recovery`, `quiz-bank-ocr-audit`, `test-driven-development`
- bundle_ids: `github`, `devops`, `note-taking`, `apple`, `autonomous-ai-agents`, `software-development`
- active_bundle_ids: `github`, `note-taking`
- missing_candidate_ids: `github-issue-to-pr`, `systematic-debugging`
- missing_shelf_ids: `systematic-debugging`
- missing_active_ids: `python-debugpy`, `systematic-debugging`

**R1 trace:**

- before_bundle_ids: `github`, `devops`, `software-development`, `productivity`, `creative`
- candidate_ids: `obsidian`, `imessage`, `computer-use`, `inspecting-hermes-desktop-dom`, `competitor-news-monitor`, `python-debugpy`, `test-driven-development`, `spike`, `github-issues`, `node-inspect-debugger`
- bundle_ids: `github`, `devops`, `software-development`, `productivity`, `creative`, `note-taking`, `apple`, `autonomous-ai-agents`
- active_bundle_ids: `github`, `software-development`, `note-taking`
- missing_candidate_ids: `github-issue-to-pr`, `systematic-debugging`
- missing_shelf_ids: `systematic-debugging`
- missing_active_ids: `systematic-debugging`

### ST-01/S4

Raw evidence:

> PR #42 is open and its CI checks are passing.
> The user says: "Before I merge this PR, review the diff and leave feedback on any problems you find."

**Extracted need (verbatim):**

> Review a pull request diff for problems and leave feedback on findings.

Confidence: 0.99. Evidence basis: The PR is open with passing CI, and the user now requests a diff review and feedback before merging.

| Observation | R0 | R1 |
|---|---|---|
| Route | bundle-local | global |
| Accepted Bundle | github | None |
| Top score | 0.41203487355953705 | 0.32491611384645375 |
| Margin | 0.20450463353383458 | 0.044877543726149716 |
| Skill corpus size | 7 | 87 |
| Matcher descriptors | 6 | 8 |

R0 confidently matches github, searches seven Skills and recovers
github-code-review. R1's extracted review phrase fails the unchanged matching
criteria (top score below 0.35 and margin below 0.05), so it searches all 87.
github-code-review still enters the candidates and expands the existing active
GitHub Bundle, making candidate/Shelf/Active coverage successful. **The correct
GitHub-local route is not retained.** This is a routing-efficiency regression,
not an unrecovered required Skill. Later Shelves differ because of previous
R1 routing decisions, so this is a sequential trajectory comparison rather
than a frozen-Shelf matcher-only ablation.

**R0 trace:**

- before_bundle_ids: `github`, `devops`, `note-taking`, `apple`, `autonomous-ai-agents`, `software-development`
- candidate_ids: `github-pr-workflow`, `github-issue-to-pr`, `github-code-review`, `github-issues`, `github-repo-management`
- bundle_ids: `github`, `devops`, `note-taking`, `apple`, `autonomous-ai-agents`, `software-development`
- active_bundle_ids: `github`, `note-taking`
- missing_candidate_ids: (none)
- missing_shelf_ids: (none)
- missing_active_ids: (none)

**R1 trace:**

- before_bundle_ids: `github`, `devops`, `software-development`, `productivity`, `creative`, `note-taking`, `apple`, `autonomous-ai-agents`
- candidate_ids: `sdlc-review`, `imessage`, `github-issue-to-pr`, `docx`, `windows-system-maintenance`, `github-code-review`, `requesting-code-review`, `spike`
- bundle_ids: `github`, `devops`, `software-development`, `productivity`, `creative`, `note-taking`, `apple`, `autonomous-ai-agents`
- active_bundle_ids: `github`, `software-development`, `note-taking`, `devops`
- missing_candidate_ids: (none)
- missing_shelf_ids: (none)
- missing_active_ids: (none)

## Remaining required misses, classified after routing

C takes precedence over A when the relevant Bundle existed before this stage;
B means a candidate is lost by Bundle/member budget; D means registered but
inactive. These are stage-local operational causes. Extraction validity and
wrong routing are separately audited; no category boundary was changed.

| Case/stage | Required Skill | R0 | R1 |
|---|---|---|---|
| ST-03/S1 | himalaya | A | A, unchanged initial retrieval; excluded from runtime metrics |
| ST-03/S2 | google-workspace | B | D, now on Shelf but inactive |
| ST-01/S2 | systematic-debugging | A, wrong local domain | A, global retrieval miss |
| ST-01/S3 | systematic-debugging | A, domain absent beforehand | C, relevant Bundle now known but no expansion recovery |
| ST-01/S3 | python-debugpy | D | Resolved; Bundle already activated at S2 |

No other Shelf/Active required misses remain in either trajectory. There is no
new Skill-coverage failure at S4, but its local-route success criterion fails.

## Conclusion and limits

Capability need is useful for separating an ongoing GitHub task from a newly
needed debugging capability in S2. It does **not** satisfy the requested combined
criterion: S2 avoids wrong GitHub-local routing, **but S4 loses correct local
routing**. Improved Shelf recall alone does not justify adoption. The main
remaining issues are exact systematic-debugging retrieval, insufficient
confident matching of the new phrasing to existing descriptors, and best-delta
activation (calendar Skill remains inactive). More Skills on Shelf and Active
are an exposure cost, not automatically a benefit.

This is one accepted completion sample per transition, four transitions / two
cases, with uncalibrated model confidence. No threshold or prompt sweep, Gold
changes, parser, SRC, category/group-key change or architecture expansion was
performed. The capability-need experiment remains opt-in, as does the prior
hierarchical experiment.

## Reproduce and artifacts

```bash
export PATH=/home/lixianbao/miniforge3/envs/skill-control-plane/bin:$PATH
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
skill-control-plane eval stage-bundle /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 --max-bundles 4 --max-skills-per-bundle 4 \
  --capability-need-ab \
  --capability-need-command 'python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions' \
  --json
```

For a fresh real completion run, replace `--replay-dir ...` in the completion
command with `--model gpt-6-astra --audit-dir <new-directory>`. This uses existing
Codex authentication and may yield different wording; keep its results separate.
Omit `--json` for the text report with all extraction/route traces.

Local artifacts (ignored by git):

- `local_artifacts/experiments/stage-bundle-local87-v0.2-capability-need-ab.json`: analyzed exact-completion replay.
- `local_artifacts/experiments/stage-bundle-local87-v0.2-capability-need-ab.txt`: official CLI text report from the same replay.
- `local_artifacts/experiments/stage-bundle-local87-v0.2-capability-need-live.json`: accepted live end-to-end run.
- `local_artifacts/experiments/capability-need-local87-extractions/`: four exact prompt/completion/event audits, keyed by prompt SHA-256.
- `local_artifacts/experiments/capability-need-local87-input-audit.json`: full outbound prompt audit, before extraction.
- `local_artifacts/experiments/stage-bundle-local87-v0.2-capability-need-provenance.json`: artifact/source hashes and verification results.

## Validation

Full `pytest -q`: **71 passed** (previous 48 retained). New tests cover injected
extractors and fake completers, Gold-label perturbation with identical prompts,
initial-stage bypass and transition-only calls, need-vs-raw matcher inputs,
unchanged 0.35/0.05 thresholds, malformed/empty/low-confidence/null-confidence/
exception/timeout handling, conservative global fallback, prior R0 stage
identity, transition metric denominators, default/mutually-exclusive CLI flags,
text traces, shell-free completion invocation, provider event auditing and tool
rejection, and exact-input replay with no network call. Both real old CLI
baselines retain exact JSON equality. No main merge.
