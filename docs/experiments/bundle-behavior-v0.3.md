# Bundle behavior experiment v0.3

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


This experiment observes four consecutive tasks through one unchanged
`ExperimentalSkillAgent`, one recording client backed by one conversation client,
and one growing history. It preserves RuntimeCapabilityState across turns and
copies each per-run trace before the next `run()` resets that trace. Runtime,
retrieval, Skill instruction delivery and Bundle metadata rendering are unchanged.

The fixed tasks live in `src/skill_control_plane/evals/bundle_behavior.py` and are
supplied sequentially, without expected actions, candidate IDs, or evaluator
results in model context. Tasks request plans based on installed instructions;
there are no execution tools or input documents. The final task changes domain
explicitly to a GitHub/Python CI debugging problem.

| Turn | Task | Predeclared expectation |
| --- | --- | --- |
| 1 | Scanned PDF → extracted text/tables → editable presentation for reusable team work | Search, select pdf/ocr-and-documents/powerpoint, CREATE a document Bundle |
| 2 | Modify title, slide order and speaker notes of the same presentation | Use existing capabilities; no search; preserve state |
| 3 | Export extracted tables into an Excel workbook with formulas/formatting | Search xlsx; EXTEND the original document Bundle |
| 4 | Investigate intermittent GitHub/Python CI failures | Search; separate CREATE or DIRECT; preserve document Bundle |

## Method and interpretation

The evaluator runs after all turns. Its checks never alter model requests, choose
skills, patch decisions or retry until the expectations pass. Deviations are
recorded alongside successful turns and do not stop later tasks. Protocol/validation errors are recorded without repair; the runner advances to
the next user task with exactly the surviving state and history on the same Agent.
Such runs are labelled `completed_with_turn_errors` and the affected turns fail
`turn_completed`. Transport/provider errors stop the run with the partial trace.
The script returns a nonzero exit for errors; completed runs with behavioral
mismatches remain valid experimental observations.

For each turn the artifact records the task, actions, load flag, complete retrieval
results with representations/evidence, state before/after, final response, copied
step trace, history offsets and the Bundle metadata extracted from every actual
system message. Full model messages and final conversation history are also saved.
The existing v0.2 wire-message audit verifies history continuity and body placement.

These checks are observational. A DIRECT/CREATE label alone does not establish
semantic relevance of every selected skill; inspect the actual choices and reasons.
Similarly, a single conversation with both Bundle metadata and previously delivered
Skill instructions does not isolate Bundle metadata as the cause of later behavior.
The experiment can show behavior consistent with the mechanism, but a causal claim
would require a separate controlled comparison. No such comparison is run here.

Deterministic tests use scripted completions to verify that a satisfied follow-up
can finish without retrieval, related skills can extend an existing Bundle,
unrelated skills can stay separate, and state/history survive all turns. Negative
cases verify detection of redundant retrieval, related CREATE instead of EXTEND,
and a schema-valid but inappropriate unrelated EXTEND. These tests do not claim to
predict or enforce real model semantics.

## Running

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_bundle_behavior.py tests/test_experimental_agent.py tests/test_runtime_capability_loading.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check

set -a
source .env
set +a
PYTHONPATH=src .venv/bin/python scripts/bundle_behavior_e2e.py
```

The live script uses the frozen current87 registry/cards, configured GLM chat and
Dense providers, BM25 + Dense + RRF, and no lexical-only fallback. It writes unique
artifacts under `local_artifacts/bundle-behavior/live-*/trace.json`, checkpointing
requests/responses and completed turns. Credentials are not stored. No lifecycle,
hard trigger, sufficiency strategy, OpenPI integration or runtime policy is added.

## Live observations (2026-09-11)

Baseline v0.2 was committed and pushed as
`d51cae3da68814ef392b516b94903af7989cb362`; local and origin branch SHAs were verified
identical before creating `feature/bundle-behavior-experiment-v0.3`. The v0.3
experiment adds only the evaluator/runner, script, deterministic tests and this
report. Runtime/retrieval/v0.2 injection code remains byte-for-byte unchanged.

The complete four-turn artifact is:
`local_artifacts/bundle-behavior/live-_nkfzcf0/trace.json`.
It started at 2026-09-10T16:38:35 UTC (2026-09-11 00:38:35 Asia/Shanghai), using
`GLM-5.3-flash`, `GLM-Embedding-3` (2048 dimensions), and BM25 + Dense + RRF.

**Capability organization matched all four turn expectations, but the live E2E
failed strict final-action validity in all four turns.** Its recorded status is
`completed_with_turn_errors`, with `matches_all_expectations: false`. The evaluator
keeps `turn_completed: false` for every turn; all other recorded behavior checks
are true. These failures are not repaired or counted as successful final actions.

Let D be `cap-b685d243ab8c481e8546db55e7f650da`. Its purpose is:
“扫描版PDF文字/表格提取(OCR)并制作可编辑pptx演示文稿的可复用文档工作流”.

| Turn | Observed search / organization | Bundle before → after | Direct state after | Behavior vs expectation |
| --- | --- | --- | --- | --- |
| 1 | Search → CREATE selecting pdf, ocr-and-documents, powerpoint | Empty → D(pdf, ocr-and-documents, powerpoint) | Empty | Match |
| 2 | No search and no apply; prose response explicitly refers to loaded powerpoint guidance | D → unchanged D | Empty | Reuse/no-search match; valid final action absent |
| 3 | Search → EXTEND D with xlsx | D(3 members) → D(4 members, adding xlsx) | Empty | Match |
| 4 | Search → DIRECT selecting github-pr-workflow, github-code-review, python-debugpy | D(4 members) → unchanged D | Those three GitHub/debugging skills | Match; no unrelated EXTEND |

Search results in ranking order:

- Turn 1: ocr-and-documents, quiz-bank-ocr-audit, pdf, powerpoint, nano-pdf,
  document-to-action-items, xlsx, arxiv, docx, llm-wiki.
- Turn 2: no retrieval.
- Turn 3: xlsx, docx, powerpoint, windows-system-maintenance, quiz-bank-ocr-audit,
  pdf, nano-pdf, apple-notes, excalidraw, airtable.
- Turn 4: github-issue-to-pr, github-pr-workflow, github-code-review, github-issues,
  github-repo-management, python-debugpy, claude-code, codebase-inspection, codex,
  github-auth.

Each actual system surface is recorded in both the per-call and per-turn data.
Turn 1 starts empty and displays D after CREATE. Turn 2 displays the existing
three-member D. Turn 3 starts with that D and displays its fourth member after
EXTEND. Turn 4 keeps the four-member D; DIRECT introduces no extra system surface.

There were **3 / 1 / 3 / 3 model calls** per turn. Actual request message counts
were **2, 4, 6, 8, 10, 12, 14, 16, 18, 20**. The same Agent/client/history served
all calls. Completed history boundaries were **0→6→8→14→20**. The existing v0.2
wire audit passed for all requests, verifying preserved history, selected bodies
only in history, no body copies in system and valid compact member metadata.

### Protocol failures and prior attempts

In the complete four-turn run, Turn 1's final JSON contained literal control
characters inside a string. Turns 2–4 returned prose instead of the required JSON
`final` action. All raised `JSONDecodeError`. Successful CREATE/EXTEND/DIRECT state
updates survived these later errors, and the next task used the same state and
unmodified history, including the invalid raw assistant reply. No final text was
reformatted into valid actions by the experiment.

Two earlier attempts stopped in Turn 1 under the runner's initial stop-on-any-error
policy. Both successfully searched and created a document Bundle before malformed
final JSON (unescaped quotes) aborted the run:

- `local_artifacts/bundle-behavior/live-x9h91zbv/trace.json`
- `local_artifacts/bundle-behavior/live-no8p35xp/trace.json`

To collect the requested four-turn observations, only the experimental runner was
changed to advance after recorded protocol/validation errors. The same fixed tasks,
model configuration and runtime code were used in the complete run; no response
repair or retry-until-behavior-passes strategy was introduced. All three artifacts
are retained. No further live retries were performed.

### Conclusions and verification

Observed choices support maintained-context reuse, related capability extension,
and separation of unrelated capability needs in this conversation. They do not
isolate Bundle metadata from retained Skill instructions/task history as the cause.
The model's final-action adherence is a separate, repeatedly observed failure;
this run is not an end-to-end protocol success and no external task was executed.

Targeted pytest: **60 passed**. Full pytest: **209 passed**.
`git diff --check` passed. The current runtime, retrieval implementation, injection
path and legacy live script have no diff against the synchronized v0.2 baseline.
The v0.3 experiment changes remain uncommitted for review.

## Native tool-calling rerun (2026-09-11)

The Agent's text JSON action protocol was replaced with provider-native,
OpenAI-compatible function calling. This changed only the conversation transport:
Skill search, Harness validation, Bundle state and the body-in-history policy are
unchanged. The provider receives `load_capability` and `apply_capability` schemas;
assistant tool-call messages and matching `role: tool` results, keyed by
`tool_call_id`, remain in the same history. Ordinary assistant content is final.

The successful complete artifact is
`local_artifacts/bundle-behavior/live-f1pw3y4x/trace.json`. It used
`GLM-5.3-flash`, real `GLM-Embedding-3`, BM25 + Dense + RRF, and reports
`status: completed`, `matches_all_expectations: true`, and `context_audit_status:
passed`. There were no JSON action-protocol errors.

| Turn | Native tool behavior | Resulting state |
| --- | --- | --- |
| 1 | Two searches, then CREATE | document Bundle with ocr-and-documents, pdf, powerpoint |
| 2 | No tool calls; ordinary content final | document Bundle unchanged |
| 3 | Search, then EXTEND original Bundle with xlsx | document Bundle gains xlsx |
| 4 | Search, then DIRECT github-pr-workflow and systematic-debugging | document Bundle unchanged; two direct skills |

Turn 1 also demonstrates correction in one history: its first CREATE lacked the
required `reason`, so Harness returned a paired native tool error without mutating
state. The model then issued a valid CREATE. This is a validation event, not a
JSON formatting failure. The call counts were 4 / 1 / 3 / 3 and message counts
were 2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23. The audit verified every tool call had
a matching tool result, previous history remained intact, bodies stayed out of
system context, and selected bodies remained only in history.

An independent concurrent complete run,
`local_artifacts/bundle-behavior/live-lepesxpx/trace.json`, had valid native
protocol but did not meet Bundle-organization expectations: after two searches,
the model's invalid CREATE attempts were rejected because the latest-search rule
was in force, and it eventually chose DIRECT. This remains an observed semantic
variation, not a protocol error. The main trace above is the successful requested
four-turn outcome.
