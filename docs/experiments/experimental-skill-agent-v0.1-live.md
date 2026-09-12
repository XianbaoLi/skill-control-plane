# Experimental Skill Agent v0.1 live acceptance

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


A live run on 2026-09-10 completed the single-model dynamic loading cycle with
`GLM-5.3-flash`, real `GLM-Embedding-3` Dense retrieval and BM25/RRF over current87.
The task asked for a concrete installed-tool workflow to extract scanned PDF
text/tables and prepare an editable presentation. It did not name Skills or
instruct the model to emit a load/apply action.

The full local trace (including actual messages, responses and state snapshots):
`local_artifacts/experimental-skill-agent/live-n1djgwsw/trace.json`.
This generated artifact is local and is not committed to the repository.

| Step | Model action | Result | Bodies injected at model call |
| --- | --- | --- | --- |
| 1 | load_capability | Requested scanned PDF OCR text/table extraction; 10 real candidates returned; state unchanged | None |
| 2 | apply_capability / DIRECT | Selected ocr-and-documents, pdf, powerpoint; added direct skills | None |
| 3 | final | Produced a proposed tool/dependency/command/verification workflow | ocr-and-documents, pdf, powerpoint |

The returned candidates, in rank order, were `ocr-and-documents`,
`quiz-bank-ocr-audit`, `pdf`, `document-to-action-items`, `nano-pdf`, `arxiv`, `xlsx`,
`powerpoint`, `docx`, `llm-wiki`. No decision or candidate fixture was supplied.
There was no Dense fallback, hidden resolver call, or deterministic trigger.

Actual message counts were **2 → 4 → 6**, with one current system message per
request. A post-run audit verified that all prior history was preserved, injected
bodies exactly matched `SkillRecord.body`, and no unselected registry body appeared
in any system message. The run status was `success`.

Network-free acceptance after this change: targeted tests **56 passed**; full
suite **199 passed**. `git diff --check` passed.

This run verifies the mechanism and one observed self-trigger, not general
self-trigger quality or correctness of every proposed command. No commands from
the model's final answer were executed. CREATE and EXTEND, invalid candidate and
target selection, stale candidate rejection, failures, and step limits are covered
by deterministic unit tests; they were not separate live cases in this run.
