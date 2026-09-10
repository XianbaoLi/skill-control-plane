# Experimental Skill Agent v0.2 live acceptance

Run: 2026-09-11 00:10 Asia/Shanghai (trace start: 2026-09-10T16:10:51 UTC).
Result: **success**, using `GLM-5.3-flash` and real `GLM-Embedding-3` Dense + BM25/RRF
against current87. No lexical-only fallback, candidate fixtures or scripted model
decisions were used.

Full local trace, including actual wire messages and a per-call `context_audit`:
`local_artifacts/experimental-skill-agent/live-qrqk14rq/trace.json`.
The generated trace remains local; this report records its observable results.

The user task requested a reusable team workflow for scanned PDF text/table
extraction and editable presentations, based on installed operating instructions.
It named neither catalog IDs nor a required load/apply action.

| Call | System capability surface | History available at the call | Model action |
| --- | --- | --- | --- |
| 1 | Empty maintained bundles, no bodies | Task | load_capability: scanned PDF OCR and editable presentation guidance |
| 2 | Empty maintained bundles, no bodies | Task + load action + real candidates/evidence | CREATE with pdf, ocr-and-documents, powerpoint |
| 3 | Updated Bundle metadata with three members, no bodies | Prior history + apply action + apply result containing the three full bodies | final workflow guidance |

All system messages also contain the fixed Agent rules and action interfaces.
Message counts are **2 → 4 → 6**. The created bundle is
`cap-c08f258a9ecd46c3881c23447f3bd154`, with purpose
“扫描版 PDF 提取文字与表格并生成可编辑演示文稿的可复用文档处理工作流”.
Its system metadata includes each member's ID, name and short description only.

The ten retrieved IDs, in order: `ocr-and-documents`, `quiz-bank-ocr-audit`, `pdf`,
`document-to-action-items`, `nano-pdf`, `powerpoint`, `xlsx`, `arxiv`, `docx`,
`llm-wiki`. Only `pdf`, `ocr-and-documents`, and `powerpoint` were selected.

The independent audit of actual requests passed:

- No registry Skill body appears in any system message.
- Load results use RRF and contain candidates/evidence, with no bodies.
- Selected body strings exactly match their registry records in apply results.
- Each selected body is delivered once in history; unselected bodies stay absent.
- All previous history and assistant responses are retained by subsequent calls.
- Call 3 has the new Bundle metadata in system and the three bodies in history.

Verification: targeted pytest **59 passed**, full pytest **202 passed**,
`git diff --check` passed. Deterministic tests also cover DIRECT, EXTEND, initial
nonempty state, short metadata, rejection/atomicity, repeated selection and
continued access to bodies across subsequent `run()` calls.

This experiment proves this information flow and one observed self-trigger/CREATE
choice. It does not establish general task accuracy or execute the model's proposed
commands. Full history is still sent on each request; “once” means one original
body delivery in the conversation, not provider-side caching. No lifecycle,
eviction, sufficiency retry policy, hard trigger or OpenPI integration was added.
