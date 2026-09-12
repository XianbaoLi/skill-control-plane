# V0.7 真实 Hermes 87-skill E2E 验收

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


结论：检索 → Resolver 数据链路可运行，但四项真实语义验收仅 CREATE 符合预期。当前不建议作为“Capability Loading E2E 已验收”提交。未修改 V0.7 产品实现，未 commit/push。

## 语料与执行边界

- 使用 `local_artifacts/v0.5/hermes-current87`，87 skills，snapshot `f670e9d5ecdbb299a5c5ac4fcc48ece35e0adadcfb7e24b96850a4ec5379e509`。
- manifest 身份和 RetrievalCard 的完整 ID/content_hash 对齐校验均通过。使用的是此前 65/39-query 实验冻结的 current87。
- 历史重建 snapshot `1b7e524e…` 的 cards 有 9 条 stale，不能声称本次验收覆盖该历史版本；没有绕过校验或重新生成 cards。
- Fixture 在第一次检索前定义；所有 case 固定默认 k=5，没有按结果修改 query、bundle 成员或阈值，没有手工注入候选。
- fixture 的 validated=true 表示人为声明的验收前置条件，不代表这些 bundle 已被真实执行验证。
- 新查询 BigModel embedding-3/2048 返回 HTTP 429；四项主验收使用实际 BM25 + RetrievalCard。
- 另用已有真实查询/语料向量重算 BigModelDenseRetriever + BM25 + RRF(60)，source depth=10。不是 mock 或直接回放排名。
- 四条缓存 RRF 查询的目标 skill 排名与历史 full 报告一致，测试独立比较该结果。
- 原 corpus/cards/embedding 缓存保持不变；新运行缓存和完整 representation/evidence 写入 local_artifacts/v0.7/real-e2e/。

## 四项主验收（真实 BM25）

### direct: 预期 DIRECT，实际 CREATE

Need: Search arXiv for recent papers on retrieval augmented generation.

| Rank | Skill ID | BM25 score | Retrieval reason |
|---|---|---:|---|
| 1 | arxiv | 18.361351 | matched_terms: search, arxiv, for, papers, on |
| 2 | ocr-and-documents | 12.188190 | matched_terms: search, arxiv, papers, on |
| 3 | research-paper-writing | 9.579115 | matched_terms: arxiv, for, papers |
| 4 | simplify-code | 6.665501 | matched_terms: for, recent, on |
| 5 | imessage | 6.229992 | matched_terms: for, recent, on |

| Existing bundle | Covered skills | Missing skills | Purpose compatible |
|---|---|---|---|
| github-pr-review | 无 | arxiv, ocr-and-documents, research-paper-writing, simplify-code, imessage | False |
| python-failure-investigation | 无 | arxiv, ocr-and-documents, research-paper-writing, simplify-code, imessage | False |

Decision reason: arxiv 排第一，但所有五个检索候选都被视为组合，因此未进入单 skill DIRECT 分支。

### reuse: 预期 REUSE，实际 CREATE

Need: Review a GitHub pull request, inspect its diff, leave inline comments, and verify CI before merging.

| Rank | Skill ID | BM25 score | Retrieval reason |
|---|---|---:|---|
| 1 | github-code-review | 33.098090 | matched_terms: review, a, github, pull, request, diff, inline, comments, and, before |
| 2 | github-pr-workflow | 27.506533 | matched_terms: review, a, github, pull, request, diff, and, ci, merging. |
| 3 | github-issue-to-pr | 20.159507 | matched_terms: a, github, pull, request, its, and, ci |
| 4 | codex | 16.048309 | matched_terms: review, a, github, pull, request, comments, and |
| 5 | sdlc-review | 12.231080 | matched_terms: review, a, request, its, and, verify |

| Existing bundle | Covered skills | Missing skills | Purpose compatible |
|---|---|---|---|
| github-pr-review | github-code-review, github-pr-workflow, github-issue-to-pr | codex, sdlc-review | True |
| python-failure-investigation | github-pr-workflow | github-code-review, github-issue-to-pr, codex, sdlc-review | False |

Decision reason: PR bundle 覆盖 3/5，缺 codex、sdlc-review；REUSE 要求全覆盖，EXTEND 要求至少 2/3 且最多缺 1 项，均不满足。

### extend: 预期 EXTEND，实际 CREATE

Need: Investigate a failing Python test, find the root cause, inspect the codebase, and use a debugger to inspect runtime variables.

| Rank | Skill ID | BM25 score | Retrieval reason |
|---|---|---:|---|
| 1 | systematic-debugging | 28.066986 | matched_terms: investigate, a, failing, test, find, the, root, cause, and, use, to |
| 2 | python-debugpy | 21.989588 | matched_terms: a, failing, python, test, the, inspect, and, use, debugger, to |
| 3 | node-inspect-debugger | 18.558722 | matched_terms: a, test, find, the, inspect, and, use, debugger, to |
| 4 | codebase-inspection | 15.576873 | matched_terms: a, python, find, inspect, codebase, and, use, to |
| 5 | github-issue-to-pr | 11.727870 | matched_terms: a, test, the, root, cause, and, use, to |

| Existing bundle | Covered skills | Missing skills | Purpose compatible |
|---|---|---|---|
| github-pr-review | github-issue-to-pr | systematic-debugging, python-debugpy, node-inspect-debugger, codebase-inspection | False |
| python-failure-investigation | systematic-debugging, codebase-inspection | python-debugpy, node-inspect-debugger, github-issue-to-pr | True |

Decision reason: Python bundle 覆盖 2/5，缺 python-debugpy、node-inspect-debugger、github-issue-to-pr；相关候选与替代/噪声候选未被区分，未达到 EXTEND 条件。

### create: 预期 CREATE，实际 CREATE

Need: Extract text from scanned PDF documents and turn the findings into a PowerPoint presentation.

| Rank | Skill ID | BM25 score | Retrieval reason |
|---|---|---:|---|
| 1 | ocr-and-documents | 24.191358 | matched_terms: extract, text, from, scanned, pdf, documents, and, a |
| 2 | document-to-action-items | 22.832224 | matched_terms: extract, text, from, scanned, documents, and, turn, into, a |
| 3 | pdf | 21.496965 | matched_terms: extract, text, from, scanned, pdf, documents, and, a |
| 4 | powerpoint | 17.097073 | matched_terms: extract, text, from, pdf, and, the, a, powerpoint |
| 5 | nano-pdf | 12.408089 | matched_terms: text, pdf, documents, the, a |

| Existing bundle | Covered skills | Missing skills | Purpose compatible |
|---|---|---|---|
| github-pr-review | 无 | ocr-and-documents, document-to-action-items, pdf, powerpoint, nano-pdf | False |
| python-failure-investigation | 无 | ocr-and-documents, document-to-action-items, pdf, powerpoint, nano-pdf | False |

Decision reason: 与两个已有 bundle 均无共享 skill，purpose 不匹配，CREATE 合理。但 OCR/PDF 候选之间仍可能有替代或冗余关系。

## 真实缓存 Dense/RRF 补充链路

### ST-01/S3 → CREATE

Need: Step through build_session() during the failing test and watch the session dictionary to identify where the token field is lost.

| Rank | Skill ID | RRF score | Retrieval reason |
|---|---|---:|---|
| 1 | test-driven-development | 0.031754 | matched_terms: the, failing, test, and, to; semantic_similarity: 0.3903 |
| 2 | node-inspect-debugger | 0.031319 | matched_terms: step, through, the, test, and, watch, to, where, is; semantic_similarity: 0.3746 |
| 3 | python-debugpy | 0.031258 | matched_terms: step, through, the, failing, test, and, to, is; semantic_similarity: 0.3773 |
| 4 | systematic-debugging | 0.030622 | matched_terms: through, the, failing, test, and, to, where, is; semantic_similarity: 0.3688 |
| 5 | session-librarian | 0.016393 | semantic_similarity: 0.4045 |

Existing bundle comparison:
- github-pr-review: covered=[]; missing=['test-driven-development', 'node-inspect-debugger', 'python-debugpy', 'systematic-debugging', 'session-librarian']; purpose_compatible=False.
- python-failure-investigation: covered=['systematic-debugging']; missing=['test-driven-development', 'node-inspect-debugger', 'python-debugpy', 'session-librarian']; purpose_compatible=False.

Decision reason: 无 purpose-compatible 的合格 bundle，返回 CREATE。

### ST-01/S4 → CREATE

Need: Review the pull request diff for problems and leave feedback on findings.

| Rank | Skill ID | RRF score | Retrieval reason |
|---|---|---:|---|
| 1 | github-code-review | 0.032787 | matched_terms: review, pull, request, diff, for, and, feedback, on; semantic_similarity: 0.6453 |
| 2 | github-pr-workflow | 0.031754 | matched_terms: review, the, pull, request, diff, for, and, on; semantic_similarity: 0.5701 |
| 3 | github-issue-to-pr | 0.031514 | matched_terms: the, pull, request, for, and, on; semantic_similarity: 0.5875 |
| 4 | requesting-code-review | 0.030579 | matched_terms: review, diff, for, and, on; semantic_similarity: 0.5721 |
| 5 | sdlc-review | 0.030536 | matched_terms: review, the, request, for, and; semantic_similarity: 0.5173 |

Existing bundle comparison:
- github-pr-review: covered=['github-code-review', 'github-pr-workflow', 'github-issue-to-pr', 'requesting-code-review']; missing=['sdlc-review']; purpose_compatible=False.
- python-failure-investigation: covered=['github-pr-workflow', 'requesting-code-review']; missing=['github-code-review', 'github-issue-to-pr', 'sdlc-review']; purpose_compatible=False.

Decision reason: 无 purpose-compatible 的合格 bundle，返回 CREATE。

### AT-02/S2 → CREATE

Need: Perform local OCR on all 18 scanned PDF pages to extract page-associated text.

| Rank | Skill ID | RRF score | Retrieval reason |
|---|---|---:|---|
| 1 | ocr-and-documents | 0.032787 | matched_terms: perform, local, ocr, on, scanned, pdf, pages, to, extract; semantic_similarity: 0.6097 |
| 2 | pdf | 0.032258 | matched_terms: ocr, on, scanned, pdf, pages, to, extract, text.; semantic_similarity: 0.5677 |
| 3 | quiz-bank-ocr-audit | 0.031498 | matched_terms: ocr, pdf; semantic_similarity: 0.4953 |
| 4 | document-to-action-items | 0.031258 | matched_terms: ocr, scanned, to, extract; semantic_similarity: 0.4650 |
| 5 | nano-pdf | 0.030331 | matched_terms: on, pdf, to; semantic_similarity: 0.4761 |

Existing bundle comparison:
- github-pr-review: covered=[]; missing=['ocr-and-documents', 'pdf', 'quiz-bank-ocr-audit', 'document-to-action-items', 'nano-pdf']; purpose_compatible=False.
- python-failure-investigation: covered=[]; missing=['ocr-and-documents', 'pdf', 'quiz-bank-ocr-audit', 'document-to-action-items', 'nano-pdf']; purpose_compatible=False.

Decision reason: 无 purpose-compatible 的合格 bundle，返回 CREATE。

### AT-07/S2 → CREATE

Need: Investigate the deterministic assertion failure using a tight red/green feedback loop to establish the root cause before proposing a fix.

| Rank | Skill ID | RRF score | Retrieval reason |
|---|---|---:|---|
| 1 | systematic-debugging | 0.032787 | matched_terms: investigate, the, failure, a, tight, loop, to, root, cause, before; semantic_similarity: 0.5901 |
| 2 | github-issue-to-pr | 0.032002 | matched_terms: the, a, green, to, root, cause; semantic_similarity: 0.4815 |
| 3 | test-driven-development | 0.032002 | matched_terms: the, failure, a, green, to, before; semantic_similarity: 0.5638 |
| 4 | requesting-code-review | 0.030769 | matched_terms: a, loop, before; semantic_similarity: 0.4555 |
| 5 | github-pr-workflow | 0.030331 | matched_terms: the, failure, a, loop, to; semantic_similarity: 0.4443 |

Existing bundle comparison:
- github-pr-review: covered=['github-issue-to-pr', 'requesting-code-review', 'github-pr-workflow']; missing=['systematic-debugging', 'test-driven-development']; purpose_compatible=False.
- python-failure-investigation: covered=['systematic-debugging', 'requesting-code-review', 'github-pr-workflow']; missing=['github-issue-to-pr', 'test-driven-development']; purpose_compatible=False.

Decision reason: 无 purpose-compatible 的合格 bundle，返回 CREATE。

## DIRECT 与决策语义审计

- arXiv 查询从 k=5 改成 k=1：CREATE → DIRECT，返回 arxiv；唯一变化是候选截断。
- 更强的反例：扫描 PDF → PowerPoint 的多能力请求在 k=1 返回 DIRECT(ocr-and-documents)，并没有检查 PowerPoint 能力是否覆盖。
- 因此 DIRECT 确实混淆候选数量与 capability sufficiency；warnings 虽披露限制，但没有阻止错误语义。按用户要求未重设计该规则。
- REUSE/EXTEND 的集合运算符合当前代码规则，但把 top-k pool 当作必需 composition，真实查询中的噪声/替代项会阻断预期决策。
- purpose 必须逐字规范化相等：缓存 PR 查询已覆盖 4/5，仍因正常措辞差异被拒绝。这不是 wiring 故障，而是当前规则的语义限制。
- CREATE 在不相关请求上成立；但不应把前三项误落 CREATE 解释为系统正确识别了新组合。

## 集成缺陷与修改

未发现 ID、表示投影、索引、Dense/RRF 接线或 registry 传递错误；没有修复或重构产品代码。发现的是现有 composition/sufficiency 决策假设在真实查询上的缺陷。本次仅增加 fixture、可复跑 runner、真实本地集成测试和本报告。

建议在后续获授权的决策语义工作中解决候选选择/充分性与 purpose compatibility，再重新验收；不能仅调整 k 或迎合输出构造 bundle 来宣称通过。

## 复现与结果

```bash
PYTHONPATH=src .venv/bin/python scripts/capability_loading_real_e2e.py
PYTHONPATH=src .venv/bin/pytest -q tests/test_capability_loading_real_e2e.py tests/test_capability_loading.py
PYTHONPATH=src .venv/bin/pytest -q
git diff --check
```

本地真实测试使用现有资产，缺少本地资产时显式 skip；测试通过意味着复现当前行为及检索一致性，不意味着失败的语义 acceptance 被改判通过。完整原始 trace、每条 representation、source_scores、bundle matches、warnings 和输入 SHA-256：`local_artifacts/v0.7/real-e2e/report.json`。

最终结果：targeted **18 passed**；全量 **143 passed in 1.36s**；`git diff --check` 通过。原始输入 SHA-256 复核一致。
