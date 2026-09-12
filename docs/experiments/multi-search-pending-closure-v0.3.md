# Native schema / multi-search pending closure v0.3

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


分支：`feature/bundle-behavior-experiment-v0.3`。未 commit / push。

## 实现边界

- 当前分支 native schema 原本已经 required `action / skill_ids / reason`，enum 为 `DIRECT / EXTEND / CREATE`；本轮增加契约回归测试。`target_bundle_id / purpose` 保持 optional，action-specific 校验继续复用现有 Harness validator。
- Harness 按 skill_id 合并同 turn 未提交 candidates 和 representations；只在搜索成功后更新 pool。每次 tool result 仍为本次原始搜索结果。
- apply 成功清池，validation 失败不改 state/pool；普通 final 清池，每次 run 新 user turn 开始清池。成功 apply 后重新 search 不含已消费候选。
- 增加逐步及逐工具调用的 pending_pool before/after trace；未修改 retrieval、Bundle surface、Skill body 注入、DIRECT lifecycle、validate_decision/apply_decision；未接 OpenPI，未加入自动 sufficiency 判断。

## 本地验证

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_experimental_agent.py tests/test_bundle_behavior.py tests/test_runtime_capability_loading.py -q
# 75 passed
PYTHONPATH=src .venv/bin/python -m pytest -q
# 225 passed
git diff --check
# passed, no output
```

初次 pytest 未设置 PYTHONPATH，collection 报包不可导入；补上仓库 src 路径后上述 targeted/full 全通过。

## 真实 multi-search

[完整 live trace](/home/lixianbao/workspace/skill-control-plane/local_artifacts/experimental-skill-agent/live-t2cm0x_z/trace.json)。模型 `GLM-5.3-flash`，87 Skills，真实 BM25 + Dense + RRF，无 Dense fallback。

```bash
set -a
source .env
set +a
PYTHONPATH=src .venv/bin/python scripts/experimental_skill_agent_e2e.py --multi-search
```

任务为扫描 PDF OCR/提取校验 + GitHub Python CI 间歇性失败排查。测试 user task 明确要求先后搜索两种能力；同一个真实模型决定查询和 skill_ids，没有注入候选/脚本化工具响应。核验在运行后进行，不干预 runtime。

| Step | 动作 | Pool before → after |
|---|---|---|
| 1 | load_capability | 0 → 10 |
| 2 | load_capability | 10 → 20 |
| 3 | apply_capability | 20 → 0 |
| 4 | final content | 0 → 0 |

搜索 1：`从扫描版 PDF 中提取文字与表格并验证 OCR 识别质量的操作说明`

本次候选：`ocr-and-documents`, `quiz-bank-ocr-audit`, `pdf`, `document-to-action-items`, `nano-pdf`, `arxiv`, `xlsx`, `powerpoint`, `docx`, `windows-system-maintenance`。

搜索 2：`排查 GitHub Python 项目中仅在 CI 出现的间歇性测试失败：读取 PR 差异与 CI 日志、本地复现、根因定位与修复验证`

本次候选：`github-issue-to-pr`, `github-pr-workflow`, `github-issues`, `github-code-review`, `github-repo-management`, `python-debugpy`, `systematic-debugging`, `github-auth`, `claude-code`, `codex`。

成功 apply：`DIRECT`，选择 `ocr-and-documents`, `systematic-debugging`, `github-pr-workflow`, `python-debugpy`。

第一批独有的 `ocr-and-documents` 与第二批独有的 `systematic-debugging / github-pr-workflow / python-debugpy` 联合提交成功。两次 tool result 各仅包含本次 10 个候选；apply 前 pool 含完整 20 个候选，成功后清空。

`multi_search_audit` 全部检查通过；context audit 通过。无协议/工具错误，无候选丢失。

## 真实原始 4-turn 重跑

[完整 live trace](/home/lixianbao/workspace/skill-control-plane/local_artifacts/bundle-behavior/live-ca02q6gu/trace.json)。原始四条 TASKS 未修改，同一 agent/client/history。

```bash
PYTHONPATH=src .venv/bin/python scripts/bundle_behavior_e2e.py
```

四轮均 completed，context_audit_status=passed；matches_all_expectations=false。保留此次行为偏差，不重试筛选结果。

| Turn | 真实行为 | Pool / 结果 |
|---|---|---|
| 1 | 两次 search；CREATE 缺 reason 被拒；同一模型改 DIRECT | 合并后 14 → 拒绝仍 14 → 成功 0；加载 OCR/PDF/PowerPoint，未创建 Bundle |
| 2 | 无检索，普通回答 | 0 → 0，state 不变 |
| 3 | search → DIRECT xlsx | 0 → 10 → 0；没有文档 Bundle 可 EXTEND |
| 4 | search → DIRECT GitHub PR/code review | 0 → 10 → 0 |

仍观察到一次模型协议/参数错误：native schema 已 required reason，但 provider 返回的 CREATE 调用缺少 reason。Harness validator 拒绝并保留 state/pool，同一 LLM 修正后成功；没有 history 工具配对错误，也没有候选丢失。该行为不能报告为四轮 Bundle CREATE/EXTEND 预期通过。
