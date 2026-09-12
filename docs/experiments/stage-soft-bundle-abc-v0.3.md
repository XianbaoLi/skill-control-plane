# Finding

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


本轮固定预算下 **Hard → Soft 不值得替换基线；Soft → Soft+Repair 也未显示收益**：候选召回从 70.83% 降至 45.83%，repair 触发 75% 却未恢复任何 required Skill。

实验日期：2026-09-09。分支 `feature/soft-bundle-retrieval-v0.3`，基线 checkpoint `d0ff3f1` 保持可回退；未 merge main。

**固定配置（在运行新实验之前确定，未 Gold sweep）**：A 为 K=5/BM25+Dense、threshold=0.35、margin=0.05；三臂 `max_bundles=4`、`max_skills_per_bundle=4`。B/C global K=3/检索器；最多 2 个 prior；local K=2/检索器/prior，多 prior 轮流取候选，共享最多 4 个 local 候选。首轮上限 10，C 最多一次 repair，union 后上限 20。选择依据仅为旧 A 实际池规模 5–10，没有用 required labels 选参数。

**数据来源限制**：使用真实 `/mnt/d/Hermes/skills` 87 Skills、原 Gold、manifest 和离线 `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`。下文 C 使用上一轮真实 Codex completion 的完整 prompt SHA-256 精确重放；没有为此次实验生成新 wording。新在线尝试因 sandbox 外 Codex 状态目录只读而失败；后续在线重跑被自动审批拒绝。重放没有新的外部模型调用，不能将逻辑调用计数宣称成本轮实际模型账单。失败尝试另存，未混入主比较。

## A / B / C definition

```text
A: raw evidence → 原 Dense Bundle matcher
                  ├─ confident → Bundle-local BM25 + Dense
                  └─ otherwise → global BM25 + Dense
B: raw evidence → 显式 prior(s) → local BM25 + Dense ─┐
   raw evidence ───────────────→ global BM25 + Dense ├→ union/dedup
C: B first pass → cheap uncertainty trigger?         │
                  ├─ no → first pool                │
                  └─ yes → existing extractor → B(capability_need)
                            → first ∪ repair（保留首轮顺序与全部候选）
```

S1 三臂均执行旧初始检索与 Shelf 构建；后续继续使用原 `integrate_retrieval_delta` 和 activation policy。B 无 extractor；C 不调用额外 judge，无 repair hint。Extractor 输入只有 initial_task 与 runtime_evidence。Gold、case ID、expected Skill 均不进入 runtime routing 或 prompt。

显式规则按 github、software-development、email、productivity 的固定顺序检查；仅保留 corpus 已存在 group。规则分别匹配 pull request / 带编号 PR / review diff / github；pytest / traceback / breakpoint / debugger / stack trace；email / inbox / sender；calendar / meeting invite / invitation。大小写不敏感、词边界匹配。`CI failed` 本身不命中任何 prior。多组可共存，最多两个；未命中仍执行 global。规则不把裸 repository、issue、event 等模糊词视为充分条件。

local pool 先按 prior 轮流取各自 BM25-first union，再应用共享 4 个预算；最终 local-first 加 global union。没有修改原 BM25、Dense 或 candidate_union。此顺序是固定机械顺序，不是语义排序，可能影响现有 activation。

Repair trigger 固定为任一：global Dense top score < 0.35；global BM25/Dense top-3 没有交集；首轮 pool < 2。这里的 0.35 是新检索充分性启发式，与原 Hard matcher 参数独立。Extractor 继续原 confidence >=0.5/unknown 验证规则。异常、空结果或低置信度保留首轮；第二次检索异常也保留首轮。无 Gold-based 充分性判断。

## Frozen-stage results

每个 runtime stage 使用 **A sequential 同一 pre-stage Shelf/Active**，包括全部 Bundle 字段与成员；该值对象不可变，B/C 结果不写回 A 或其他 frozen stage。表中 recall 均针对 `required_now`，包括先前已有能力；候选 recall 不等于 Shelf recall。

| Stage | Arm | Pool | Candidate recall | Shelf recall | Active recall | Repair |
| --- | --- | --- | --- | --- | --- | --- |
| ST-03/S2 | A | 8 | 100.00% | 0.00% | 0.00% | — |
| ST-01/S2 | A | 5 | 50.00% | 50.00% | 50.00% | — |
| ST-01/S3 | A | 10 | 33.33% | 66.67% | 33.33% | — |
| ST-01/S4 | A | 5 | 100.00% | 100.00% | 100.00% | — |
| ST-03/S2 | B | 7 | 100.00% | 100.00% | 0.00% | not-triggered |
| ST-01/S2 | B | 7 | 50.00% | 50.00% | 50.00% | not-triggered |
| ST-01/S3 | B | 9 | 33.33% | 66.67% | 66.67% | not-triggered |
| ST-01/S4 | B | 4 | 0.00% | 0.00% | 0.00% | not-triggered |
| ST-03/S2 | C | 8 | 100.00% | 100.00% | 0.00% | ok |
| ST-01/S2 | C | 9 | 50.00% | 50.00% | 50.00% | ok |
| ST-01/S3 | C | 10 | 33.33% | 66.67% | 66.67% | ok |
| ST-01/S4 | C | 4 | 0.00% | 0.00% | 0.00% | not-triggered |

### ST-01/S2 — cross-domain

A 搜索 GitHub local，python-debugpy 与 systematic-debugging 都不在候选。B 命中 software-development prior，python-debugpy 通过 local lane 进入（不能把这一恢复归功于 global lane）；global 始终存在，保证域外探索机会，但机会不保证召回。systematic-debugging 首轮仍 miss，C rewrite 后仍 miss。

### ST-01/S3 — debugger

B/C 首轮包含 python-debugpy 并使 software-development 活跃；A 虽有 python-debugpy 候选，但最佳 delta 属于其他组，仍 inactive。systematic-debugging 在全部 arm 仍 miss。Frozen 中 A 的 pre-stage 尚无 software-development，故归 Failure-A；sequential B/C 已在 S2 建立该 Bundle，故 S3 同一 missing Skill 归 Failure-C。

### ST-01/S4 — within-domain review

A 的 5 个候选含 github-code-review。B/C 命中 github prior，但 local K=2 两个检索器的结果只有 github-pr-workflow、github-issue-to-pr；global top-3 也未提供 github-code-review。故失去原正例，不能称为成功方案。首轮两检索器有交集且 Dense top score 足够高，C 不触发 repair：这只是没有付 rewrite 成本，不代表充分性判定正确。预算分配与 trigger 均存在不足；没有针对该失败调整 K 或规则。

### ST-03/S2 — calendar invitation

email 与 productivity prior 共存。google-workspace 在 B 首轮通过 local lane 进入 Shelf，但 email 组最佳，仍未激活 productivity。A 的 Failure-B 变成 B/C 的 Failure-D；未解决 activation 问题，也未为其调整策略。

## Sequential E2E results

原 Gold 含两个 trajectory：ST-01 为 S1→S2→S3→S4，ST-03 为 S1→S2。下列均值只含 4 个 runtime transitions，S1 排除；final counts 按两个 case 平均。

| Metric | A | B | C |
| --- | --- | --- | --- |
| required_skill_candidate_recall | 0.7083 | 0.4583 | 0.4583 |
| required_skill_shelf_recall | 0.5417 | 0.5417 | 0.5417 |
| required_skill_active_recall | 0.4583 | 0.2917 | 0.2917 |
| mean_candidate_pool_size | 7.0000 | 6.7500 | 7.7500 |
| candidate_recall_per_candidate | 0.1012 | 0.0679 | 0.0591 |
| mean_registered_surface_growth | 2.7500 | 3.5000 | 3.7500 |
| mean_active_surface_growth | 1.5000 | 1.7500 | 1.7500 |
| mean_final_registered_skill_count | 10.0000 | 11.5000 | 12.0000 |
| mean_final_active_skill_count | 5.5000 | 6.0000 | 6.0000 |

B 在 S2 就把 debugging group 放入 Shelf/Active，影响 S3 的前置状态和 failure 分类。C 在先前阶段追加候选，最终 registered count 从 B 的 11.5 增到 12。由于当前 soft routing 不读取 Shelf，frozen 与 sequential 的 B/C 候选相同；Shelf/Active 状态与增量仍不同。因此这种相同并不是漏跑了 frozen。两类实验的三项 recall 此次碰巧相等，不能泛化到其他轨迹。

## Cost / Effectiveness

unique search corpus size 指一个 transition 覆盖的唯一 Skill 数；record visits 按每 pass 的 global size + local subset sizes 累加，同一记录可重复计数，未将 BM25/Dense 再乘二。A 的 descriptor matcher 成本另存在完整旧 trace。这些仅为搜索/暴露面 proxy，不是 latency 或真实 token savings。

| Metric | A | B | C |
| --- | --- | --- | --- |
| mean_search_corpus_size | 47.0000 | 87.0000 | 87.0000 |
| mean_search_record_visits | 47.0000 | 99.2500 | 169.5000 |
| prior_rule_hit_rate | 0.0000 | 1.0000 | 1.0000 |
| mean_prior_bundle_count | 0.0000 | 1.2500 | 1.2500 |
| local_retrieval_rate | 0.5000 | 1.0000 | 1.0000 |
| global_lane_rate | 0.5000 | 1.0000 | 1.0000 |
| local_candidate_count | 2.5000 | 3.2500 | 3.2500 |
| global_candidate_count | 4.5000 | 5.2500 | 5.2500 |
| repair_trigger_rate | 0.0000 | 0.0000 | 0.7500 |
| rewrite_rate | 0.0000 | 0.0000 | 0.7500 |
| mean_retrieval_passes | 1.0000 | 1.0000 | 1.7500 |
| repair_success_rate | 0.0000 | 0.0000 | 0.0000 |
| extra_llm_calls_per_transition | 0.0000 | 0.0000 | 0.7500 |

local/global candidate count 与 prior rate 是首轮统计，各 repair pass 独立 trace 保留实际 lane、priors、计数、score、overlap 与 candidate IDs；最终 pool size 为两轮 dedup 后的数量。B/C 所有实际 pass 都执行了 global lane。

B 比 A 平均候选少 0.25（7→6.75），但唯一搜索 corpus 增 40（47→87）；record visits 增 52.25（47→99.25，约 +111.17%）。candidate recall 下降 25 个百分点，Shelf recall 无变化，Active recall 下降 16.67 个百分点。不能因为局部 debugging 正例就宣称整体改善。

C 比 B 平均候选增加 1（6.75→7.75），record visits 增 70.25（99.25→169.5），required recall 未变。3/4 transition 触发 rewrite，每 transition 逻辑 extractor 调用 0.75；成功完成第二次检索 3 次，平均 passes=1.75。repair_success_rate 定义为恢复至少一个首轮未命中 required Skill 的触发数 / 全部触发数，此次 0/3。`required_skill_recovered_by_repair=[]`。

candidate_recall_per_candidate 定义为 macro required_skill_candidate_recall / mean_candidate_pool_size，仅为简单 efficiency proxy，不是正式 statistical utility。A/B/C 为 0.10119 / 0.06790 / 0.05914。

重放中没有实际新模型 token 使用；无可可靠归属于本次在线成功调用的 token audit，因此不估算 llm_input_tokens/llm_output_tokens。Frozen 与 sequential 分别逻辑调用 3 次；相同输入复用历史 completion，不能当作 6 个独立模型样本。

| Transition | Capability need（历史真实 completion） | 新增候选 | 恢复 required |
| --- | --- | --- | --- |
| ST-03/S2 | Create a calendar event for tomorrow at 3 PM and send an invitation to the email sender. | pdf | 无 |
| ST-01/S2 | Diagnose and fix a reproducible Python test failure caused by KeyError: 'token' at src/auth.py:87. | session-librarian, quiz-bank-ocr-audit | 无 |
| ST-01/S3 | Step through build_session() during the failing test and watch the session dictionary to identify when the token field disappears. | spike | 无 |
| ST-01/S4 | — | — | 无 |

## Failure analysis

沿用四类，依序判断：required 已注册但 inactive → Failure-D；在 candidate pool 但未注册 → Failure-B；未召回且 pre-stage 已知其 Bundle → Failure-C；其他 retrieval miss → Failure-A。这里 taxonomy 的预算截断指 candidate→Shelf 层；较小 retrieval K 的漏召回仍按 Failure-A/C 分类，不另加类别。

| Mode | Arm | Stage | Required Skill | Failure |
| --- | --- | --- | --- | --- |
| frozen | A | ST-03/S2 | google-workspace | Failure-B |
| frozen | A | ST-01/S2 | systematic-debugging | Failure-A |
| frozen | A | ST-01/S3 | python-debugpy | Failure-D |
| frozen | A | ST-01/S3 | systematic-debugging | Failure-A |
| frozen | B | ST-03/S2 | google-workspace | Failure-D |
| frozen | B | ST-01/S2 | systematic-debugging | Failure-A |
| frozen | B | ST-01/S3 | systematic-debugging | Failure-A |
| frozen | B | ST-01/S4 | github-code-review | Failure-C |
| frozen | C | ST-03/S2 | google-workspace | Failure-D |
| frozen | C | ST-01/S2 | systematic-debugging | Failure-A |
| frozen | C | ST-01/S3 | systematic-debugging | Failure-A |
| frozen | C | ST-01/S4 | github-code-review | Failure-C |
| sequential | A | ST-03/S2 | google-workspace | Failure-B |
| sequential | A | ST-01/S2 | systematic-debugging | Failure-A |
| sequential | A | ST-01/S3 | python-debugpy | Failure-D |
| sequential | A | ST-01/S3 | systematic-debugging | Failure-A |
| sequential | B | ST-03/S2 | google-workspace | Failure-D |
| sequential | B | ST-01/S2 | systematic-debugging | Failure-A |
| sequential | B | ST-01/S3 | systematic-debugging | Failure-C |
| sequential | B | ST-01/S4 | github-code-review | Failure-C |
| sequential | C | ST-03/S2 | google-workspace | Failure-D |
| sequential | C | ST-01/S2 | systematic-debugging | Failure-A |
| sequential | C | ST-01/S3 | systematic-debugging | Failure-C |
| sequential | C | ST-01/S4 | github-code-review | Failure-C |

systematic-debugging 在 S2/S3 首轮和 repair 后均未恢复。S4 github-code-review 在 B/C 发生新增 miss。cheap trigger 在 S4 错把一致性当成充分性；修复增加了无 required 收益的候选。结论局限于两条小样本轨迹、一次固定预算、历史固定 rewrite 样本，既不证明所有 soft prior 无效，也不支持采用本实现替换 baseline。

## Reproduce and artifacts

```bash
cd ~/workspace/skill-control-plane
export PATH=/home/lixianbao/miniforge3/envs/skill-control-plane/bin:$PATH
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
skill-control-plane eval stage-bundle /mnt/d/Hermes/skills \
  --gold evals/gold/stage-transition-v0.2.jsonl \
  --manifest local_artifacts/corpora/hermes-local-v0.1/manifest.json \
  --dense-model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --k 5 --max-bundles 4 --max-skills-per-bundle 4 \
  --soft-bundle-repair-ab \
  --capability-need-command 'python scripts/capability_need_codex.py --replay-dir local_artifacts/experiments/capability-need-local87-extractions' \
  --json
```

仅 A/B 使用 `--soft-bundle-ab` 并移除 command。两个新 flag 与已有 hierarchical/capability-need flag 互斥，均 opt-in；JSON 同时输出 frozen、sequential、完整 baseline_A。新在线调用需要先解除当前审批阻塞。

本地 ignored artifacts：

- `local_artifacts/experiments/stage-soft-bundle-abc-v0.3-replay.json`：主比较原始结果。
- `local_artifacts/experiments/soft-v0.3-abc-recheck.json`：最终代码的精确重放复核，含基础参数。
- `local_artifacts/experiments/stage-soft-bundle-abc-v0.3-live.json`：在线 transport 失败结果，C 安全保留 B；不是有效 repair 效果实验。
- `local_artifacts/experiments/soft-bundle-v0.3-extractions/`：失败 provider audit。
- `local_artifacts/experiments/capability-need-local87-extractions/`：历史真实 prompt/completion/provider audit，精确 SHA-256 校验重放。
- `local_artifacts/experiments/soft-v0.3-*-recheck.json` 与 `soft-v0.3-verification.json`：旧 CLI 与新 A/B/ABC identity 验证。

## Validation and acceptance status

`pytest -q`：85 passed（原 71 + 新 14）。新增覆盖 explicit 多 prior/no match/不读取 labels、global lane、去重与共享预算、easy case 零 rewrite、异常 fallback、两轮 union、trigger/恢复率分母、完整 A equality、Gold label 扰动不改变 routing/extractor 输入、frozen state 与 sequential 隔离、新 CLI opt-in 与互斥。原测试继续覆盖旧 CLI、extractor parsing/低置信度/provider 安全边界。

Arm A 完整历史 JSON 保留并对比，不仅比较 recall。旧默认、hierarchical、capability-need CLI 完整输出另行复核。所有禁止修改项保持原样，checkpoint 与 main 不变。

实现、测试、frozen/sequence 精确重放与报告已完成。**研究成功条件未全部达成**：B 未保留 S4 正例，C 未恢复 systematic-debugging；没有用调参掩盖负结果。**新的在线 ABC 尚未完成**：自动审批拒绝外部模型 evidence 发送及工作区外运行状态写入，需要明确批准后再执行；不能把历史 completion 重放冒充新在线实验。
