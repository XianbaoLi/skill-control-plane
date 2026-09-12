# Pi / OpenPI Bundle 动态更新：本地 runtime 验证

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


日期：2026-09-09。范围仅限 Bundle Update；不实现 Trigger、Reroute 或通用控制器。

**结论：可以在同一 AgentSession 的轮次之间替换 Skill 集合，不需要修改 Pi core 或 OpenPI 源码。** 本次采用 SDK adapter 的 `skillsOverride` + `session.reload()`。这是同一会话的资源/扩展重载，不是所有扩展状态无损的热替换。

## 运行资产与证据边界

- 用户指定全局 Pi：`/home/lixianbao/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent`，版本 **0.85.0**。直接阅读其 `dist/core` 实现和声明。
- OpenPI：`/home/lixianbao/workspace/openpi`，HEAD `72fbba52832841cfc78c2f7e7947eeb89efc73bc`，checkout 无修改；实际依赖 Pi **0.84.1**。
- **执行通过的是 OpenPI 所依赖的 0.84.1**，包括纯 Pi 与加载真实 OpenPI extensions 两种环境。全局 0.85.0 的 SDK barrel 导入报 `ERR_MODULE_NOT_FOUND: @earendil-works/pi-server`。直接导入 core 也因 extension loader 回引 barrel 失败。未修补安装包、未替换依赖、未将 0.84.1 通过冒充为 0.85.0 通过。
- 普通 `pi list` 无法在只读的用户配置目录创建 lock，故不能据其输出判断用户全局安装状态。另建临时 agentDir，只配置一个 OpenPI 绝对路径，执行全局 CLI 的 `pi list` 成功，唯一来源为上述 checkout。测试也使用此绝对路径，断言成功加载 checkout 扩展。
- 真实 `AgentSession`、资源加载器、原生工具、OpenPI extensions；仅 provider 用 Pi 官方 `fauxProvider` 确定性响应，捕获 provider 实际收到的 context。没有远程模型调用，不声称验证模型选 Skill 的效果。

## 最小调用路径

```js
const bundle = createBundleResources(pi, {
  cwd, agentDir, settingsManager,
  additionalSkillPaths: [A, B],
}); // 内部强制 noSkills: true
await bundle.loader.reload();
const { session } = await pi.createAgentSession({
  cwd, agentDir, settingsManager, modelRuntime, model,
  resourceLoader: bundle.loader,
  sessionManager,
});
await session.bindExtensions({ mode: 'print', onError });
await session.prompt('turn 1');
await bundle.update(session, [A, C]);
await session.prompt('turn 2');
```

adapter 保存当前显式路径快照；`skillsOverride` 每次通过 Pi 原生 `loadSkills({ includeDefaults: false, skillPaths: selected })` 返回完整集合。它既能加 C，也能排除 B；扩展 discovery 之后再次应用 override，避免额外 Skill 混入。Skill 文件路径仍指向原库，不复制或改写正文、相对资源根目录或执行语义。

当前实现还有更短的可行路径：构造 loader 时保留传入的 `additionalSkillPaths` 数组，原地 `splice` 成 `[A,C]` 后调用 `session.reload()`。已在纯 Pi 测试中通过，但数组引用别名只是当前实现行为，不是公开的 set-paths 契约，adapter 不依赖它。直接赋值 Pi 的私有 `additionalSkillPaths` 没有必要。

## 源码调用链

以下行号对应全局 **0.85.0**；0.84.1 的对应路径也已阅读并运行。

1. `dist/core/resource-loader.js:166`：构造器保存 `additionalSkillPaths`；没有公开替换 setter。
2. `resource-loader.js:263`：`reload()` 重读设置、加载扩展、计算 Skill paths；`noSkills` 排除普通发现来源，仍保留显式 CLI / additional paths。
3. `resource-loader.js:500`：`updateSkillsFromPaths()` → `loadSkills(includeDefaults:false)` → `skillsOverride` → 写入 skills / diagnostics；`getSkills()` 返回当前加载结果。
4. `dist/core/agent-session.js:2217`：`session.reload()` → old runner `session_shutdown(reason:reload)` → invalidate → settings reload → resetApiProviders → resourceLoader.reload → `_buildRuntime()`。
5. `_buildRuntime()` → `_refreshToolRegistry()` → `setActiveToolsByName()` → `_rebuildSystemPrompt()`；`agent-session.js:755` 从当前 loader `getSkills()` 取 metadata；`dist/core/system-prompt.js` → `formatSkillsForPrompt()`，有 read 或 bash 时添加 metadata，禁用 model invocation 的 Skill 不进入列表。
6. 已绑定的 session 发出 `session_start(reason:reload)` → `extendResourcesFromExtensions('reload')` → `resources_discover`；若有返回路径则 `extendResources()`，并再次重建 system prompt。
7. 下一次 `prompt()` 用新的 base system prompt 进入 `before_agent_start`，允许扩展覆盖，然后交给 agent/provider。测试捕获的是这个最终 context，而非仅检查 loader。

**loader.reload 本身不通知 session**。反例断言 loader 已是 A/C，而 session.systemPrompt 仍等于旧值。因此不能只更新 registry 就宣告 context 已刷新。单独 reload loader 还会加载尚未绑定的新扩展；探索时在 OpenPI 中观察到未初始化的事件回调错误，最终受支持 adapter 路径不采用此操作。

## slash commands 与 extension API

- `agent-session.js:989` 的 Skill slash expansion 每次查询当前 `resourceLoader.getSkills()`；正文按调用时读取。
- `agent-session.js:1989` 的 extension `pi.getCommands()` 动态从 loader 生成 `skill:<name>` 命令。测试验证重载后为 A/C，B 命令不再展开，C 命令可读取真实正文。
- TUI 自动补全另有缓存：`dist/modes/interactive/interactive-mode.js:5015` 的标准 reload handler 调用 session.reload，之后调用 `setupAutocompleteProvider()`。裸 SDK session.reload 不拥有 TUI 的补全控件；本次未做交互界面验收。
- Pi `ExtensionCommandContext.reload()` 可触发标准重载；普通 tool/event 的 `ExtensionContext` 没有 reload 方法。reload 后旧 command ctx / pi 失效，不能继续拿旧句柄操作。
- 扩展可监听 `session_shutdown`、`session_start`（reason=reload）和 `resources_discover`。后者返回 `skillPaths` 等；它是资源发现 hook，不是任意调用的 replace hook。`extendResources` 本身只有合并，没有 retire。
- 纯 extension 方案可以在 reload 后的 discovery 返回一个持久 manifest 的当前 paths，但必须先去掉固定的旧 `[A,B]` 基线，否则 B 仍在基础集合。该方案本次仅源代码分析，未冒充为已执行路径；SDK adapter 已提供被执行验证的精确替换方案。
- OpenPI 自身没有独立 Skill registry；`extensions/shared/child-session.ts:568` 的 `createChildResources()` 使用 Pi DefaultResourceLoader，是未来 child adapter 的自然接入点。父 session controller 应接在持有 session/loader 的宿主 adapter；不要从扩展工具的普通 ctx 强转拿 reload。

## 可重复测试

在 skill-control-plane 根目录运行：

```bash
node --test scripts/pi_bundle_runtime/hot-bundle.test.mjs
```

默认明确使用 OpenPI checkout 的 Pi 依赖，并打印其路径和版本。可用 `OPENPI_ROOT`、`PI_CORE_ROOT` 指定其他已有安装，不自动降级或下载依赖。

保留 provider context 证据：

```bash
BUNDLE_EVIDENCE_DIR=local_artifacts/pi-bundle-runtime \
  node --test scripts/pi_bundle_runtime/hot-bundle.test.mjs
```

复现全局 SDK 导入问题：

```bash
PI_CORE_ROOT=/home/lixianbao/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent \
  node --test scripts/pi_bundle_runtime/hot-bundle.test.mjs
```

真实 Skill：

| 标记 | hermes-current87 下的路径 |
|---|---|
| A | software-development/python-debugpy/SKILL.md |
| B | software-development/node-inspect-debugger/SKILL.md |
| C | devops/sdlc-review/SKILL.md |

三个测试全部通过：纯 Pi adapter、纯 Pi 原始数组机制及 loader-only 反例、Pi + OpenPI adapter。每个案例均验证：

- 初始 getSkills 精确为 A/B；真实第一轮 provider 收到 A/B metadata。
- 更新后 getSkills 精确为 A/C；同一 session ID，消息历史逐项不变。
- 第二轮 provider context 含 A/C name 与 description；无 B metadata name、description 或 filePath。
- extension command registry 是 A/C；B slash 不再展开，C slash 原生展开。
- shutdown/start/discovery 的 reload 生命周期发生，provider 未发生错误，extension errors 为空。
- 调用 C 后再退休 C，其 metadata 消失，但历史中的 C Skill block 仍保留。
- 三个源文件 SHA-256 前后一致。

本地证据位于 `local_artifacts/pi-bundle-runtime/`：`results.tap`、`provenance.json`、`context-*.json`、`global-sdk-failure.tap`。context 文件保存前两轮完整 system prompt 和消息；测试后临时 session/agentDir 删除。

## 最小改动与下一步

新增 `scripts/pi_bundle_runtime/adapter.mjs`、`scripts/pi_bundle_runtime/hot-bundle.test.mjs` 和本文。没有修改 OpenPI、Pi core、hermes-current87 或现有 Python 业务代码。

Bundle Controller 下一步接 **宿主 SDK / OpenPI adapter 层**：维护当前 Bundle paths，在 turn boundary 调用 update，并串行化 prompt/steer/followUp 与更新。当前 adapter 拒绝 streaming / compacting 状态；不是跨并发请求的锁，也不是失败可回滚事务。reload 失败应阻止下一轮并恢复，不能假装更新成功。此范围没有实现通用控制器或并发事务。

两个实质限制：

1. **retire 是从当前 discovery / metadata 移除，不是抹去历史，也不是文件访问权限撤销。** 对先前已 read / slash 展开的 B，不能保证完整后续消息 context 完全没有 B。原生 read 仍能访问已知路径。本次指定的两轮 metadata 测试通过；更强的“遗忘/撤权”不由 Bundle reload 提供。
2. **session.reload 会重载全部扩展。** OpenPI 的后台终端在 shutdown 时终止，subagent/workflow 等扩展也执行生命周期清理；不能承诺运行中后台任务和扩展内存无损。若之后要求高频更新并保留这些状态，才需要考虑 Pi 增加“只刷新 Skill registry + base system prompt + UI resources changed 通知”的窄 hook；当前 MVP 不需要 core patch。
