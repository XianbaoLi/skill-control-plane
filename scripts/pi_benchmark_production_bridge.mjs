/** Production Pi benchmark bridge. The configured model autonomously performs tasks. */
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const repo = process.cwd();
const envFile = path.join(repo, '.env');
if (existsSync(envFile)) {
  for (const line of (await readFile(envFile, 'utf8')).split(/\r?\n/)) {
    const match = line.match(/^([A-Z][A-Z0-9_]*)=(.*)$/);
    if (match && /^(BIGMODEL_|PARATERA_)/.test(match[1]) && process.env[match[1]] === undefined) {
      process.env[match[1]] = match[2].trim().replace(/^['"]|['"]$/g, '');
    }
  }
}
const required = name => {
  const value = process.env[name];
  if (!value) throw new Error(`missing ${name}`);
  return value;
};
const arm = required('BENCHMARK_ARM');
const corpus = required('BENCHMARK_CORPUS');
const prompt = required('BENCHMARK_PROMPT');
const workspace = required('BENCHMARK_WORKSPACE');
const tracePath = required('BENCHMARK_TRACE');
const providerId = required('BENCHMARK_PROVIDER');
const modelId = required('BENCHMARK_MODEL');
const turns = JSON.parse(process.env.BENCHMARK_TURNS || '[]');
const turnChecks = JSON.parse(process.env.BENCHMARK_TURN_CHECKS || '[]');
if (!['native', 'control-plane'].includes(arm)) throw new Error(`invalid arm: ${arm}`);

const openpi = process.env.OPENPI_ROOT ?? path.join(homedir(), 'workspace/openpi');
const piRoot = process.env.PI_CORE_ROOT ?? path.join(openpi, 'node_modules/@earendil-works/pi-coding-agent');
const pi = await import(pathToFileURL(path.join(piRoot, 'dist/index.js')));
const agentDir = process.env.PI_AGENT_DIR ?? path.join(homedir(), '.pi/agent');
const skillRoot = path.join(repo, 'local_artifacts/corpora/benchmark-corpus-v0.1', corpus, 'skills');
let artifacts = path.join(repo, 'local_artifacts/benchmark-v0.1', corpus);
let cards = path.join(artifacts, 'retrieval-cards-v0.1.jsonl');
let dense = path.join(artifacts, 'dense-index-v1.json');

const walkSkills = async directory => {
  const result = [];
  for (const category of await readdir(directory, { withFileTypes: true })) {
    if (!category.isDirectory()) continue;
    for (const skill of await readdir(path.join(directory, category.name), { withFileTypes: true })) {
      const candidate = path.join(directory, category.name, skill.name, 'SKILL.md');
      if (skill.isDirectory() && existsSync(candidate)) result.push(candidate);
    }
  }
  return result.sort();
};
const skillPaths = await walkSkills(skillRoot);
if (!existsSync(cards) || !existsSync(dense)) {
  const ids = new Set(skillPaths.map(file => path.basename(path.dirname(file))));
  const source = path.join(repo, 'local_artifacts/benchmark-v0.1/S128');
  artifacts = path.join(workspace, '.benchmark-artifacts', corpus);
  cards = path.join(artifacts, 'retrieval-cards-v0.1.jsonl');
  dense = path.join(artifacts, 'dense-index-v1.json');
  await mkdir(artifacts, { recursive: true });
  const cardLines = (await readFile(path.join(source, 'retrieval-cards-v0.1.jsonl'), 'utf8'))
    .split(/\r?\n/).filter(line => line.trim() && ids.has(JSON.parse(line).skill_id));
  await writeFile(cards, cardLines.join('\n') + '\n');
  const index = JSON.parse(await readFile(path.join(source, 'dense-index-v1.json'), 'utf8'));
  index.records = index.records.filter(record => ids.has(record.skill_id));
  await writeFile(dense, JSON.stringify(index) + '\n');
}
const embeddingTelemetry = path.join(workspace, '.embedding-telemetry.jsonl');
if (arm === 'control-plane') {
  process.env.SKILL_CONTROL_PLANE_PYTHON = path.join(repo, 'scripts/benchmark_production_sidecar.py');
  process.env.SKILL_CONTROL_PLANE_SKILL_ROOT = skillRoot;
  process.env.SKILL_CONTROL_PLANE_RETRIEVAL_CARDS = cards;
  process.env.SKILL_CONTROL_PLANE_DENSE_INDEX = dense;
  process.env.BENCHMARK_EMBEDDING_TELEMETRY = embeddingTelemetry;
  process.env.PYTHONPATH = path.join(repo, 'src');
}

const settingsManager = pi.SettingsManager.inMemory(
  { compaction: { enabled: false }, retry: { enabled: false } },
  { projectTrusted: true },
);
const modelRuntime = await pi.ModelRuntime.create({
  authPath: path.join(agentDir, 'auth.json'), modelsPath: path.join(agentDir, 'models.json'),
});
const model = modelRuntime.getModel(providerId, modelId);
if (!model) throw new Error(`configured Pi model not found: ${providerId}/${modelId}`);

const events = [];
const messages = [];
let eventSeq = 0;
let currentTurn = null;
const maxTurns = Number(process.env.BENCHMARK_MAX_TURNS || 20);
const reasoning = process.env.BENCHMARK_REASONING || undefined;
const record = event => events.push({ event_seq: ++eventSeq, turn_id: currentTurn, ...event });
const subscribe = session => session.subscribe(event => {
  if (event.type === 'tool_execution_start') {
    record({ type: 'tool_call', tool: event.toolName, args: event.args });
    if (event.toolName === 'load_capability') {
      record({ type: 'capability_search', query: event.args?.need });
    }
    if (event.toolName === 'apply_capability') {
      record({ type: 'capability_activation', skill_ids: event.args?.skill_ids ?? [],
        action: event.args?.action });
    }
    if (event.toolName === 'read' && String(event.args?.path).includes('/skills/')) {
      record({ type: 'capability_activation',
        skill_ids: [path.basename(path.dirname(event.args.path))], action: 'native-read' });
    }
  }
  if (event.type === 'tool_execution_end') {
    record({ type: 'tool_result', tool: event.toolName, is_error: event.isError });
    const rendered = JSON.stringify(event.result);
    for (const match of rendered.matchAll(/BENCHMARK_EVIDENCE:([A-Za-z0-9_-]+)/g)) {
      record({ type: 'evidence_emitted', event_id: match[1], marker: match[0] });
    }
  }
  if (event.type === 'message_end' && event.message.role === 'assistant') {
    messages.push({ turn_id: currentTurn, message: event.message });
    if (messages.length >= maxTurns) void session.abort();
  }
});

const makeLoader = async () => {
  if (arm === 'native') {
    const loader = new pi.DefaultResourceLoader({ cwd: workspace, agentDir, settingsManager,
      noSkills: false, additionalSkillPaths: skillPaths, noPromptTemplates: true });
    await loader.reload();
    return loader;
  }
  const extension = (await import(pathToFileURL(path.join(repo, 'adapters/pi/src/index.ts')))).default;
  const loader = new pi.DefaultResourceLoader({ cwd: workspace, agentDir, settingsManager,
    noSkills: true, noPromptTemplates: true, extensionFactories: [extension] });
  await loader.reload();
  return loader;
};
const createSession = async (manager, reason = 'startup', previousSessionFile) => {
  const loader = await makeLoader();
  const created = await pi.createAgentSession({ cwd: workspace, agentDir, model, modelRuntime,
    settingsManager, resourceLoader: loader, sessionManager: manager,
    ...(reasoning ? { thinkingLevel: reasoning } : {}),
    tools: arm === 'native'
      ? ['read', 'write', 'edit', 'bash']
      : ['read', 'write', 'edit', 'bash', 'load_capability', 'apply_capability', 'load_skill_body'],
    sessionStartEvent: { type: 'session_start', reason, previousSessionFile },
  });
  subscribe(created.session);
  await created.session.bindExtensions({ mode: 'print', onError: error => { throw error; } });
  return created.session;
};
const stop = async session => {
  await session.abort();
  await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'quit' });
  session.dispose();
};
const checkTurn = checks => checks.map(check => {
  if (check.type === 'command') {
    const argv = [...check.argv];
    if (argv[0] === 'python') argv[0] = process.env.PYTHON || 'python3';
    const result = spawnSync(argv[0], argv.slice(1), { cwd: workspace, encoding: 'utf8',
      env: { ...process.env, BENCHMARK_VERIFIER: process.env.BENCHMARK_VERIFIER || '' } });
    return { success: result.status === 0, detail: `${result.stdout || ''}${result.stderr || ''}`.slice(-1000) };
  }
  if (check.type === 'json_contract') {
    try {
      const value = JSON.parse(execFileSync(process.execPath, ['-e',
        `process.stdout.write(require('fs').readFileSync(${JSON.stringify(path.join(workspace, check.path))},'utf8'))`],
      { encoding: 'utf8' }));
      const success = Object.entries(check.fields).every(([key, rule]) => {
        const item = value[key];
        const typed = rule.type === 'array' ? Array.isArray(item)
          : rule.type === 'object' ? item && typeof item === 'object' && !Array.isArray(item)
          : typeof item === 'string';
        const text = JSON.stringify(item).toLowerCase();
        return typed && (!rule.nonempty || Boolean(item) && (!Array.isArray(item) || item.length)) &&
          (rule.contains || []).every(term => text.includes(term.toLowerCase()));
      });
      return { success, detail: 'semantic JSON turn check' };
    } catch (error) { return { success: false, detail: String(error) }; }
  }
  return { success: false, detail: `unsupported turn check ${check.type}` };
});

const sessionDir = path.join(workspace, 'sessions');
await mkdir(sessionDir, { recursive: true });
let manager = pi.SessionManager.create(workspace, sessionDir);
let session = await createSession(manager);
const turnMetrics = [];
if (turns.length) {
  currentTurn = turns[0].turn_id;
  const before = messages.length;
  await session.prompt(turns[0].prompt);
  turnMetrics.push({ turn_id: currentTurn, success_checks: checkTurn(turnChecks[0] || []),
    assistant_message_count: messages.length - before });
  const sessionFile = manager.getSessionFile();
  await stop(session);
  if (!sessionFile || !existsSync(sessionFile)) throw new Error('Pi did not persist T1 session');
  manager = pi.SessionManager.open(sessionFile, sessionDir);
  session = await createSession(manager, 'resume', sessionFile);
  record({ type: 'session_restored', session_file: sessionFile });
  currentTurn = turns[1].turn_id;
  const beforeT2 = messages.length;
  await session.prompt(turns[1].prompt);
  turnMetrics.push({ turn_id: currentTurn, success_checks: checkTurn(turnChecks[1] || []),
    assistant_message_count: messages.length - beforeT2 });
} else {
  currentTurn = 'T1';
  await session.prompt(prompt);
}
await stop(session);

const toolCalls = events.filter(event => event.type === 'tool_call');
const activations = events.filter(event => event.type === 'capability_activation');
const applies = toolCalls.filter(event => event.tool === 'apply_capability');
const searches = toolCalls.filter(event => event.tool === 'load_capability');
const explicitBodyLoads = toolCalls.filter(event => event.tool === 'load_skill_body');
const nativeBodyLoads = toolCalls.filter(event => event.tool === 'read' &&
  String(event.args?.path).includes('/skills/'));
const bodyLoads = arm === 'native'
  ? nativeBodyLoads.map(event => ({ ...event, skill_id: path.basename(path.dirname(event.args.path)) }))
  : [...applies.flatMap(event => (event.args?.skill_ids ?? []).map(skill_id => ({ ...event, skill_id }))),
     ...explicitBodyLoads.map(event => ({ ...event, skill_id: event.args?.skill_id }))];
const usage = messages.reduce((sum, row) => ({
  input: sum.input + (row.message.usage?.input ?? 0),
  output: sum.output + (row.message.usage?.output ?? 0),
  cacheRead: sum.cacheRead + (row.message.usage?.cacheRead ?? 0),
}), { input: 0, output: 0, cacheRead: 0 });
let embeddingRows = [];
if (existsSync(embeddingTelemetry)) embeddingRows = (await readFile(embeddingTelemetry, 'utf8'))
  .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
const sha256 = async file => createHash('sha256').update(await readFile(file)).digest('hex');
const gitSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim();
const piVersion = JSON.parse(await readFile(path.join(piRoot, 'package.json'), 'utf8')).version;
const trace = {
  events, usage, total_tool_calls: toolCalls.length,
  activated_skills: [...new Set(activations.flatMap(event => event.skill_ids))],
  discoveries: searches.map(event => event.args?.need),
  skill_body_loads: bodyLoads.map(event => event.skill_id),
  bundle_actions: applies.map(event => event.args?.action),
  bundle_reuse_count: turnMetrics.length > 1 && applies.length === 1 ? 1 : 0,
  session_restore_success: turns.length ? events.some(event => event.type === 'session_restored') : null,
  turn_metrics: turnMetrics.map(row => ({ ...row,
    discovery_count: searches.filter(event => event.turn_id === row.turn_id).length,
    skill_activation_count: activations.filter(event => event.turn_id === row.turn_id).length,
    skill_body_load_count: bodyLoads.filter(event => event.turn_id === row.turn_id).length,
    usage: messages.filter(event => event.turn_id === row.turn_id).reduce((sum, event) => ({
      input: sum.input + (event.message.usage?.input ?? 0), output: sum.output + (event.message.usage?.output ?? 0),
      cacheRead: sum.cacheRead + (event.message.usage?.cacheRead ?? 0),
    }), { input: 0, output: 0, cacheRead: 0 }),
  })),
  query_embedding_calls_startup: embeddingRows.filter(row => row.phase === 'startup').length,
  query_embedding_calls_runtime: embeddingRows.filter(row => row.phase === 'runtime').length,
  control_plane_telemetry: arm === 'control-plane'
    ? { retrieval_calls: searches.length, retrieval_events: [] }
    : { retrieval_calls: null, retrieval_events: null },
  system_context_projection: { source: 'Pi resource loader; benchmark gold is host-only' },
  tool_descriptions: arm === 'native'
    ? ['Read files', 'Write files', 'Edit files', 'Run shell commands']
    : ['Read files', 'Write files', 'Edit files', 'Run shell commands',
       'Search matching capabilities', 'Commit selected capabilities', 'Reload a Skill body'],
  experiment_identity: {
    pi_package: '@earendil-works/pi-coding-agent', pi_version: piVersion,
    provider: providerId, model: modelId,
    reasoning_config: process.env.BENCHMARK_REASONING || null,
    temperature: process.env.BENCHMARK_TEMPERATURE ? Number(process.env.BENCHMARK_TEMPERATURE) : null,
    max_turns: maxTurns,
    timeout_seconds: Number(process.env.BENCHMARK_TIMEOUT_SECONDS || 300),
    corpus_version: 'benchmark-corpus-v0.1', corpus_subset: corpus,
    retrieval_card_identity: await sha256(cards), dense_index_identity: await sha256(dense),
    adapter_commit_sha: gitSha,
  },
};
await mkdir(path.dirname(tracePath), { recursive: true });
await writeFile(tracePath, JSON.stringify(trace, null, 2) + '\n');
