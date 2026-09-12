/** Minimal real-Pi smoke bridge. Uses Pi's deterministic faux provider so the
 * harness is offline and both arms share the exact model/provider. */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
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
const arm = process.env.BENCHMARK_ARM;
const corpus = process.env.BENCHMARK_CORPUS;
const prompt = process.env.BENCHMARK_PROMPT;
const workspace = process.env.BENCHMARK_WORKSPACE;
const tracePath = process.env.BENCHMARK_TRACE;
if (!['native', 'control-plane'].includes(arm) || !corpus || !prompt || !workspace || !tracePath) {
  throw new Error('benchmark bridge environment is incomplete');
}
const openpi = process.env.OPENPI_ROOT ?? path.join(homedir(), 'workspace/openpi');
const piRoot = process.env.PI_CORE_ROOT ?? path.join(openpi, 'node_modules/@earendil-works/pi-coding-agent');
const pi = await import(pathToFileURL(path.join(piRoot, 'dist/index.js')));
const nestedAi = path.join(piRoot, 'node_modules/@earendil-works/pi-ai/dist/index.js');
const ai = await import(pathToFileURL(existsSync(nestedAi) ? nestedAi : path.join(piRoot, '../pi-ai/dist/index.js')));
const skillRoot = path.join(repo, 'local_artifacts/corpora/benchmark-corpus-v0.1', corpus, 'skills');
let artifacts = path.join(repo, 'local_artifacts/benchmark-v0.1', corpus);
let cards = path.join(artifacts, 'retrieval-cards-v0.1.jsonl');
let dense = path.join(artifacts, 'dense-index-v1.json');

const walkSkills = async (directory) => {
  const result = [];
  for (const category of await (await import('node:fs/promises')).readdir(directory, { withFileTypes: true })) {
    if (!category.isDirectory()) continue;
    for (const skill of await (await import('node:fs/promises')).readdir(path.join(directory, category.name), { withFileTypes: true })) {
      const candidate = path.join(directory, category.name, skill.name, 'SKILL.md');
      if (skill.isDirectory() && existsSync(candidate)) result.push(candidate);
    }
  }
  return result.sort();
};
const skillPaths = await walkSkills(skillRoot);
if (arm === 'control-plane' && (!existsSync(cards) || !existsSync(dense)) && corpus !== 'S128') {
  // Smoke-only, byte/value-preserving subset view. The frozen S128 artifacts are
  // read-only and no card, representation, vector, rank, or retrieval setting is changed.
  const ids = new Set(skillPaths.map(file => path.basename(path.dirname(file))));
  const source = path.join(repo, 'local_artifacts/benchmark-v0.1/S128');
  artifacts = path.join(workspace, '.benchmark-artifacts', corpus);
  cards = path.join(artifacts, 'retrieval-cards-v0.1.jsonl');
  dense = path.join(artifacts, 'dense-index-v1.json');
  await mkdir(artifacts, { recursive: true });
  const cardLines = (await readFile(path.join(source, 'retrieval-cards-v0.1.jsonl'), 'utf8')).split(/\r?\n/)
    .filter(line => line.trim() && ids.has(JSON.parse(line).skill_id));
  await writeFile(cards, cardLines.join('\n') + '\n');
  const index = JSON.parse(await readFile(path.join(source, 'dense-index-v1.json'), 'utf8'));
  index.records = index.records.filter(record => ids.has(record.skill_id));
  await writeFile(dense, JSON.stringify(index) + '\n');
}
const target = prompt.includes('messaging.json') ? 'storybrand-messaging' : 'requirements-analysis';
const outputName = target === 'storybrand-messaging' ? 'messaging.json' : 'requirements.json';
const outputValue = target === 'storybrand-messaging'
  ? { customer: 'clinic managers', problem: 'manual scheduling', guide: 'the product', plan: ['import', 'verify', 'publish'], call_to_action: 'Start an import', failure: 'another late rota', success: 'a trusted rota before Monday' }
  : { functional_requirements: ['guest checkout', 'persistent basket'], non_functional_requirements: ['holiday readiness'], assumptions: ['legal review is a gate'], acceptance_criteria: ['a guest can restore a basket'] };

const provider = ai.fauxProvider({ api: 'benchmark-faux', provider: 'benchmark-faux',
  models: [{ id: 'benchmark-faux-1', name: 'Benchmark Faux 1', reasoning: false }] });
const responses = [];
if (arm === 'native') {
  responses.push(ai.fauxAssistantMessage(ai.fauxToolCall('read', { path: skillPaths.find(p => p.endsWith(`/${target}/SKILL.md`)) }), { stopReason: 'toolUse' }));
} else {
  if (!existsSync(cards) || !existsSync(dense)) throw new Error(`no Control Plane artifact view for ${corpus}: ${artifacts}`);
  responses.push(ai.fauxAssistantMessage(ai.fauxToolCall('load_capability', { need: prompt, k: 10 }), { stopReason: 'toolUse' }));
  responses.push(ai.fauxAssistantMessage(ai.fauxToolCall('apply_capability', {
    action: 'CREATE', skill_ids: [target], reason: 'matches the task evidence',
    purpose: 'complete the requested artifact', coverage: [{ need: prompt, covered_by: target }], remaining_gaps: [],
  }), { stopReason: 'toolUse' }));
}
responses.push(ai.fauxAssistantMessage(ai.fauxToolCall('write', {
  path: path.join(workspace, outputName), content: JSON.stringify(outputValue, null, 2) + '\n',
}), { stopReason: 'toolUse' }));
responses.push(ai.fauxAssistantMessage('Completed the requested artifact.'));
provider.setResponses(responses);

const agentDir = path.join(workspace, '.pi-agent');
await mkdir(agentDir, { recursive: true });
const settingsManager = pi.SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: true });
const modelRuntime = await pi.ModelRuntime.create({ authPath: path.join(agentDir, 'auth.json'), modelsPath: path.join(agentDir, 'models.json') });
modelRuntime.registerNativeProvider(provider.provider);
await modelRuntime.setRuntimeApiKey(provider.provider.id, 'offline-fixture-key');
let loader;
if (arm === 'native') {
  loader = new pi.DefaultResourceLoader({ cwd: workspace, agentDir, settingsManager,
    noSkills: false, additionalSkillPaths: skillPaths, noPromptTemplates: true });
} else {
  process.env.SKILL_CONTROL_PLANE_PYTHON = path.join(repo, 'scripts/benchmark_offline_sidecar.py');
  process.env.SKILL_CONTROL_PLANE_SKILL_ROOT = skillRoot;
  process.env.SKILL_CONTROL_PLANE_RETRIEVAL_CARDS = cards;
  process.env.SKILL_CONTROL_PLANE_DENSE_INDEX = dense;
  process.env.PYTHONPATH = path.join(repo, 'src');
  const extension = (await import(pathToFileURL(path.join(repo, 'adapters/pi/src/index.ts')))).default;
  loader = new pi.DefaultResourceLoader({ cwd: workspace, agentDir, settingsManager,
    noSkills: true, noPromptTemplates: true, extensionFactories: [extension] });
}
await loader.reload();
const { session } = await pi.createAgentSession({ cwd: workspace, agentDir, model: provider.getModel(),
  modelRuntime, settingsManager, resourceLoader: loader,
  sessionManager: pi.SessionManager.create(workspace, path.join(workspace, 'sessions')),
  tools: arm === 'native' ? ['read', 'write'] : ['write', 'load_capability', 'apply_capability', 'load_skill_body'] });
const events = [];
const assistantMessages = [];
session.subscribe(event => {
  if (event.type === 'tool_execution_start') events.push({ type: event.type, tool: event.toolName, args: event.args });
  if (event.type === 'tool_execution_end') events.push({ type: event.type, tool: event.toolName, is_error: event.isError, result: event.result });
  if (event.type === 'message_end' && event.message.role === 'assistant') assistantMessages.push(event.message);
});
try {
  await session.bindExtensions({ mode: 'print', onError: error => { throw error; } });
  await session.prompt(prompt);
  const starts = events.filter(event => event.type === 'tool_execution_start');
  const apply = starts.filter(event => event.tool === 'apply_capability');
  const loads = starts.filter(event => event.tool === 'load_capability');
  const reads = starts.filter(event => event.tool === 'read' && String(event.args?.path).includes('/skills/'));
  const usage = assistantMessages.reduce((sum, message) => ({
    input: sum.input + (message.usage?.input ?? 0), output: sum.output + (message.usage?.output ?? 0),
    cacheRead: sum.cacheRead + (message.usage?.cacheRead ?? 0),
  }), { input: 0, output: 0, cacheRead: 0 });
  const activated = arm === 'native' ? reads.map(event => path.basename(path.dirname(event.args.path)))
    : apply.flatMap(event => event.args.skill_ids ?? []);
  const compactRetrieval = events.filter(event => event.type === 'tool_execution_end' && event.tool === 'load_capability').map(event => {
    try {
      const payload = JSON.parse(event.result.content.find(block => block.type === 'text').text);
      return { query: payload.query, candidates: payload.candidates.map(candidate => {
        const raw = payload.retrieval_trace?.candidates?.find(row => row.skill_id === candidate.skill_id);
        return { skill_id: candidate.skill_id, rrf_rank: candidate.rank,
          bm25_rank: null, dense_rank: null,
          bm25_score: raw?.source_scores?.bm25 ?? null,
          dense_score: raw?.source_scores?.dense ?? null };
      }), component_rank_status: 'unavailable-from-current-sidecar-trace' };
    } catch { return { parse_error: true }; }
  });
  const trace = { pi_version: JSON.parse(await readFile(path.join(piRoot, 'package.json'), 'utf8')).version,
    provider: provider.provider.id, model: provider.getModel().id, usage, total_tool_calls: starts.length,
    activated_skills: activated, discoveries: loads.map(event => event.args.need), skill_body_loads: [],
    bundle_actions: apply.map(event => event.args.action), bundle_reuse_count: 0,
    query_embedding_calls: loads.length, smoke_query_embedding_provider: 'local-zero-vector-not-for-benchmark-results',
    control_plane_telemetry: arm === 'control-plane' ? { retrieval_calls: loads.length,
      retrieval_events: compactRetrieval } : { retrieval_calls: 0, retrieval_events: null },
  };
  await mkdir(path.dirname(tracePath), { recursive: true });
  await writeFile(tracePath, JSON.stringify(trace, null, 2) + '\n');
} finally {
  await session.abort();
  await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'quit' });
  session.dispose();
}
