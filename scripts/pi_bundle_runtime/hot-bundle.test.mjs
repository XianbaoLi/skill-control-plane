import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, rm, readFile, writeFile } from 'node:fs/promises';
import { tmpdir, homedir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { createBundleResources } from './adapter.mjs';

const openpi = process.env.OPENPI_ROOT ?? path.join(homedir(), 'workspace/openpi');
// Default to the checkout's actual dependency, never silently fall back.
const piRoot = process.env.PI_CORE_ROOT ?? path.join(openpi, 'node_modules/@earendil-works/pi-coding-agent');
console.log(JSON.stringify({ piRoot, piVersion: JSON.parse(await readFile(path.join(piRoot, 'package.json'), 'utf8')).version, openpi }));
const corpus = path.resolve('local_artifacts/v0.5/hermes-current87');
const pi = await import(pathToFileURL(path.join(piRoot, 'dist/index.js')));
const { existsSync } = await import('node:fs');
const nestedAi = path.join(piRoot, 'node_modules/@earendil-works/pi-ai/dist/index.js');
const ai = await import(pathToFileURL(existsSync(nestedAi) ? nestedAi : path.join(piRoot, '../pi-ai/dist/index.js')));
const paths = ['software-development/python-debugpy', 'software-development/node-inspect-debugger', 'devops/sdlc-review'].map(p => path.join(corpus, p, 'SKILL.md'));
const names = ['python-debugpy', 'node-inspect-debugger', 'sdlc-review'];
const hashes = () => Promise.all(paths.map(async p => createHash('sha256').update(await readFile(p)).digest('hex')));

for (const withOpenPI of [false, true]) for (const mode of (withOpenPI ? ['adapter'] : ['adapter', 'native-path-array'])) test(`same session exact bundle replacement; OpenPI=${withOpenPI}; ${mode}`, async () => {
  const before = await hashes();
  const root = await mkdtemp(path.join(tmpdir(), 'pi-hot-bundle-'));
  const agentDir = path.join(root, 'agent');
  await mkdir(agentDir);
  const original = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = agentDir;
  let session;
  try {
    const snapshots = [], events = [], errors = [];
    let commands;
    const provider = ai.fauxProvider({ api: 'bundle-test', provider: `bundle-${withOpenPI}`, models: [{ id: 'fixture', name: 'Fixture', reasoning: false }] });
    provider.setResponses(Array.from({ length: 10 }, () => context => {
      snapshots.push({ systemPrompt: context.systemPrompt, messages: structuredClone(context.messages) });
      return ai.fauxAssistantMessage('Turn completed.');
    }));
    const settingsManager = pi.SettingsManager.inMemory({ packages: withOpenPI ? [openpi] : [], compaction: { enabled: false }, retry: { enabled: false } }, { projectTrusted: false });
    const modelRuntime = await pi.ModelRuntime.create({ authPath: path.join(agentDir, 'auth.json'), modelsPath: path.join(agentDir, 'models.json') });
    modelRuntime.registerNativeProvider(provider.provider);
    await modelRuntime.setRuntimeApiKey(provider.provider.id, 'fixture-key');
    const explicitPaths = paths.slice(0, 2);
    const options = {
      cwd: root, agentDir, settingsManager, additionalSkillPaths: explicitPaths, noPromptTemplates: true,
      extensionFactories: [api => {
        commands = () => api.getCommands();
        api.on('session_start', e => { events.push(`start:${e.reason}`); });
        api.on('session_shutdown', e => { events.push(`shutdown:${e.reason}`); });
        api.on('resources_discover', e => { events.push(`discover:${e.reason}`); });
      }],
    };
    const bundle = mode === 'adapter' ? createBundleResources(pi, options) : {
      loader: new pi.DefaultResourceLoader({ ...options, noSkills: true }),
      async update(session, next) {
        // Probe current constructor aliasing, not a documented replacement API.
        explicitPaths.splice(0, explicitPaths.length, ...next);
        const oldPrompt = session.systemPrompt;
        await this.loader.reload();
        assert.deepEqual(this.loader.getSkills().skills.map(s => s.name).sort(), next.map(p => names[paths.indexOf(p)]).sort());
        assert.equal(session.systemPrompt, oldPrompt, 'loader reload alone leaves cached system prompt stale');
        await session.reload();
      },
    };
    await bundle.loader.reload();
    assert.deepEqual(bundle.loader.getExtensions().errors, []);
    if (withOpenPI) assert.ok(bundle.loader.getExtensions().extensions.some(e => e.path.startsWith(openpi)));
    const skillNames = () => bundle.loader.getSkills().skills.map(s => s.name).sort();
    assert.deepEqual(skillNames(), names.slice(0, 2).sort());
    ({ session } = await pi.createAgentSession({ cwd: root, agentDir, model: provider.getModel(), modelRuntime, settingsManager, resourceLoader: bundle.loader, sessionManager: pi.SessionManager.create(root, path.join(root, 'sessions')) }));
    await session.bindExtensions({ mode: 'print', onError: e => errors.push(e) });
    const identity = session.sessionManager.getSessionId();
    await session.prompt('First turn. Reply briefly.');
    const check = (snapshot, included, excluded) => {
      for (const index of included) {
        assert.ok(snapshot.systemPrompt.includes(`<name>${names[index]}</name>`));
        assert.ok(snapshot.systemPrompt.includes(bundle.loader.getSkills().skills.find(s => s.name === names[index]).description));
      }
      for (const index of excluded) assert.ok(!JSON.stringify(snapshot).includes(`<name>${names[index]}</name>`));
    };
    check(snapshots[0], [0, 1], [2]);
    const retired = bundle.loader.getSkills().skills.find(s => s.name === names[1]);
    const history = structuredClone(session.messages);
    await bundle.update(session, [paths[0], paths[2]]);
    assert.equal(session.sessionManager.getSessionId(), identity);
    assert.deepEqual(session.messages, history);
    assert.deepEqual(skillNames(), [names[0], names[2]].sort());
    const skillCommands = commands().filter(c => c.source === 'skill').map(c => c.name).sort();
    assert.deepEqual(skillCommands, [ `skill:${names[0]}`, `skill:${names[2]}` ].sort());
    await session.prompt('Second turn. Reply briefly.');
    assert.equal(snapshots.length, 2);
    check(snapshots[1], [0, 2], [1]);
    assert.ok(!JSON.stringify(snapshots[1]).includes(retired.description));
    assert.ok(!JSON.stringify(snapshots[1]).includes(retired.filePath));
    if (process.env.BUNDLE_EVIDENCE_DIR) {
      await mkdir(process.env.BUNDLE_EVIDENCE_DIR, { recursive: true });
      await writeFile(path.join(process.env.BUNDLE_EVIDENCE_DIR, `context-${withOpenPI}-${mode}.json`), JSON.stringify({ piRoot, openpi: withOpenPI ? openpi : null, sessionId: identity, before: snapshots[0], after: snapshots[1] }, null, 2));
    }
    assert.ok(events.includes('shutdown:reload'));
    assert.ok(events.includes('start:reload'));
    assert.ok(events.includes('discover:reload'));
    // Retired slash lookup no longer expands; added slash lookup reads real body.
    await session.prompt(`/skill:${names[1]} retired`);
    assert.equal(ai.contentText(session.messages.filter(m => m.role === 'user').at(-1).content), `/skill:${names[1]} retired`);
    await session.prompt(`/skill:${names[2]} added`);
    assert.equal(pi.parseSkillBlock(ai.contentText(session.messages.filter(m => m.role === 'user').at(-1).content))?.name, names[2]);
    // Retirement removes discovery, not already-expanded conversation history.
    await bundle.update(session, paths.slice(0, 2));
    await session.prompt('Continue after retiring the previously invoked skill.');
    assert.ok(!snapshots.at(-1).systemPrompt.includes(`<name>${names[2]}</name>`));
    assert.ok(snapshots.at(-1).messages.some(m => m.role === 'user' && pi.parseSkillBlock(ai.contentText(m.content))?.name === names[2]));
    assert.deepEqual(errors, []);
    assert.ok(!session.messages.some(m => m.role === 'assistant' && m.stopReason === 'error'));
    assert.deepEqual(await hashes(), before);
    console.log(JSON.stringify({ withOpenPI, mode, sessionId: identity, finalSkillsAfterHistoryProbe: skillNames(), bundle1Commands: skillCommands, events, turns: snapshots.length, sourceHashes: before }));
  } finally {
    if (session) { await session.abort(); await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'quit' }); session.dispose(); }
    if (original === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = original;
    await rm(root, { recursive: true, force: true });
  }
});
