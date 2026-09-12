import { strict as assert } from "node:assert";
import test from "node:test";
import { createSkillControlPlaneAdapter } from "../src/index.ts";
import {
  FakeContext,
  FakePi,
  RecordingSidecar,
  fakeConfig,
  typeboxStub,
} from "./support.ts";

function createAdapter(sidecar: RecordingSidecar) {
  const adapter = createSkillControlPlaneAdapter({
    config: fakeConfig(),
    typebox: typeboxStub,
    sidecarFactory: () => sidecar as any,
  });
  const pi = new FakePi();
  adapter.register(pi);
  return { adapter, pi };
}

function searchResult() {
  return {
    query: "write release notes",
    candidates: [{ skill_id: "release-notes", name: "Release notes", rank: 1 }],
    search_control: { remaining_search_budget: 2 },
    retrieval_trace: { backend: "rrf" },
  };
}

test("registers three LLM tools and maps them to sidecar methods", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  assert.deepEqual([...pi.tools.keys()].sort(), ["apply_capability", "load_capability", "load_skill_body"]);

  sidecar.responses.set("search_capability", searchResult());
  const searched = await pi.tools.get("load_capability").execute("call", { need: "write release notes" });
  assert.deepEqual(sidecar.calls.at(-1), { method: "search_capability", params: { need: "write release notes" } });
  assert.equal(JSON.parse(searched.content[0].text).retrieval_trace.backend, "rrf");

  sidecar.responses.set("load_skill_body", { status: "loaded", skill_id: "release-notes", body: "BODY" });
  const loaded = await pi.tools.get("load_skill_body").execute("call", { skill_id: "release-notes" });
  assert.deepEqual(sidecar.calls.at(-1), { method: "load_skill_body", params: { skill_id: "release-notes" } });
  assert.match(loaded.content[0].text, /BODY/);
});

test("lifecycle maps to sidecar begin, end, compact and shutdown methods", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  const ctx = new FakeContext();
  await adapter.handleSessionStart(ctx);
  await adapter.handleTurnStart();
  await adapter.handleBeforeAgentStart();
  await adapter.handleTurnEnd();
  await adapter.handleSessionCompact(ctx);
  await adapter.handleSessionShutdown(ctx);
  assert.deepEqual(sidecar.calls.map((call) => call.method), [
    "handshake",
    "begin_turn",
    "context_snapshot",
    "end_turn",
    "mark_all_skill_bodies_evicted",
    "export_state",
    "export_state",
  ]);
  assert.equal(sidecar.stopped, true);
});

test("before_agent_start injects compact cards, policy and budget", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  sidecar.responses.set("context_snapshot", {
    maintained_bundles: [{
      bundle_id: "cap-doc",
      purpose: "Documentation",
      capabilities: ["write release notes"],
      members: [{ skill_id: "release-notes", name: "Release notes", member_role: "maintained", body_state: "resident", short_description: "one-off note" }],
    }],
    remaining_search_budget: 1,
  });
  const result = await adapter.handleBeforeAgentStart();
  const message = result?.message as any;
  assert.equal(message.customType, "skill-control-plane/runtime-policy");
  assert.equal(message.display, false);
  assert.match(message.content[0].text, /cap-doc/);
  assert.match(message.content[0].text, /DIRECT/);
  assert.match(message.content[0].text, /Remaining load_capability search budget for this turn: 1/);
});

test("intermediate Pi turns do not clear pending candidates", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  const ctx = new FakeContext();
  await adapter.handleSessionStart(ctx);
  await pi.emit("turn_start", {}, ctx);
  assert.equal(sidecar.calls.some((call) => call.method === "begin_turn"), false);
  await pi.emit("agent_start", {}, ctx);
  assert.equal(sidecar.calls.some((call) => call.method === "begin_turn"), true);
  await pi.emit("turn_end", {}, ctx);
  assert.equal(sidecar.calls.some((call) => call.method === "end_turn"), false);
  await pi.emit("agent_end", { messages: [] }, ctx);
  assert.equal(sidecar.calls.some((call) => call.method === "end_turn"), true);
});

test("restores latest valid custom state and appends state after apply", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  const ctx = new FakeContext();
  const snapshot = {
    version: "state-snapshot-v1",
    store_fingerprint: "skill-store-v1:test",
    bundles: [],
  };
  ctx.entries.push({ type: "custom", customType: "other", data: {} });
  ctx.entries.push({ type: "custom", customType: "skill-control-plane/state-v1", data: snapshot });
  await adapter.handleSessionStart(ctx);
  assert.deepEqual(sidecar.calls.find((call) => call.method === "restore_state")?.params, snapshot);

  sidecar.responses.set("search_capability", searchResult());
  sidecar.responses.set("apply_capability", {
    action: "CREATE",
    selected_skill_ids: ["release-notes"],
    affected_bundle_id: "cap-new",
    skill_bodies: [{ skill_id: "release-notes", body: "BODY" }],
  });
  sidecar.responses.set("export_state", snapshot);
  await pi.tools.get("apply_capability").execute("call", {
    action: "CREATE",
    skill_ids: ["release-notes"],
    reason: "reusable writing work",
    purpose: "Writing",
    coverage: [{ need: "release notes", covered_by: "skill:release-notes" }],
  });
  assert.equal(pi.entries.at(-1)?.customType, "skill-control-plane/state-v1");
  assert.deepEqual(pi.entries.at(-1)?.data, snapshot);
});

test("invalid latest state fails closed and shuts sidecar down", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  const ctx = new FakeContext();
  ctx.entries.push({ type: "custom", customType: "skill-control-plane/state-v1", data: { version: "wrong" } });
  await assert.rejects(adapter.handleSessionStart(ctx), /invalid Skill Control Plane state/);
  assert.match(ctx.notifications[0], /startup failed/);
  assert.equal(sidecar.stopped, true);
});

test("failed compaction does not evict Skill bodies", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  const ctx = new FakeContext();
  await pi.emit("session_compact_failed", { reason: "manual", errorMessage: "provider unavailable" }, ctx);
  assert.equal(sidecar.calls.some((call) => call.method === "mark_all_skill_bodies_evicted"), false);
  assert.equal(sidecar.calls.some((call) => call.method === "export_state"), false);
  assert.match(ctx.notifications[0], /were not evicted/);
});

test("context hook projects through sidecar state without mutating input", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  sidecar.responses.set("context_snapshot", {
    maintained_bundles: [{ bundle_id: "cap-doc", purpose: "Documentation", capabilities: [], members: [{ skill_id: "release-notes", name: "Release notes", member_role: "maintained", body_state: "evicted", short_description: null }] }],
    remaining_search_budget: 1,
  });
  const original = [{
    role: "toolResult",
    toolCallId: "call",
    toolName: "load_skill_body",
    content: [{ type: "text", text: JSON.stringify({ skill_id: "release-notes", body: "LARGE BODY" }) }],
  }];
  const handler = pi.handlers.get("context")![0];
  const result = await handler({ type: "context", messages: original }, new FakeContext());
  assert.notEqual(result.messages, original);
  assert.match(JSON.parse(result.messages[0].content[0].text).body, /Skill body evicted/);
  assert.match(original[0].content[0].text, /LARGE BODY/);
});
