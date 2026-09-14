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

test("registers selection/apply tools without the deprecated gap-check tool", async () => {
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

test("apply schema makes action-dependent fields unambiguous", () => {
  const { pi } = createAdapter(new RecordingSidecar());
  const tool = pi.tools.get("apply_capability");
  assert.match(tool.description, /EXTEND.*target_bundle_id/);
  assert.deepEqual(tool.parameters.fields.action.string.enum, ["DIRECT", "EXTEND", "CREATE"]);
  assert.match(tool.parameters.fields.target_bundle_id.optional.string.description, /EXTEND/);
  assert.match(tool.parameters.fields.purpose.optional.string.description, /CREATE/);
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
  sidecar.responses.set("context_snapshot", {
    maintained_bundles: [{ bundle_id: "cap-new", purpose: "Writing", capabilities: ["release notes"],
      members: [{ skill_id: "release-notes", name: "Release notes", member_role: "maintained", body_state: "resident", short_description: "notes" }] }],
    remaining_search_budget: 2,
  });
  const result = await pi.tools.get("apply_capability").execute("call", {
    action: "CREATE",
    skill_ids: ["release-notes"],
    reason: "reusable writing work",
    purpose: "Writing",
    coverage: [{ need: "release notes", covered_by: "skill:release-notes" }],
  });
  const visible = JSON.parse(result.content[0].text);
  assert.equal(visible.affected_bundle_id, "cap-new");
  assert.equal(visible.bundle_target.bundle_id, "cap-new");
  assert.equal(visible.bundle_target.bundle.members[0].skill_id, "release-notes");
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

test("restore ablation can start a resumed Pi session without CapabilityMemory", async () => {
  const previous = process.env.SKILL_CONTROL_PLANE_RESTORE_ENABLED;
  process.env.SKILL_CONTROL_PLANE_RESTORE_ENABLED = "0";
  try {
    const sidecar = new RecordingSidecar();
    const { adapter } = createAdapter(sidecar);
    const ctx = new FakeContext();
    ctx.entries.push({ type: "custom", customType: "skill-control-plane/state-v1", data: { version: "wrong" } });
    await adapter.handleSessionStart(ctx);
    assert.equal(sidecar.calls.some((call) => call.method === "restore_state"), false);
  } finally {
    if (previous === undefined) delete process.env.SKILL_CONTROL_PLANE_RESTORE_ENABLED;
    else process.env.SKILL_CONTROL_PLANE_RESTORE_ENABLED = previous;
  }
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

function checkpointSnapshot() {
  return {
    maintained_bundles: [{
      bundle_id: "cap-runtime", purpose: "Endpoint implementation",
      capabilities: ["implement endpoints"],
      members: [{ skill_id: "fastify", name: "Fastify", member_role: "maintained",
        body_state: "resident", short_description: "Fastify service patterns" }],
    }],
    remaining_search_budget: 2,
  };
}

function failureEvidence(text = "test failure: burn-rate alert configuration is missing") {
  return [{ role: "toolResult", toolCallId: "call-1", toolName: "bash", isError: true,
    content: [{ type: "text", text }] }];
}

test("failure evidence is converted and discovered candidates are projected", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  sidecar.responses.set("observe_runtime_evidence", {
    status: "discovered",
    evidence_id: "call-1",
    need: "configure burn-rate monitoring",
    candidates: [{ skill_id: "monitoring", name: "Monitoring", rank: 1 }],
  });

  const projected = await adapter.projectContext(failureEvidence());
  const discovery = projected.at(-1) as any;
  assert.equal(projected.length, 2);
  assert.equal(discovery.role, "user");
  assert.match(discovery.content[0].text, /configure burn-rate monitoring/);
  assert.match(discovery.content[0].text, /monitoring/);
  assert.match(discovery.content[0].text, /cap-runtime/);
  const observed = sidecar.calls.find(call => call.method === "observe_runtime_evidence")!;
  assert.equal((observed.params.evidence as any).kind, "test_failure");
  assert.equal((observed.params.evidence as any).source, "bash");
  assert.match((observed.params.evidence as any).fingerprint, /^[a-f0-9]{64}$/);
});

test("Trace A: ordinary successful tool results create no eligible runtime evidence", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const messages = [{ role: "toolResult", toolCallId: "call-1", toolName: "bash", isError: false,
    content: [{ type: "text", text: "all tests passed" }] }];
  assert.equal((await adapter.projectContext(messages)).length, 1);
  assert.equal(sidecar.calls.some(call => call.method === "observe_runtime_evidence"), false);
});

test("successful source reads mentioning Error are not failure evidence", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const messages = [{ role: "toolResult", toolCallId: "call-1", toolName: "read", isError: false,
    content: [{ type: "text", text: "export function todo() { throw new Error('TODO'); }" }] }];
  assert.equal((await adapter.projectContext(messages)).length, 1);
});

test("duplicate outcome from the controller schedules no duplicate projection", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  sidecar.responses.set("observe_runtime_evidence", { status: "duplicate", evidence_id: "call-1" });
  assert.equal((await adapter.projectContext(failureEvidence())).length, 1);
});

test("different evidence in one user turn is independently sent to the controller", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const evidenceA = failureEvidence("test failure: item endpoint validation is missing");
  sidecar.responses.set("observe_runtime_evidence", { status: "no_gap", evidence_id: "call-1" });
  assert.equal((await adapter.projectContext(evidenceA)).length, 1);

  const evidenceB = [{ role: "toolResult", toolCallId: "call-2", toolName: "bash", isError: true,
    content: [{ type: "text", text: "verifier failure: burn-rate alert configuration is missing" }] }];
  sidecar.responses.set("observe_runtime_evidence", { status: "discovered", evidence_id: "call-2",
    need: "configure burn-rate monitoring", candidates: [{ skill_id: "monitoring", rank: 1 }] });
  const second = await adapter.projectContext(evidenceB);
  assert.equal(second.length, 2);
  assert.match((second.at(-1) as any).content[0].text, /monitoring/);
  assert.equal(sidecar.calls.filter(call => call.method === "observe_runtime_evidence").length, 2);
});

test("Control Plane tool results cannot recursively form evidence", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const internalResults = ["capability_gap_check", "load_capability", "search_capability", "discover_capability", "apply_capability", "load_skill_body", "telemetry"]
    .map((toolName, index) => ({ role: "toolResult", toolCallId: `internal-${index}`, toolName,
      isError: true, content: [{ type: "text", text: "internal verifier failure" }] }));
  assert.equal((await adapter.projectContext(internalResults)).length, internalResults.length);
  assert.equal(sidecar.calls.some(call => call.method === "observe_runtime_evidence"), false);
});

test("apply forwards reroute evidence id for runtime-owned selection and commit", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  sidecar.responses.set("apply_capability", { action: "CREATE", selected_skill_ids: ["monitoring"],
    affected_bundle_id: "cap-monitoring" });
  sidecar.responses.set("context_snapshot", { maintained_bundles: [], remaining_search_budget: 2 });
  sidecar.responses.set("export_state", { version: "state-snapshot-v1", store_fingerprint: "x", bundles: [] });

  await pi.tools.get("apply_capability").execute("apply", {
    action: "CREATE", skill_ids: ["monitoring"], reason: "alerts", purpose: "Monitoring",
    reroute_evidence_id: "call-1",
  });

  const applyCalls = sidecar.calls.filter(call => call.method === "apply_capability");
  assert.equal(applyCalls.length, 1);
  assert.equal((applyCalls[0].params as any).reroute_evidence_id, "call-1");
  assert.equal(sidecar.calls.some(call => call.method === "select_reroute_candidates"), false);
  assert.equal(sidecar.calls.some(call => call.method === "commit_reroute"), false);
});
