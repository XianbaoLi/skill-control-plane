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

test("registers four LLM tools and maps them to sidecar methods", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  assert.deepEqual([...pi.tools.keys()].sort(), ["apply_capability", "capability_gap_check", "load_capability", "load_skill_body"]);

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

test("failure evidence injects one capability-gap checkpoint with evidence and Bundle Cards", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());

  const projected = await adapter.projectContext(failureEvidence());
  const checkpoint = projected.at(-1) as any;
  assert.equal(projected.length, 2);
  assert.equal(checkpoint.role, "user");
  assert.match(checkpoint.content[0].text, /burn-rate alert/);
  assert.match(checkpoint.content[0].text, /Active skills: fastify/);
  assert.match(checkpoint.content[0].text, /Bundle Cards:/);
  assert.match(checkpoint.content[0].text, /cap-runtime/);
});

test("ordinary successful tool results do not inject a checkpoint", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const messages = [{ role: "toolResult", toolCallId: "call-1", toolName: "bash", isError: false,
    content: [{ type: "text", text: "all tests passed" }] }];
  assert.equal((await adapter.projectContext(messages)).length, 1);
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

test("duplicate evidence in one user turn schedules no duplicate checkpoint", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  assert.equal((await adapter.projectContext(failureEvidence())).length, 2);
  assert.equal((await adapter.projectContext(failureEvidence())).length, 1);
});

test("different eligible evidence in one user turn each receives a checkpoint", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  const evidenceA = failureEvidence("test failure: item endpoint validation is missing");
  const first = await adapter.projectContext(evidenceA);
  assert.equal(first.length, 2);
  await pi.tools.get("capability_gap_check").execute("call-a", {
    needs_capability: false, need: null,
  });

  const evidenceB = [{ role: "toolResult", toolCallId: "call-2", toolName: "bash", isError: true,
    content: [{ type: "text", text: "verifier failure: burn-rate alert configuration is missing" }] }];
  const second = await adapter.projectContext([...evidenceA, ...evidenceB]);
  assert.equal(second.length, 3);
  assert.match((second.at(-1) as any).content[0].text, /burn-rate alert/);
  const telemetry = JSON.parse((await pi.tools.get("capability_gap_check").execute("call-b", {
    needs_capability: true, need: "configure burn-rate monitoring",
  })).content[0].text);
  assert.match(telemetry.evidence_fingerprint, /burn-rate alert/);
});

test("Control Plane tool results cannot recursively schedule checkpoints", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  assert.equal((await adapter.projectContext(failureEvidence())).length, 2);
  await pi.tools.get("capability_gap_check").execute("call", {
    needs_capability: false, need: null,
  });
  const internalResults = ["capability_gap_check", "load_capability", "search_capability", "apply_capability", "load_skill_body"]
    .map((toolName, index) => ({ role: "toolResult", toolCallId: `internal-${index}`, toolName,
      isError: true, content: [{ type: "text", text: "internal verifier failure" }] }));
  assert.equal((await adapter.projectContext(internalResults)).length, internalResults.length);
});

test("capability_gap_check validates false decisions and does not search", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  const gap = pi.tools.get("capability_gap_check");
  const result = await gap.execute("call", { needs_capability: false, need: null });
  const telemetry = JSON.parse(result.content[0].text);
  assert.equal(telemetry.needs_capability, false);
  assert.equal(telemetry.generated_need, null);
  assert.equal(telemetry.search_started, false);
  assert.equal(sidecar.calls.some(call => call.method === "search_capability"), false);
  await assert.rejects(gap.execute("call", { needs_capability: false, need: "monitoring" }), /empty or null/);
});

test("capability_gap_check true instructs existing load path and reports complete telemetry", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter, pi } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  await adapter.handleTurnStart();
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  await adapter.projectContext(failureEvidence());
  const gap = pi.tools.get("capability_gap_check");
  const result = await gap.execute("call", { needs_capability: true, need: "configure burn-rate monitoring" });
  const telemetry = JSON.parse(result.content[0].text);
  assert.deepEqual(Object.keys(telemetry).filter(key => ["trigger_reason", "evidence_fingerprint", "active_skill_ids", "needs_capability", "generated_need", "search_started"].includes(key)).sort(),
    ["active_skill_ids", "evidence_fingerprint", "generated_need", "needs_capability", "search_started", "trigger_reason"]);
  assert.equal(telemetry.search_started, true);
  assert.deepEqual(telemetry.active_skill_ids, ["fastify"]);
  assert.match(telemetry.next_step, /load_capability/);
  sidecar.responses.set("search_capability", searchResult());
  await pi.tools.get("load_capability").execute("call", { need: telemetry.generated_need });
  assert.equal(sidecar.calls.filter(call => call.method === "search_capability").length, 1);
  await assert.rejects(gap.execute("call", { needs_capability: true, need: "" }), /concise non-empty/);
});

test("checkpoint evidence cache is cleared at end_turn and permits a new user turn", async () => {
  const sidecar = new RecordingSidecar();
  const { adapter } = createAdapter(sidecar);
  await adapter.handleSessionStart(new FakeContext());
  sidecar.responses.set("context_snapshot", checkpointSnapshot());
  await adapter.handleTurnStart();
  assert.equal((await adapter.projectContext(failureEvidence())).length, 2);
  await adapter.handleTurnEnd();
  await adapter.handleTurnStart();
  assert.equal((await adapter.projectContext(failureEvidence())).length, 2);
});
