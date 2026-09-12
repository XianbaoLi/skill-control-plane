import { strict as assert } from "node:assert";
import test from "node:test";
import { formatRuntimePolicy, projectSkillBodies } from "../src/projection.ts";
import type { ContextSnapshot, ToolResultMessage } from "../src/types.ts";

const snapshot: ContextSnapshot = {
  maintained_bundles: [{
    bundle_id: "cap-test",
    purpose: "Document preparation",
    capabilities: ["write markdown"],
    members: [
      { skill_id: "skill-resident", name: "Resident", member_role: "maintained", body_state: "resident", short_description: "current resident work" },
      { skill_id: "skill-evicted", name: "Evicted", member_role: "direct", body_state: "evicted", short_description: null },
    ],
  }],
  remaining_search_budget: 2,
};

function toolResult(name: string, payload: unknown): ToolResultMessage {
  return {
    role: "toolResult",
    toolCallId: "call-1",
    toolName: name,
    content: [{ type: "text", text: JSON.stringify(payload) }],
  };
}

test("projection keeps resident Skill bodies", () => {
  const messages = [toolResult("load_skill_body", { skill_id: "skill-resident", body: "RESIDENT BODY" })];
  const projected = projectSkillBodies(messages, snapshot);
  assert.equal(projected[0], messages[0]);
  assert.match((projected[0] as ToolResultMessage).content[0].text, /RESIDENT BODY/);
});

test("projection replaces evicted Skill bodies", () => {
  const messages = [
    toolResult("apply_capability", {
      selected_skill_ids: ["skill-evicted"],
      skill_bodies: [{ skill_id: "skill-evicted", body: "LARGE BODY" }],
    }),
    toolResult("load_skill_body", { skill_id: "skill-evicted", body: "LARGE BODY" }),
  ];
  const [projected] = projectSkillBodies(messages, snapshot);
  const payload = JSON.parse((projected as ToolResultMessage).content[0].text);
  assert.match(payload.skill_bodies[0].body, /Skill body evicted/);
  assert.equal(messages[0], messages[0]);
});

test("projection is idempotent and ignores unrelated tool results", () => {
  const messages = [
    toolResult("load_skill_body", { skill_id: "skill-evicted", body: "LARGE BODY" }),
    toolResult("read", { skill_id: "skill-evicted", body: "LARGE BODY" }),
  ];
  const first = projectSkillBodies(messages, snapshot);
  const second = projectSkillBodies(first, snapshot);
  assert.deepEqual(second, first);
  assert.equal((second[0] as ToolResultMessage).content[0].text, (first[0] as ToolResultMessage).content[0].text);
});

test("runtime policy includes bundle cards and DIRECT rules", () => {
  const text = formatRuntimePolicy(snapshot);
  assert.match(text, /cap-test/);
  assert.match(text, /skill-evicted/);
  assert.match(text, /DIRECT/);
  assert.match(text, /apply_capability/);
  assert.match(text, /Remaining load_capability search budget for this turn: 2/);
});
