import type { AgentMessage, ContextSnapshot, TextContent, ToolResultMessage } from "./types.ts";

export const EVICTED_MARKER_PREFIX = "Skill body evicted; call load_skill_body(";

function isEvictedMarker(text: string): boolean {
  return text.startsWith(EVICTED_MARKER_PREFIX);
}

export function formatBundleCards(snapshot: ContextSnapshot): string {
  const lines: string[] = [];
  for (const bundle of snapshot.maintained_bundles ?? []) {
    lines.push(`Bundle ${bundle.bundle_id}: ${bundle.purpose}`);
    for (const member of bundle.members) {
      const role = member.member_role === "direct" ? "DIRECT" : "MAINTAINED";
      const capabilities = member.short_description ?? "";
      lines.push(
        `- ${member.skill_id} (${member.name}, ${role}, ${member.body_state})${capabilities ? `: ${capabilities}` : ""}`,
      );
    }
    if (bundle.capabilities.length) {
      lines.push(`- capabilities: ${bundle.capabilities.join("; ")}`);
    }
  }
  return lines.join("\n");
}

export function formatRuntimePolicy(snapshot: ContextSnapshot): string {
  const cards = formatBundleCards(snapshot);
  const budget = snapshot.remaining_search_budget ?? 0;
  return [
    "Skill Control Plane runtime policy:",
    "- You can discover capabilities on demand with load_capability.",
    "- Use the existing Bundle Cards below first to decide whether current capabilities are enough.",
    cards ? `Bundle Cards:\n${cards}` : "Bundle Cards: none",
    "- If a capability is missing, call load_capability with a concise need.",
    "- After searching, you must call apply_capability for the selected Skill IDs.",
    "- CREATE and EXTEND are the default maintained actions.",
    "- DIRECT is only for clearly one-off or short-term capability use, and must target an existing Bundle ID.",
    "- DIRECT members remain metadata in the direct Bundle.",
    "- If a Skill body was evicted, call load_skill_body(skill_id) when needed.",
    "- Do not repeat an identical capability search.",
    `- Remaining load_capability search budget for this turn: ${budget}.`,
  ].join("\n");
}

function parseToolResultPayload(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function replaceBody(value: unknown, bodyState: (skillId: string) => "resident" | "evicted"): unknown {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const record = { ...(value as Record<string, unknown>) };

  if (typeof record.skill_id === "string" && typeof record.body === "string") {
    if (bodyState(record.skill_id) === "evicted" && !isEvictedMarker(record.body)) {
      record.body = `${EVICTED_MARKER_PREFIX}${JSON.stringify(record.skill_id)}) if needed.`;
    }
  }

  if (Array.isArray(record.skill_bodies)) {
    record.skill_bodies = record.skill_bodies.map((item) => replaceBody(item, bodyState));
  }

  return record;
}

function projectToolResult(
  message: ToolResultMessage,
  bodyState: (skillId: string) => "resident" | "evicted",
): ToolResultMessage {
  let changed = false;
  const content = message.content.map((item): TextContent | { type: string; [key: string]: unknown } => {
    if (item.type !== "text" || !("text" in item) || typeof (item as TextContent).text !== "string") {
      return item;
    }
    const originalText = (item as TextContent).text;
    const payload = parseToolResultPayload(originalText);
    if (payload === undefined) {
      return item;
    }
    const projected = replaceBody(payload, bodyState);
    const projectedText = JSON.stringify(projected);
    if (projectedText !== originalText) {
      changed = true;
      return { type: "text", text: projectedText };
    }
    return item;
  });
  return changed ? { ...message, content } : message;
}

export function projectSkillBodies(
  messages: readonly AgentMessage[],
  snapshot: ContextSnapshot,
): AgentMessage[] {
  const states = new Map<string, "resident" | "evicted">();
  for (const bundle of snapshot.maintained_bundles) {
    for (const member of bundle.members) {
      states.set(member.skill_id, member.body_state);
    }
  }
  const bodyState = (skillId: string) => states.get(skillId) ?? "evicted";

  return messages.map((message) => {
    if (message.role !== "toolResult") {
      return message;
    }
    const toolName = (message as ToolResultMessage).toolName;
    if (toolName !== "apply_capability" && toolName !== "load_skill_body") {
      return message;
    }
    return projectToolResult(message as ToolResultMessage, bodyState);
  });
}
