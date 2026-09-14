import { createHash } from "node:crypto";
import type { TextContent, ToolResultMessage } from "./types.ts";

export const CONTROL_PLANE_TOOL_NAMES = new Set([
  "capability_gap_check",
  "load_capability",
  "search_capability",
  "discover_capability",
  "apply_capability",
  "load_skill_body",
  "observe_runtime_evidence",
  "reroute_snapshot",
  "telemetry",
]);

export type RuntimeEvidenceKind =
  | "tool_error"
  | "test_failure"
  | "verifier_failure"
  | "new_subgoal"
  | "host_signal";

export type RuntimeEvidence = {
  evidence_id: string;
  kind: RuntimeEvidenceKind;
  source: string;
  text: string;
  fingerprint: string;
  metadata?: Record<string, unknown>;
};

const FAILURE_PATTERNS: Array<[RuntimeEvidenceKind, RegExp]> = [
  ["verifier_failure", /\b(?:verifier|verification)\s+(?:failed|failure|error)\b/i],
  ["test_failure", /\b(?:test(?:s| suite)?\s+(?:failed|failure|error)|assertion(?:\s+failed)?)\b/i],
];

export function runtimeEvidenceFromToolResult(
  result: ToolResultMessage,
): RuntimeEvidence | null {
  if (CONTROL_PLANE_TOOL_NAMES.has(result.toolName)) return null;
  const text = result.content
    .filter((item): item is TextContent => item.type === "text")
    .map(item => item.text)
    .join("\n")
    .trim();
  if (!text) return null;

  let kind: RuntimeEvidenceKind | null = null;
  if (result.toolName === "bash") {
    kind = FAILURE_PATTERNS.find(([, pattern]) => pattern.test(text))?.[0] ?? null;
  }
  if (kind === null && result.isError === true) kind = "tool_error";
  if (kind === null) return null;
  const normalized = text.replace(/\s+/g, " ").trim();
  return {
    evidence_id: result.toolCallId,
    kind,
    source: result.toolName,
    text,
    fingerprint: createHash("sha256")
      .update(`${kind}\0${result.toolName}\0${normalized}`)
      .digest("hex"),
    metadata: { tool_call_id: result.toolCallId },
  };
}
