import { adapterConfigFromEnvironment, type AdapterConfig } from "./config.ts";
import { formatRuntimePolicy, projectSkillBodies } from "./projection.ts";
import { SidecarClient, SidecarError } from "./sidecar-client.ts";
import { STATE_ENTRY_TYPE, isStateSnapshot } from "./state.ts";
import { runtimeEvidenceFromToolResult } from "./runtime-evidence.ts";
import type { AgentMessage, ExtensionContext, Handshake, PiAPI, SidecarClientLike, ToolResultMessage } from "./types.ts";

type TypeBoxLike = {
  Object(fields: Record<string, unknown>, options?: Record<string, unknown>): unknown;
  Optional(schema: unknown): unknown;
  String(options?: Record<string, unknown>): unknown;
  Integer(options?: Record<string, unknown>): unknown;
  Array(schema: unknown, options?: Record<string, unknown>): unknown;
};

export type AdapterDependencies = {
  config: AdapterConfig;
  typebox: { Type: TypeBoxLike };
  sidecarFactory?: (config: AdapterConfig) => StartableSidecar;
};

export type StartableSidecar = SidecarClientLike & {
  start(): Promise<Handshake>;
};

export class PiSidecarAdapter {
  private readonly config: AdapterConfig;
  private readonly typebox: { Type: TypeBoxLike };
  private readonly makeSidecar: (config: AdapterConfig) => StartableSidecar;
  private client: StartableSidecar | null = null;
  private pi: PiAPI | null = null;

  constructor({ config, typebox, sidecarFactory }: AdapterDependencies) {
    this.config = config;
    this.typebox = typebox;
    this.makeSidecar = sidecarFactory ?? ((input) => new SidecarClient(input));
  }

  register(pi: PiAPI): void {
    this.pi = pi;
    pi.on("session_start", async (_event, ctx) => {
      await this.handleSessionStart(ctx);
    });
    pi.on("agent_start", async () => {
      await this.handleTurnStart();
    });
    pi.on("before_agent_start", async () => {
      return await this.handleBeforeAgentStart();
    });
    pi.on("context", async (event) => {
      return { messages: await this.projectContext(event.messages as AgentMessage[]) };
    });
    pi.on("agent_end", async () => {
      await this.handleTurnEnd();
    });
    pi.on("session_compact", async (_event, ctx) => {
      await this.handleSessionCompact(ctx);
    });
    pi.on("session_compact_failed", async (event, ctx) => {
      this.handleSessionCompactFailed(event, ctx);
    });
    pi.on("session_shutdown", async (_event, ctx) => {
      await this.handleSessionShutdown(ctx);
    });
    this.registerTools(pi);
  }

  private requireSidecar(): SidecarClientLike {
    if (this.client === null) {
      throw new SidecarError("NOT_READY", "sidecar is not started");
    }
    return this.client;
  }

  async handleSessionStart(ctx: ExtensionContext): Promise<void> {
    const sidecar = this.makeSidecar(this.config);
    this.client = sidecar;
    try {
      await sidecar.start();
      const snapshot = this.restoreEnabled() ? this.latestSnapshot(ctx) : null;
      if (snapshot !== null) {
        await this.requireSidecar().request("restore_state", snapshot);
      }
    } catch (error) {
      this.client = null;
      await sidecar.shutdown();
      const message = error instanceof Error ? error.message : String(error);
      ctx.ui.notify(`Skill Control Plane startup failed: ${message}`, "error");
      throw error;
    }
  }

  private latestSnapshot(ctx: ExtensionContext): Record<string, unknown> | null {
    for (const entry of [...ctx.sessionManager.getEntries()].reverse()) {
      if (entry.type !== "custom" || entry.customType !== STATE_ENTRY_TYPE) {
        continue;
      }
      if (!isStateSnapshot(entry.data)) {
        throw new Error(`invalid Skill Control Plane state entry: ${STATE_ENTRY_TYPE}`);
      }
      return entry.data as Record<string, unknown>;
    }
    return null;
  }

  private restoreEnabled(): boolean {
    // Explicitly supports the benchmark restore ON/OFF ablation.  Production
    // remains restore-on by default; only the literal "0" opts out.
    return process.env.SKILL_CONTROL_PLANE_RESTORE_ENABLED !== "0";
  }

  async handleTurnStart(): Promise<void> {
    await this.requireSidecar().request("begin_turn");
  }

  async handleBeforeAgentStart(): Promise<{ message: Record<string, unknown> } | undefined> {
    if (this.client === null) {
      return undefined;
    }
    const snapshot = await this.client.request<Parameters<typeof formatRuntimePolicy>[0]>(
      "context_snapshot",
      { compact: true },
    );
    return {
      message: {
        customType: "skill-control-plane/runtime-policy",
        content: [{ type: "text", text: formatRuntimePolicy(snapshot) }],
        display: false,
        details: { compact: true },
      },
    };
  }

  async projectContext(messages: readonly AgentMessage[]): Promise<AgentMessage[]> {
    if (this.client === null) {
      throw new SidecarError("NOT_READY", "sidecar is not started");
    }
    const snapshot = await this.client.request<Parameters<typeof projectSkillBodies>[1]>(
      "context_snapshot",
      { compact: true },
    );
    const projected = projectSkillBodies(messages, snapshot);
    const discoveries: Array<Record<string, unknown>> = [];
    for (const message of projected) {
      if (message.role !== "toolResult") continue;
      const result = message as ToolResultMessage;
      const evidence = runtimeEvidenceFromToolResult(result);
      if (evidence === null) continue;
      const outcome = await this.requireSidecar().request<Record<string, unknown>>(
        "observe_runtime_evidence", { evidence },
      );
      if (outcome.status === "discovered") discoveries.push(outcome);
    }
    if (discoveries.length === 0) return projected;
    return [...projected, ...discoveries.map(outcome => ({
      role: "user",
      content: [{ type: "text", text: `Runtime reroute discovery (selection remains yours):\n${JSON.stringify({
        evidence_id: outcome.evidence_id,
        need: outcome.need,
        candidates: outcome.candidates,
        current_bundle_cards: snapshot.maintained_bundles,
      })}\nSelect only needed candidate Skills. Pass reroute_evidence_id to apply_capability when committing this discovery.` }],
    }))];
  }

  async handleTurnEnd(): Promise<void> {
    await this.requireSidecar().request("end_turn");
  }

  async handleSessionCompact(ctx: ExtensionContext): Promise<void> {
    const sidecar = this.requireSidecar();
    await sidecar.request("mark_all_skill_bodies_evicted");
    await this.persistState();
  }

  handleSessionCompactFailed(event: { reason?: string; errorMessage?: string; aborted?: boolean }, ctx: ExtensionContext): void {
    ctx.ui.notify(
      `Skill Control Plane compaction failed; Skill bodies were not evicted (${event.reason ?? "unknown"}).`,
      "warning",
    );
  }

  async handleSessionShutdown(ctx: ExtensionContext): Promise<void> {
    try {
      await this.persistState();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      ctx.ui.notify(`Skill Control Plane failed to persist state: ${message}`, "error");
    } finally {
      const sidecar = this.client;
      this.client = null;
      if (sidecar !== null) {
        await sidecar.shutdown();
      }
    }
  }

  private async persistState(): Promise<void> {
    const state = await this.requireSidecar().request("export_state");
    if (this.pi !== null) {
      this.pi.appendEntry(STATE_ENTRY_TYPE, state);
    }
  }

  private registerTools(pi: PiAPI): void {
    const { Type } = this.typebox;
    pi.registerTool({
      name: "load_capability",
      label: "Load capability",
      description: "Search Skill Control Plane for capabilities matching the current need.",
      parameters: Type.Object({
        need: Type.String({ minLength: 1 }),
        k: Type.Optional(Type.Integer({ minimum: 1 })),
      }),
      execute: async (_toolCallId: string, params: { need: string; k?: number }) => {
        return await this.executeTool("search_capability", {
          need: params.need,
          ...(params.k === undefined ? {} : { k: params.k }),
        });
      },
    });

    pi.registerTool({
      name: "apply_capability",
      label: "Apply capability",
      description: "Commit selected Skills after discovery. CREATE requires purpose and creates a Bundle; EXTEND and DIRECT require target_bundle_id. Every selected Skill requires a coverage item whose covered_by is exactly skill:<selected_skill_id>; existing Bundle coverage uses bundle:<active_bundle_id>. For runtime reroute candidates, include reroute_evidence_id from the discovery projection.",
      // Some OpenAI-compatible providers emit `{}` for JSON-schema unions.
      // Keep the provider-facing shape a plain object and put the conditional
      // contract in the description; the sidecar remains the strict validator.
      parameters: Type.Object({
        action: Type.String({ enum: ["DIRECT", "EXTEND", "CREATE"] }),
        skill_ids: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
        reason: Type.String({ minLength: 1 }),
        target_bundle_id: Type.Optional(Type.String({ minLength: 1,
          description: "Required for EXTEND and DIRECT; omit for CREATE." })),
        purpose: Type.Optional(Type.String({ minLength: 1,
          description: "Required for CREATE; omit for EXTEND and DIRECT." })),
        coverage: Type.Optional(Type.Array(Type.Object({
          need: Type.String({ minLength: 1,
            description: "The concrete need covered by this selected Skill or active Bundle." }),
          covered_by: Type.String({ minLength: 1,
            description: "Use exactly skill:<selected_skill_id> or bundle:<active_bundle_id>; bare IDs are invalid." }),
        }))),
        remaining_gaps: Type.Optional(Type.Array(Type.String({ minLength: 1 }))),
        reroute_evidence_id: Type.Optional(Type.String({ minLength: 1 })),
      }, { additionalProperties: false }),
      execute: async (_toolCallId: string, params: Record<string, unknown>) => {
        const applied = await this.requireSidecar().request<Record<string, unknown>>(
          "apply_capability", params,
        );
        const snapshot = await this.requireSidecar().request<Parameters<typeof formatRuntimePolicy>[0]>(
          "context_snapshot", { compact: true },
        );
        const bundleId = typeof applied.affected_bundle_id === "string"
          ? applied.affected_bundle_id : null;
        const bundle = bundleId === null ? null : snapshot.maintained_bundles
          .find((candidate) => candidate.bundle_id === bundleId) ?? null;
        // The immediate tool result is the stable, canonical target for a
        // same-turn EXTEND/DIRECT; it must not depend on a later policy refresh.
        const result = {
          ...applied,
          bundle_target: bundleId === null ? null : { bundle_id: bundleId, bundle },
        };
        await this.persistState();
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    pi.registerTool({
      name: "load_skill_body",
      label: "Load Skill body",
      description: "Reload an evicted Skill body from Skill Control Plane.",
      parameters: Type.Object({
        skill_id: Type.String({ minLength: 1 }),
      }),
      execute: async (_toolCallId: string, params: { skill_id: string }) => {
        return await this.executeTool("load_skill_body", { skill_id: params.skill_id });
      },
    });
  }

  private async executeTool(method: string, params: Record<string, unknown>) {
    const result = await this.requireSidecar().request(method, params);
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
      details: result,
    };
  }
}

export function createSkillControlPlaneAdapter(
  dependencies: AdapterDependencies,
): PiSidecarAdapter {
  return new PiSidecarAdapter(dependencies);
}

export default async function skillControlPlanePiExtension(pi: PiAPI): Promise<void> {
  const typebox = await import("typebox");
  const adapter = createSkillControlPlaneAdapter({
    config: adapterConfigFromEnvironment(process.env),
    typebox: typebox as unknown as { Type: TypeBoxLike },
  });
  adapter.register(pi);
}
