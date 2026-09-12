export type TextContent = {
  type: "text";
  text: string;
};

export type ToolResultMessage = {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: Array<TextContent | { type: string; [key: string]: unknown }>;
  [key: string]: unknown;
};

export type AgentMessage = ToolResultMessage | { role: string; [key: string]: unknown };

export type SidecarValue = string | number | boolean | null | SidecarValue[] | { [key: string]: SidecarValue };

export type SidecarResponse = {
  id: string;
  ok: boolean;
  result?: unknown;
  error?: { code: string; message: string; details?: unknown };
};

export type SidecarRequest = {
  id: string;
  method: string;
  params: Record<string, unknown>;
};

export type Readiness = {
  ready: boolean;
  [key: string]: unknown;
};

export type Handshake = {
  protocol_version: string;
  readiness: Readiness;
  supported_methods: string[];
};

export type ContextSnapshot = {
  maintained_bundles: Array<{
    bundle_id: string;
    purpose: string;
    capabilities: string[];
    members: Array<{
      skill_id: string;
      name: string;
      member_role: "direct" | "maintained";
      body_state: "resident" | "evicted";
      short_description: string | null;
    }>;
  }>;
  remaining_search_budget: number;
  [key: string]: unknown;
};

export type SidecarClientLike = {
  request<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T>;
  shutdown(): Promise<void>;
};

export type ExtensionContext = {
  ui: {
    notify(message: string, type?: "info" | "warning" | "error"): void;
  };
  sessionManager: {
    getEntries(): Array<{ type?: string; customType?: string; data?: unknown }>;
  };
};

export type PiAPI = {
  on(event: string, handler: (event: any, ctx: ExtensionContext) => unknown): void;
  registerTool(tool: Record<string, unknown>): void;
  appendEntry(customType: string, data?: unknown): unknown;
};
