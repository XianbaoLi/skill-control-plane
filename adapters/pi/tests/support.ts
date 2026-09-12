import { strict as assert } from "node:assert";

export class FakeStream {
  private listeners = new Map<string, Array<(chunk: Buffer) => void>>();
  public writes: string[] = [];

  on(event: string, listener: (chunk: Buffer) => void): void {
    const list = this.listeners.get(event) ?? [];
    list.push(listener);
    this.listeners.set(event, list);
  }

  emit(chunk: string | Buffer): void {
    const data = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, "utf-8");
    for (const listener of this.listeners.get("data") ?? []) {
      listener(data);
    }
  }

  write(data: string): boolean {
    this.writes.push(data);
    return true;
  }

  end(): void {
    this.writes.push("__end__");
  }
}

export class FakeSidecarProcess {
  public pid = 12345;
  public exitCode: number | null = null;
  public stdin = new FakeStream();
  public stdout = new FakeStream();
  public stderr = new FakeStream();
  public requests: Array<{ id: string; method: string; params: Record<string, unknown> }> = [];
  public signals: string[] = [];
  public spawnError: Error | null = null;
  private exitListeners: Array<() => void> = [];
  private errorListeners: Array<(error: Error) => void> = [];

  respond(method: string, result?: unknown, responseId?: string): void {
    const request = this.requests.find((item) => item.method === method);
    assert.ok(request, `missing request for ${method}`);
    this.stdout.emit(JSON.stringify({
      id: responseId ?? request.id,
      ok: true,
      result: result ?? {},
    }) + "\n");
  }

  emitStdout(value: string): void {
    this.stdout.emit(value);
  }

  emitStderr(value: string): void {
    this.stderr.emit(value);
  }

  emitError(error: Error): void {
    this.spawnError = error;
    for (const listener of this.errorListeners) {
      listener(error);
    }
  }

  emitExit(code = 0): void {
    this.exitCode = code;
    for (const listener of this.exitListeners) {
      listener();
    }
  }

  kill(signal?: string): boolean {
    this.signals.push(signal ?? "SIGTERM");
    this.emitExit(0);
    return true;
  }

  on(event: string, listener: (...args: any[]) => void): void {
    if (event === "exit") {
      this.exitListeners.push(listener as () => void);
    }
    if (event === "error") {
      this.errorListeners.push(listener as (error: Error) => void);
    }
  }
}

export function recordRequest(child: FakeSidecarProcess): void {
  const originalWrite = child.stdin.write.bind(child.stdin);
  child.stdin.write = (data: string): boolean => {
    const payload = JSON.parse(data.replace(/\n$/, "")) as {
      id: string;
      method: string;
      params: Record<string, unknown>;
    };
    child.requests.push(payload);
    return originalWrite(data);
  };
}

export function fakeConfig(overrides: Record<string, string> = {}) {
  return {
    pythonExecutable: "/tmp/python",
    skillRoot: "/tmp/skills",
    retrievalCards: "/tmp/cards.jsonl",
    denseIndex: "/tmp/dense.json",
    maxSearchesPerTurn: 3,
    requestTimeoutMs: 250,
    handshakeTimeoutMs: 250,
    shutdownTimeoutMs: 250,
    ...overrides,
  };
}

export class RecordingSidecar {
  public calls: Array<{ method: string; params: Record<string, unknown> }> = [];
  public ready = false;
  public responses = new Map<string, unknown>();
  public stopped = false;

  private call(method: string, params: Record<string, unknown>): unknown {
    this.calls.push({ method, params });
    if (method === "handshake") {
      return {
        protocol_version: "sidecar-protocol-v0.1",
        readiness: { ready: true },
      };
    }
    if (["begin_turn", "end_turn", "restore_state", "export_state", "mark_all_skill_bodies_evicted", "context_snapshot"].includes(method)) {
      return this.responses.get(method) ?? {};
    }
    if (!this.responses.has(method)) {
      throw new Error(`no scripted response for ${method}`);
    }
    return this.responses.get(method);
  }

  async start(): Promise<{ protocol_version: string; readiness: { ready: boolean } }> {
    this.ready = true;
    this.call("handshake", {});
    return {
      protocol_version: "sidecar-protocol-v0.1",
      readiness: { ready: true },
    };
  }

  async request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return this.call(method, params) as T;
  }

  async shutdown(): Promise<void> {
    this.stopped = true;
  }
}

export class FakePi {
  public handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
  public tools = new Map<string, any>();
  public entries: Array<{ type?: string; customType?: string; data?: unknown }> = [];
  public notifications: string[] = [];

  on(event: string, handler: (event: any, ctx: any) => unknown): void {
    const list = this.handlers.get(event) ?? [];
    list.push(handler);
    this.handlers.set(event, list);
  }

  registerTool(tool: any): void {
    this.tools.set(tool.name, tool);
  }

  appendEntry(customType: string, data?: unknown): { type: string; customType: string; data: unknown } {
    const entry = { type: "custom", customType, data };
    this.entries.push(entry);
    return entry;
  }

  async emit(event: string, payload: any, ctx: any): Promise<unknown> {
    let result = payload;
    for (const handler of this.handlers.get(event) ?? []) {
      result = await handler(result, ctx);
    }
    return result;
  }
}

export class FakeContext {
  public notifications: string[] = [];
  public entries: Array<{ type?: string; customType?: string; data?: unknown }> = [];

  get ui() {
    const context = this;
    return {
      notify(message: string): void {
        context.notifications.push(message);
      },
    };
  }

  get sessionManager() {
    const context = this;
    return {
      getEntries(): Array<{ type?: string; customType?: string; data?: unknown }> {
        return context.entries;
      },
    };
  }
}

export const typeboxStub = {
  Type: {
    Object: (fields: Record<string, unknown>, options?: Record<string, unknown>) => ({ fields, options }),
    Optional: (schema: unknown) => ({ optional: schema }),
    String: (options?: Record<string, unknown>) => ({ string: options }),
    Integer: (options?: Record<string, unknown>) => ({ integer: options }),
    Array: (schema: unknown, options?: Record<string, unknown>) => ({ array: schema, options }),
  },
};
