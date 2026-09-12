import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { AdapterConfig } from "./config.ts";
import type { Handshake, Readiness, SidecarResponse } from "./types.ts";

export type StreamLike = {
  on(event: string, listener: (chunk: Buffer) => void): void;
  write(data: string): unknown;
  end(): void;
};

export type ProcessLike = {
  pid?: number;
  exitCode: number | null;
  stdin: StreamLike;
  stdout: StreamLike;
  stderr: StreamLike;
  kill(signal?: string): boolean;
  on(event: string, listener: (...args: any[]) => void): void;
};

export type SpawnFunction = (
  command: string,
  args: string[],
  options: Record<string, unknown>,
) => ProcessLike;

export class SidecarError extends Error {
  readonly code: string;

  constructor(code: string, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "SidecarError";
    this.code = code;
  }
}

type PendingRequest = {
  method: string;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const PROTOCOL_VERSION = "sidecar-protocol-v0.1";

export class SidecarClient {
  readonly config: AdapterConfig;
  private readonly spawnFn: SpawnFunction;
  private readonly stderrLines: string[] = [];
  private readonly pending = new Map<string, PendingRequest>();
  private process: ProcessLike | null = null;
  private stdoutBuffer = "";
  private connected = false;
  private shuttingDown = false;
  private stopped = false;
  private shutdownPromise: Promise<void> | null = null;
  private handshake: Handshake | null = null;

  constructor(config: AdapterConfig, spawnFn: SpawnFunction = spawn as unknown as SpawnFunction) {
    this.config = config;
    this.spawnFn = spawnFn;
  }

  get ready(): boolean {
    return this.connected && this.handshake?.readiness.ready === true;
  }

  get protocolVersion(): string | undefined {
    return this.handshake?.protocol_version;
  }

  get readiness(): Readiness | undefined {
    return this.handshake?.readiness;
  }

  get stderr(): readonly string[] {
    return [...this.stderrLines];
  }

  async start(): Promise<Handshake> {
    if (this.handshake) {
      return this.handshake;
    }
    if (this.stopped) {
      throw new SidecarError("NOT_READY", "sidecar has already stopped");
    }

    const child = this.spawnFn(
      this.config.pythonExecutable,
      [
        "-m",
        "skill_control_plane.sidecar",
        "--skill-root",
        this.config.skillRoot,
        "--retrieval-cards",
        this.config.retrievalCards,
        "--dense-index",
        this.config.denseIndex,
        "--max-searches-per-turn",
        String(this.config.maxSearchesPerTurn),
      ],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    this.process = child;
    this.connected = true;

    child.stdout.on("data", (chunk: Buffer) => {
      this.stdoutBuffer += chunk.toString("utf-8");
      let newline = this.stdoutBuffer.indexOf("\n");
      while (newline >= 0) {
        const rawLine = this.stdoutBuffer.slice(0, newline).replace(/\r$/, "");
        this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
        this.handleStdoutLine(rawLine);
        newline = this.stdoutBuffer.indexOf("\n");
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf-8");
      this.stderrLines.push(...text.split(/\r?\n/).filter(Boolean));
      if (this.stderrLines.length > 200) {
        this.stderrLines.splice(0, this.stderrLines.length - 200);
      }
    });
    child.on("error", (error: Error) => {
      this.failPending(new SidecarError("INTERNAL_ERROR", "sidecar process error", { cause: error }));
    });
    child.on("exit", () => {
      this.process = null;
      this.connected = false;
      this.handshake = null;
      this.failPending(new SidecarError("NOT_READY", "sidecar process exited"));
    });

    try {
      const handshake = await this.request<Handshake>(
        "handshake",
        {},
        this.config.handshakeTimeoutMs,
      );
      if (handshake.protocol_version !== PROTOCOL_VERSION) {
        throw new SidecarError(
          "INVALID_REQUEST",
          `sidecar protocol mismatch: expected ${PROTOCOL_VERSION}, got ${handshake.protocol_version}`,
        );
      }
      if (handshake.readiness.ready !== true) {
        throw new SidecarError("NOT_READY", "sidecar is not ready");
      }
      this.handshake = handshake;
      return handshake;
    } catch (error) {
      await this.shutdown();
      throw error;
    }
  }

  async request<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs: number = this.config.requestTimeoutMs,
  ): Promise<T> {
    if (!this.connected || this.process === null) {
      throw new SidecarError("NOT_READY", "sidecar is not ready");
    }

    const id = randomUUID();
    const promise = new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new SidecarError("TIMEOUT", `sidecar request timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, {
        method,
        resolve: resolve as (value: unknown) => void,
        reject,
        timer,
      });
    });

    const request = `${JSON.stringify({ id, method, params })}\n`;
    try {
      const accepted = this.process.stdin.write(request);
      if (!accepted) {
        // Node will still emit data for queued writes; drain is not protocol state.
      }
    } catch (error) {
      this.rejectRequest(id, new SidecarError("INTERNAL_ERROR", "failed to write sidecar request", { cause: error }));
    }
    return promise;
  }

  async shutdown(): Promise<void> {
    if (this.shutdownPromise) {
      return this.shutdownPromise;
    }
    if (this.stopped) {
      return;
    }
    this.shuttingDown = true;
    this.shutdownPromise = this.shutdownNow();
    return this.shutdownPromise;
  }

  private async shutdownNow(): Promise<void> {
    const child = this.process;
    if (child === null) {
      this.stopped = true;
      return;
    }

    try {
      await this.request("shutdown", {}, this.config.shutdownTimeoutMs);
    } catch {
      child.kill("SIGTERM");
    }
    try {
      child.stdin.end();
    } catch {
      // Process already exited.
    }

    const exited = new Promise<void>((resolve) => {
      if (child.exitCode !== null) {
        resolve();
        return;
      }
      child.on("exit", () => resolve());
    });
    const timeout = new Promise<void>((resolve) => {
      setTimeout(resolve, this.config.shutdownTimeoutMs);
    });
    await Promise.race([exited, timeout]);
    if (child.exitCode === null) {
      child.kill("SIGKILL");
    }
    this.stopped = true;
  }

  private handleStdoutLine(rawLine: string): void {
    if (!rawLine.trim()) {
      return;
    }
    let response: SidecarResponse;
    try {
      const parsed = JSON.parse(rawLine);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("response is not an object");
      }
      response = parsed as SidecarResponse;
    } catch (error) {
      this.failPending(new SidecarError("INVALID_REQUEST", "sidecar stdout is not valid NDJSON", { cause: error }));
      return;
    }
    if (typeof response.id !== "string" || !this.pending.has(response.id)) {
      return;
    }
    const pending = this.pending.get(response.id)!;
    this.pending.delete(response.id);
    clearTimeout(pending.timer);
    if (response.ok) {
      pending.resolve(response.result ?? {});
      return;
    }
    const code = response.error?.code ?? "INTERNAL_ERROR";
    const message = response.error?.message ?? "sidecar request failed";
    pending.reject(new SidecarError(code, message, { cause: response.error?.details }));
  }

  private rejectRequest(id: string, error: Error): void {
    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }
    this.pending.delete(id);
    clearTimeout(pending.timer);
    pending.reject(error);
  }

  private failPending(error: Error): void {
    for (const [id, pending] of [...this.pending]) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pending.delete(id);
    }
  }
}
