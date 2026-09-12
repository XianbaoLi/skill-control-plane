export type Environment = Record<string, string | undefined>;

export type AdapterConfig = {
  pythonExecutable: string;
  skillRoot: string;
  retrievalCards: string;
  denseIndex: string;
  maxSearchesPerTurn: number;
  requestTimeoutMs: number;
  handshakeTimeoutMs: number;
  shutdownTimeoutMs: number;
};

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const DEFAULT_HANDSHAKE_TIMEOUT_MS = 30_000;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000;

function required(env: Environment, name: string): string {
  const value = env[name]?.trim();
  if (!value) {
    throw new Error(`Skill Control Plane adapter requires ${name}`);
  }
  return value;
}

function optionalInteger(env: Environment, name: string, fallback: number): number {
  const raw = env[name]?.trim();
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

export function adapterConfigFromEnvironment(env: Environment = process.env): AdapterConfig {
  return {
    pythonExecutable: env.SKILL_CONTROL_PLANE_PYTHON?.trim() || "python3",
    skillRoot: required(env, "SKILL_CONTROL_PLANE_SKILL_ROOT"),
    retrievalCards: required(env, "SKILL_CONTROL_PLANE_RETRIEVAL_CARDS"),
    denseIndex: required(env, "SKILL_CONTROL_PLANE_DENSE_INDEX"),
    maxSearchesPerTurn: optionalInteger(
      env,
      "SKILL_CONTROL_PLANE_MAX_SEARCHES_PER_TURN",
      3,
    ),
    requestTimeoutMs: optionalInteger(env, "SKILL_CONTROL_PLANE_REQUEST_TIMEOUT_MS", DEFAULT_REQUEST_TIMEOUT_MS),
    handshakeTimeoutMs: optionalInteger(
      env,
      "SKILL_CONTROL_PLANE_HANDSHAKE_TIMEOUT_MS",
      DEFAULT_HANDSHAKE_TIMEOUT_MS,
    ),
    shutdownTimeoutMs: optionalInteger(env, "SKILL_CONTROL_PLANE_SHUTDOWN_TIMEOUT_MS", DEFAULT_SHUTDOWN_TIMEOUT_MS),
  };
}
