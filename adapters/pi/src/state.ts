export const STATE_ENTRY_TYPE = "skill-control-plane/state-v1";

export function isStateSnapshot(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    record.version === "state-snapshot-v1" &&
    typeof record.store_fingerprint === "string" &&
    Array.isArray(record.bundles)
  );
}
