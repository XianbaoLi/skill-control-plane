# Sidecar Protocol v0.1

The Python sidecar exposes the public `SkillControlPlane` API over private
stdio NDJSON pipes. It does not implement a Pi adapter, persist Pi session
entries, or know a Pi session schema.

## Dependency direction

```text
Pi Adapter (future)
    ↓ private stdio pipes
skill_control_plane.sidecar
    ↓
SkillControlPlane public API
    ↓
runtime → discovery → registry
```

## Protocol

The protocol version is `sidecar-protocol-v0.1`.

V0.1 is single process, single client, and strictly sequential. There is no
concurrency, batching, JSON-RPC notification, socket transport, or automatic
reconnect. Every input line is one UTF-8 JSON object, and every request gets
exactly one response line. `stdout` carries protocol NDJSON only; diagnostics
and tracebacks go to `stderr`.

Request:

```json
{"id":"request-id","method":"method_name","params":{}}
```

Success:

```json
{"id":"request-id","ok":true,"result":{}}
```

Error:

```json
{"id":"request-id","ok":false,"error":{"code":"STABLE_ERROR_CODE","message":"...","details":{}}}
```

The stable error codes are `INVALID_REQUEST`, `INVALID_PARAMS`,
`UNKNOWN_METHOD`, `CORE_VALIDATION_ERROR`, `NOT_READY`, and
`INTERNAL_ERROR`. Malformed JSON returns `id: null` when no valid request ID
can be parsed. Core validation failures do not terminate the server, and
unexpected failures return `INTERNAL_ERROR` without a traceback in `stdout`.

## Methods

Supported methods are:

`handshake`, `begin_turn`, `end_turn`, `context_snapshot`,
`search_capability`, `apply_capability`, `load_skill_body`,
`mark_skill_body_evicted`, `mark_all_skill_bodies_evicted`, `turn_audit`,
`export_state`, `restore_state`, and `shutdown`.

`handshake` must be the first Adapter call. Its result contains
`protocol_version`, the DTO from `SkillControlPlane.readiness()`, and
`supported_methods`. If production initialization fails, the sidecar stays
alive as a minimal protocol server, `readiness.ready` is false with a
structured reason, and every runtime method returns `NOT_READY`. There is no
BM25-only fallback and no sidecar-side Dense/RRF readiness decision.

Method parameters map directly to the corresponding Core DTOs. In particular,
`apply_capability` strictly decodes ordinary JSON into `CapabilityDecision`,
including every `CoverageClaim`, and rejects unknown fields.
`restore_state` strictly decodes ordinary JSON into `StateSnapshotV1` →
`StateSnapshotBundleV1` → `StateSnapshotMemberV1`; Core `restore_state()`
remains the authority for semantic validation and atomic replacement.

Core dataclasses serialize as JSON objects, enums and literals as strings, and
tuples as arrays. `shutdown` returns a success response and then exits; EOF
also exits normally.

## State and persistence boundary

`export_state` and `restore_state` expose only versioned
`state-snapshot-v1` Capability Memory: Bundle IDs, purposes, member roles, and
resident/evicted body state. The sidecar does not append Pi entries, read Pi
sessions, or own resume orchestration. A future Pi Adapter is responsible for
placing the returned JSON in a Pi custom entry and later passing that JSON to
`restore_state`.
