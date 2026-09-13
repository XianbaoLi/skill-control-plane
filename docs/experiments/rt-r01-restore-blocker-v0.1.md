# RT-R01 restore-on blocker diagnosis (v0.1)

The failed run is preserved under `local_artifacts/benchmark-runs/restore-ablation-v0.1-20260913/on`.

## Observed state

T1 retrieved `api-design` at rank 1 but did not commit it. Every
`apply_capability` tool call reached the sidecar with `{}`; consequently no
Bundle was created, no active member existed, and no skill body was resident.
At T1 shutdown the persisted state entry was:

```json
{"version":"state-snapshot-v1", "bundles":[]}
```

T2 therefore restored an empty `CapabilityMemory`: active skill IDs were empty,
Bundle Cards said `none`, and body residency had no entries. `DiscoverySession`
was correctly new for T2 (no pending candidates, zero searches, full budget).
The later `load_skill_body(api-design)` failed because `api-design` was not a
current active Bundle member.

## Root cause and semantics

`CapabilityMemory.load_skill_body` already authorizes from
`CapabilityMemory.state.active_bundles`, not from `DiscoverySession`, candidates,
or adapter cache. Its error was semantically correct for this run; this is not a
restore/lazy-load runtime bug.

The actual fault was a provider compatibility regression introduced by the
action-discriminated TypeBox union: GLM's OpenAI-compatible tool-call path emitted
empty arguments for that union. The provider-facing schema is restored to a plain
object with explicit action-dependent descriptions. The sidecar remains the strict
action-specific validator, and successful applies still return the stable
`affected_bundle_id` plus `bundle_target` card for same-turn EXTEND/DIRECT.

The restored-member invariant is covered directly: an evicted Skill created in
T1, exported, restored into a new control plane and new turn, lazy-loads without
search/apply; an inactive Skill still fails.
