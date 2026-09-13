# Generic capability-gap checkpoint architecture (v0.1)

## Audit and insertion point

The Pi adapter owns the user-turn lifecycle: `agent_start` calls
`begin_turn`, `before_agent_start` injects the compact runtime policy and Bundle
Cards, and `agent_end` calls `end_turn`.  Its `context` projection updates
evicted-body markers.  Native tool results and runtime evidence are already
visible in Pi's message stream; benchmark instrumentation observes the same
stream (`tool_execution_end` / `message_end`) and emits `evidence_emitted`.
`load_capability` maps to the existing `search_capability` sidecar method, and
`apply_capability` maps to `DiscoverySession.apply` then `CapabilityMemory.commit`.

The correct checkpoint is therefore **after a new runtime result/evidence has
entered the Pi conversation, before the next model inference**.  It belongs in
the adapter's event/context boundary, not in `DiscoverySession`, retrieval, or
the Pi agent loop.  That preserves one retrieval path and avoids a fork of Pi.

## Proposed runtime flow

1. A generic observer fingerprints newly visible runtime evidence.  Eligible
   reasons are a newly observed failure, verifier/test/check failure, a tool
   error exposing a previously unseen problem, or an explicitly identified new
   subgoal.  Ordinary successful tool results do not qualify.
2. For a fingerprint not already checked during the turn, inject a compact
   checkpoint instruction containing the evidence and current Bundle Cards.
3. The model must respond through a small `capability_gap_check` decision:
   `{ "needs_capability": boolean, "need": string? }`.
4. `false` resumes normally.  `true` uses the existing `load_capability(need)`
   and `apply_capability` path; it does not call a second retriever.
5. Cache `(evidence_fingerprint, normalized_need)` for the turn.  A duplicate
   evidence item cannot schedule another identical checkpoint or search.

The decision tool should validate that `need` is absent when false and a concise
non-empty string when true.  Its result should state whether a search was
started, allowing a model to recover rather than blindly retry.

## Telemetry contract

Emit `capability_gap_check` for every eligible, de-duplicated checkpoint with:

- `trigger_reason`
- `evidence_fingerprint` (and source event ID when available)
- `active_skill_ids`
- `needs_capability`
- `generated_need`
- `search_started`

The trace can then distinguish, in order: `evidence_observed`,
`gap_check_triggered`, `gap_detected`, `post_evidence_search`,
`target_in_candidates`, `target_selected`, and `target_committed`.  Existing
activation and body-load events remain the source of truth for the final two
runtime effects.

## Boundary conditions

This design never derives a query directly from arbitrary evidence and has no
task/event-ID special cases.  It asks the model only whether active capabilities
are sufficient; the normal discovery and memory invariants continue to decide
candidates, deduplication, commit, and body loading.
