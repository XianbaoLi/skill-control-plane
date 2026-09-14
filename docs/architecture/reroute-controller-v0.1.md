# Runtime RerouteController v0.1

`RerouteController` is the sole runtime capability-reroute mechanism. Pi converts
host/tool events into `RuntimeEvidence`, calls the sidecar, and projects a
`discovered` outcome back to the Main Agent. It does not decide capability gaps,
deduplicate evidence, or initiate retrieval itself.

The layers are deliberately separate:

1. Runtime eligibility decides whether one evidence item warrants reconsidering
   capability.
2. `CapabilityGapDecider.decide_gap(evidence, active_bundle_cards,
   current_subgoal_context)` decides whether a capability is missing and emits a
   concise need.
3. `RerouteController` calls the existing `SkillControlPlane.search_capability`
   path when the decision is positive. This remains the production Retrieval
   Card + BM25 + Dense + RRF implementation.
4. The Main Agent sees evidence ID, need, candidates, and current Bundle Cards,
   then owns candidate selection and DIRECT/EXTEND/CREATE.
5. The Main Agent supplies `reroute_evidence_id` to the existing
   `apply_capability` call. The runtime validates the selection against that
   evidence's Candidates, marks `SELECTED`, applies through the existing
   `DiscoverySession`/`CapabilityMemory` transaction, and marks `COMMITTED`
   only after the apply succeeds.
6. `CapabilityMemory` retains its existing Bundle and Skill-body lifecycle.

## RuntimeEvidence

The schema is `evidence_id`, `kind`, `source`, `text`, `fingerprint`, and optional
`metadata`. Supported kinds are `tool_error`, `test_failure`,
`verifier_failure`, `new_subgoal`, and `host_signal`. It contains no benchmark
markers or benchmark-specific parsing.

## State and lifecycle

Eligible unique evidence follows:

`NEW -> CHECKING -> NO_GAP`

or:

`NEW -> CHECKING -> GAP_FOUND -> DISCOVERED -> SELECTED -> COMMITTED`

The only terminal states are `NO_GAP`, `COMMITTED`, and `FAILED`; there is no
`DONE` state. An exception enters `FAILED` and records `failure_stage` as one of
`gap_decision`, `discovery`, `selection`, or `apply`, plus a bounded
`failure_reason`. State and fingerprint deduplication live for one user turn,
survive tool loops, and are removed by `end_turn`. They are never part of a
`CapabilityMemory` state snapshot.

Host signals are eligible only when their structured metadata has
`capability_relevant: true`. All other supported failure/subgoal kinds are
eligible. Ordinary host signals, duplicate fingerprints, and Control Plane
internal sources are ignored before the decider.

## Telemetry

Each event is keyed by `evidence_id`. The runtime emits evidence observation and
eligibility, every real state transition, gap decision, discovery candidates/ranks
or a typed failure, candidate selection, apply result, activation, and related
body loads. `state_history`, `SELECTED`, and `COMMITTED` are controller-owned
runtime facts; reporters do not infer them from messages or tool order. This is
a structured funnel and does not require reporter-side message scanning.
