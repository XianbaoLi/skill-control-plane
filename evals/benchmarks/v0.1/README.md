# Pi Native Skills vs Skill Control Plane benchmark v0.1

This directory freezes the benchmark *design*, not benchmark results. `manifest.json`
defines the fair comparison and deliberately disables full-matrix expansion.

- `scaling.jsonl`: 16 prompts reused byte-for-byte at S32, S64, and S128.
- `runtime.jsonl`: 20 S128 scenarios (6 single, 5 multi, 5 dynamic reroute, 4 reuse).
- `fixtures.json`: declarative, copied-per-run working fixtures.
- `schema.json`: online run result contract.
- `offline-cost.schema.json` / `offline-cost.json`: offline artifact-cost contract and
  honest historical availability record.

Validate without network access:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_validate.py
```

The runner accepts exactly one task/arm/corpus cell at a time. It has no full-suite
command:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_run.py one \
  --task SC-01 --arm native --corpus S32 --output /tmp/benchmark-smoke
```

`pi_benchmark_bridge.mjs` is a plumbing smoke bridge, not a benchmark model. It uses
Pi's deterministic faux provider and `benchmark_offline_sidecar.py` uses a local zero
query vector, so its success and token telemetry prove wiring only. Never include its
numbers in benchmark reports.

Formal runs use `--production --provider PROVIDER --model MODEL`, which selects
`pi_benchmark_production_bridge.mjs`. That bridge uses Pi's configured provider,
exposes identical `read`, `write`, `edit`, and `bash` tools in both arms, and does not
receive benchmark gold. The Control Plane arm alone adds its three capability tools.
The production sidecar records the readiness embedding separately from runtime query
embeddings while retaining the production BM25 + Dense + RRF path.

Dynamic reroute annotations are event-based gold: `initial_required_skills` plus
`reroute_events[{event_id, trigger_evidence, new_required_skills}]`. They do not model
runtime state. Host-side gated verifiers emit `BENCHMARK_EVIDENCE:E1` only after the
initial problem is fixed; scoring then requires each new capability activation to
occur after that evidence.

Reuse scenarios have independent T1/T2 checks. The bridge persists and shuts down T1,
opens the same Pi session file, restores adapter state, and only then submits T2.
Per-turn discovery, activation/load, Bundle reuse, and usage are recorded in
`turn_metrics`.

The body-sufficiency rationale for all required Skills is recorded in
`docs/experiments/benchmark-v0.1-body-sufficiency-audit.md`.
