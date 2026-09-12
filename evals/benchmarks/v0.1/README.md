# Pi Native Skills vs Skill Control Plane benchmark v0.1

This directory freezes the benchmark *design*, not benchmark results. `manifest.json`
defines the fair comparison and deliberately disables full-matrix expansion.

- `scaling.jsonl`: 16 prompts reused byte-for-byte at S32, S64, and S128.
- `runtime.jsonl`: 20 S128 scenarios (6 single, 5 multi, 5 transition, 4 reuse).
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
