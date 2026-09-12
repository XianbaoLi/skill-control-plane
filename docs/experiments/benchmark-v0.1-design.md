# Pi Native Skills vs Skill Control Plane benchmark v0.1

## Scope and fairness

Version v0.1 compares Pi Native Skills with Pi started with native discovery disabled
plus the Pi Adapter and Skill Control Plane. Every pair holds Pi version, provider,
model, prompt bytes, copied fixture, full package corpus, tool allowlist, 20-turn cap,
and 300-second timeout fixed. Prompts do not name required Skills. There is no human
intervention and no aggregate subjective score.

The runner exposes only `one`; it cannot accidentally launch the 96-cell scaling or
40-cell runtime matrix. Frozen corpora, Retrieval Cards, DenseIndex, fusion settings,
Bundle lifecycle, and Pi's agent loop are read-only inputs.

## Scaling tasks

The same 16 prompts run unchanged at S32/S64/S128. All targets are members of all
three nested subsets.

| Task | Category | Target |
|---|---|---|
| SC-01 | agent-meta | requirements-analysis |
| SC-02 | agent-meta | source-driven-development |
| SC-03 | ai-ml | fine-tuning-expert |
| SC-04 | animation-webgl | threejs-webgl |
| SC-05 | animation-webgl | gsap-react |
| SC-06 | backend-apis | api-design |
| SC-07 | backend-apis | fastify |
| SC-08 | databases-data | rag |
| SC-09 | design-ux | create-website |
| SC-10 | devops-cloud | terraform-engineer |
| SC-11 | devops-cloud | monitoring |
| SC-12 | frontend-ui | vite |
| SC-13 | languages | golang-pro |
| SC-14 | marketing-growth | seo |
| SC-15 | marketing-growth | storybrand-messaging |
| SC-16 | testing-quality | playwright-expert |

Selection favors executable or structured artifacts and intentionally includes close
framework, lifecycle, marketing, and planning distractors. It spans all 11 frozen
corpus categories; no task was selected from Retrieval Card wording.

## Runtime scenarios

S128 is fixed. Single-skill scenarios are RT-S01..S06. Multi-skill scenarios are
RT-M01..M05 and require 2–3 Skills. Stage transitions are RT-T01..T05; each fixture
withholds its second failure until the first stage's checks advance. Reuse/session
scenarios are RT-R01..R04, each with two explicit turns and a resume instruction.

| Kind | Count | Coverage |
|---|---:|---|
| single-skill | 6 | API contract, observability, Go, messaging, browser tests, retrieval |
| multi-skill | 5 | service+browser, site+build+search, contract+retrieval, IaC+observability, requirements+training |
| stage-transition | 5 | API→alerts, build→browser, retrieval→contract, IaC→Go helper, site→search |
| reuse/session | 4 | API contract, alerts, browser page object, brand narrative |

Only treatment has CREATE/EXTEND/reuse expectations. Scoring remains based on files,
commands, stage evidence, and observed tool events.

## Leakage and validation

`benchmark_validate.py` checks exact counts, S32/S64/S128 membership, legal required
sets, stage-specific requirements and trigger evidence, multi-turn reuse shape, and
case-insensitive token-boundary occurrences of every target/required Skill ID in the
prompt. All 36 prompts pass. Target metadata remains in the gold manifest and is never
placed in the user prompt.

## Result and cost telemetry

Each JSONL row records identity, boolean outcome with check details, required and
activated Skills, recall, wrong activations, discovery/repeated-discovery/body-load
counts, transition/Bundle/session fields, Pi input/output/cache usage, embedding call
count, elapsed milliseconds, tool calls, and Control Plane retrieval telemetry.
Native-only non-concepts are zero or null. `llm_total_tokens` is input plus output;
provider cache reads remain separately in `cached_tokens`.

Online token data comes directly from every Pi assistant message's `usage` object
(`input`, `output`, `cacheRead`). Control Plane tool start/end events provide discovery,
CREATE/EXTEND, and fused rank telemetry. The current sidecar exposes fused RRF rank and
component scores but not standalone BM25/Dense ranks; those rank fields are therefore
null with `component_rank_status=unavailable-from-current-sidecar-trace`, never
fabricated. A future telemetry-only adapter change may expose them without changing
retrieval behavior.

Offline cost is a separate artifact schema. Retrieval Card records support model,
input/output/total tokens, request/retry counts, and wall time. Dense records support
embedding model, call count, provider input amount plus unit (tokens or characters),
wall time, and vector count. The historical S128 metadata lacks complete token/request
telemetry, so `offline-cost.json` marks those values unavailable while retaining the
known 128 vector count. Regeneration runners must populate these fields from provider
responses; dollar pricing is intentionally post-processing.

## Launch paths and smoke limitation

Native uses Pi SDK `DefaultResourceLoader` with native Skills enabled and every
`SKILL.md` in the selected corpus passed as an explicit path. This is the programmatic
equivalent of normal Pi Skills startup.

Control Plane uses Pi SDK `DefaultResourceLoader({noSkills: true})`, loads the real Pi
Adapter extension, and starts the production NDJSON Sidecar/Control Plane with the
selected corpus, Retrieval Cards, and DenseIndex. This is the programmatic equivalent
of `pi --no-skills` plus the adapter.

Only S128 has a separately materialized frozen retrieval artifact directory. For S32
smoke startup, the bridge creates a temporary exact subset *view* under that run's
working fixture by selecting unchanged S128 card lines and DenseIndex records for S32
IDs. It neither writes nor regenerates the frozen artifacts.

External query embedding was not approved because the smoke prompt would leave the
machine. Consequently, the smoke-only sidecar substitutes a local zero query vector.
The real Pi 0.84.1 session, tools, adapter, sidecar, Control Plane, frozen vectors, log,
and scorer all execute; retrieval quality and the resulting token deltas are explicitly
not benchmark evidence.

Smoke commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_run.py one --task SC-01 --arm native --corpus S32 --output /tmp/scp-benchmark-smoke
PYTHONPATH=src .venv/bin/python scripts/benchmark_run.py one --task SC-01 --arm control-plane --corpus S32 --output /tmp/scp-benchmark-smoke
PYTHONPATH=src .venv/bin/python scripts/benchmark_run.py one --task RT-S04 --arm native --corpus S128 --output /tmp/scp-benchmark-smoke
PYTHONPATH=src .venv/bin/python scripts/benchmark_run.py one --task RT-S04 --arm control-plane --corpus S128 --output /tmp/scp-benchmark-smoke
```

All four smoke cells produced schema-valid logs and automatic `task_success=true`.
