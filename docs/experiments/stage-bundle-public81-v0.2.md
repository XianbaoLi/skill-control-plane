# Experiment — Public-81 Stage Bundle Baseline

Date: 2026-09-08

## Purpose

Test the first concrete Stage Capability Bundle hypothesis:

> Can one high-recall retrieval pass preserve useful future capability directions on a compact Capability Shelf, and can later raw runtime evidence reuse those directions instead of rebuilding capability awareness from scratch?

This experiment intentionally excludes SRC and additional LLM interpretation. It isolates:

```text
Initial task / raw runtime evidence
        ↓
BM25 Top-5 ∪ Dense Top-5
        ↓
Candidate Pool
        ↓
Category-based Bundle baseline
        ↓
Active Bundle + Capability Shelf
```

## Corpus status

This run is **exploratory, not an exact Gold reproduction**.

The original Stage Gold was created against a local 87-Skill Hermes snapshot:

```text
gold snapshot:
1b7e524ea167fb9057423e999d94ba920e6f11817f7ce06fede3a46722bd4bf6
```

The reproducible public Hermes Skill tree at commit
`d5fd2e93666d20616dca932a488c64ddace2b917` contains 81 Skills:

```text
public snapshot:
e2a1a87b8ebe69cf5d8204b2ac2f6d75046ceb0ba62b43767604ff6c51459e7e
```

Therefore every number below is labeled `public_subset_exploratory`.
It is valid for inspecting architecture behavior, but it must not be reported as
the calibrated 87-Skill benchmark.

## Configuration

| Setting | Value |
|---|---|
| Cases | 2 |
| Stages | 6 |
| BM25 | metadata-only |
| Dense | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Candidate pool | BM25 Top-5 ∪ Dense Top-5 |
| Max Shelf bundles per retrieval | 4 |
| Max candidate Skills per bundle | 4 |
| Bundle baseline | Skill category, then tag fallback |
| S1 query | initial task |
| S2+ query | current raw runtime evidence only |
| SRC | disabled |
| Extra LLM | disabled |

## Aggregate result

| Metric | Result |
|---|---:|
| Initial Shelf future **Skill** recall | 0 / 4 = **0.000** |
| Initial Shelf future **Bundle** recall | 1 / 3 = **0.333** |
| Later Shelf reuse rate | **0.500** |
| New Bundle rate | **0.500** |
| Mean Active required recall | **0.389** |
| Mean Shelf required recall | **0.444** |

The central signal is the difference between exact Skill recall and capability-direction recall:

```text
future exact Skill already on initial Shelf:  0%
future capability group already on Shelf:    33%
```

This supports the user's original reason for keeping a compact Shelf: the first
retrieval can preserve useful higher-level capability directions even when it
does not predict the exact later Skill.

However, the low required-Skill recall shows that **the current category-based
Bundle implementation is not yet a usable activation policy**.

## Case ST-03 — Inbox → Calendar

### S1

```text
active: email
shelf:  email, github, note-taking, apple
required recall: 0.500
```

The initial task correctly activates the email domain, but the Shelf does not
surface the later `productivity` direction containing `google-workspace`.

### S2

```text
active: email, apple
shelf:  email, github, note-taking, apple, autonomous-ai-agents
required recall: 0.000
```

Raw evidence explicitly introduces a meeting/calendar mutation, yet
`google-workspace` is not retained by the category baseline.

This is not evidence that Stage rerouting is unnecessary. It exposes a Bundle
construction problem: global candidate ordering plus a four-category budget can
discard the relevant category before Bundle selection.

## Case ST-01 — GitHub issue → Debugging → Debugger → Review

### Initial foresight

```text
future exact Skill recall:   0.000
future Bundle recall:        0.500
initial Bundle hit:          github
```

This is the clearest positive Shelf signal.

The initial GitHub task does not pre-retrieve the exact later review Skill, but
it does preserve the GitHub capability direction that becomes useful again at
the final PR-review stage.

### Trajectory

```text
S1 active=['github']
   shelf=['github', 'devops']
   required recall=1.000

S2 active=['github']
   shelf adds software-development among others
   required recall=0.500

S3 active=['github', 'note-taking']
   software-development is known but not activated
   required recall=0.667

S4 active=['github', 'note-taking']
   github is already known
   required recall=0.000
```

This trajectory exposes two distinct missing mechanisms.

## Finding 1 — Shelf is useful, but exact prefetch is not the objective

The initial Shelf did not predict any of the four exact future Skills.

That is acceptable if the Shelf's purpose is:

> expose a small set of plausible capability directions to the main Agent.

The Bundle-level metric already shows one such future direction before it is
needed.

Therefore future evaluation must keep both metrics:

- exact future Skill recall;
- future capability/Bundle recall.

Optimizing only exact Skill prefetch would push the design back toward a large,
expensive one-shot candidate set.

## Finding 2 — Category is too coarse to be the final Bundle ontology

Current grouping treats all Skills under folders such as:

```text
software-development
github
productivity
```

as a single capability Bundle.

This is useful as a deterministic baseline, but it conflates distinct runtime
capabilities. For example, `software-development` can contain debugging,
testing, plugin authoring, inspection, and refactoring.

Next Bundle formation should therefore optimize a semantic objective closer to:

```text
relevance
+ coverage
+ complementarity
- redundancy
- context cost
```

rather than inherit the filesystem category as the final capability boundary.

## Finding 3 — Existing Bundle awareness must enable bundle-local retrieval

The strongest architectural finding comes from ST-01 S4.

The system already knows the `github` Bundle, but the exact later
`github-code-review` Skill is not recovered.

Therefore maintaining a Shelf only as a label is insufficient.

When new evidence matches a known Bundle, the next path should be:

```text
new evidence
    ↓
match existing Shelf Bundle
    ↓
bundle-local Skill retrieval / expansion
    ↓
activate newly relevant member
```

and only if no existing Bundle matches should the control plane fall back to
full global discovery.

This makes the Shelf operational rather than merely descriptive.

## Finding 4 — Delta activation policy is currently wrong

The current baseline activates only the best group returned by each new global
retrieval pass.

In ST-01, `software-development` becomes visible on the Shelf during debugging
but is not activated, while unrelated groups can become active later.

The next implementation must separate:

1. **Bundle discovery** — which capability groups exist on the Shelf;
2. **Bundle matching** — which existing/new group best explains the new evidence;
3. **Skill expansion inside the matched group**;
4. **activation**.

The highest-ranked global candidate group should not automatically be treated as
the correct activation decision.

## Revised runtime hypothesis after Experiment 1

```text
Initial Task
    ↓
Global Retrieval
    ↓
Capability Shelf
    ├─ Active Bundle
    └─ compact alternative Bundles
    ↓
Work
    ↓
New Evidence
    ↓
Does Evidence match an existing Shelf Bundle?
    ├─ YES
    │    ↓
    │ Bundle-local retrieval
    │    ↓
    │ expand / activate member Skills
    │
    └─ NO
         ↓
      Global Retrieval
         ↓
      discover new Bundle
```

The optional RAW → Agent reasoning → SRC retrieval escalation remains below
both local and global retrieval paths. SRC changes query representation; it does
not change Bundle lifecycle.

## Next experiment

The next clean A/B should compare:

### A — Current baseline

```text
every capability gap → global retrieval → category grouping
```

### B — Shelf-aware hierarchical retrieval

```text
evidence → existing Bundle match?
          ├─ yes → local retrieval inside Bundle
          └─ no  → global discovery
```

Measure:

- required Skill recall;
- future Bundle recall;
- Shelf reuse;
- global retrieval rate;
- average search corpus size;
- active surface growth;
- wrong-Bundle activation.

Only after this retrieval hierarchy is stable should SRC/LLM escalation be
evaluated as the next independent variable.
