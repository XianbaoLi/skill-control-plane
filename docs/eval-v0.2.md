# Evaluation Plan V0.2 — Control Plane

V0.2 moves beyond single-Skill retrieval sanity checks and evaluates two
behaviors that matter to a Skill control plane:

1. preserving a complete set of Skills for a compositional task;
2. recovering newly required Skills when the task state changes.

## Gold sets

### Multi-Skill

`evals/gold/multi-skill-v0.2.jsonl`

Each case has one initial task and a required Skill set with at least two Skills.

Primary metrics:

- **required Skill recall** — required Skills recovered across all cases;
- **full required-set coverage** — fraction of tasks for which every required
  Skill is present in the candidate set;
- **average candidate-set size** — downstream context/cost proxy.

### Stage transition

`evals/gold/stage-transition-v0.2.jsonl`

Each case is a task trajectory. Every stage records:

- current observed state;
- transition trigger, when present;
- `required_now`;
- `new_required`.

The first stage is initialization. `new_required` on later stages is the
recovery target created by a state transition.

## Baselines

### One-shot

Retrieve once from the initial task and hold that candidate set fixed for every
later stage.

This approximates routing systems that choose capabilities only at task start.

### Dynamic reroute

At every stage, retrieve again from:

```text
initial task
+ transition trigger
+ current observed state
```

This approximates a control plane that reacts to failures, observations, and
subgoal changes.

## Stage-transition metrics

- **one-shot stage full coverage** — fraction of stages whose full
  `required_now` set is covered by the initial candidate set;
- **reroute stage full coverage** — same metric after stage-aware retrieval;
- **reroute gain** — reroute coverage minus one-shot coverage;
- **new-skill recovery** — fraction of Skills newly required after a transition
  that the reroute candidate set recovers;
- **average reroute candidate-set size** — context/cost proxy.

## What V0.2 does not yet measure

Candidate presence is not activation.

The current repository does not yet contain a real runtime semantic judge that
selects active Skills from the retrieved candidate set. Therefore these metrics
are intentionally unavailable in V0.2 retrieval/reroute evaluation:

- active-Skill precision;
- hard-negative activation rate;
- premature activation rate.

Those require the next layer:

```text
BM25 + Dense
    |
candidate union
    |
runtime LLM judge
    |
active Skills
```

Hard-negative labels remain in Gold so they can be used once that judge exists.

## First calibrated cases

The first reviewed batch contains:

- `MS-01`: OCR -> document action extraction -> Notion;
- `MS-06`: inbox-triage workflow + terminal email connector;
- `ST-01`: issue-to-PR -> systematic debugging -> Python debugger -> PR review;
- `ST-03`: terminal inbox triage -> Calendar mutation.

These are calibration cases, not yet a statistically strong benchmark. Expand
only after the evaluator and label semantics are stable.
