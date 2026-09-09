# Audited Stage-transition additions v0.3

## Purpose

This set adds **10 positive post-S1 `new_required` transitions** across **7 cases**.
The targets are intentionally unique and avoid duplicating the three existing positive
targets (`google-workspace`, `python-debugpy`, `github-code-review`).

The audit rule is strict:

- `new_required`: the newly emerged capability is actually needed to satisfy the current stage.
- `useful`: helpful methodology/context, but progress is still possible without loading it.
- `hard_negative`: semantically nearby and therefore likely to be retrieved, but wrong for the resolved runtime need.
- Runtime evidence describes observable facts and user requests; it never names the Gold target Skill.

## Added targets

| Transition | New required Skill | Why it becomes required | Main hard negative(s) |
|---|---|---|---|
| `AT-01/S2` | `document-to-action-items` | Plain extraction has finished; the user now requests cited obligations/deadlines/actions. | ocr-and-documents, pdf |
| `AT-01/S3` | `notion` | Structured actions are ready and the user explicitly approves writing them to a Notion tracker. | — |
| `AT-02/S2` | `ocr-and-documents` | Runtime reveals an image-only scanned PDF with no text layer; local OCR is now necessary. | pdf |
| `AT-03/S2` | `himalaya` | Runtime resolves the mailbox to a configured terminal IMAP/SMTP account. | google-workspace |
| `AT-04/S2` | `email-inbox-triage` | A one-message lookup expands into thread prioritization and safe reply drafting. | google-workspace |
| `AT-05/S2` | `obsidian` | The user requests a Markdown vault copy plus an Obsidian wikilink. | apple-notes, llm-wiki |
| `AT-05/S3` | `llm-wiki` | The user upgrades the folder into a maintained research wiki with schema/index/log/provenance/contradictions. | apple-notes |
| `AT-06/S2` | `apple-notes` | On macOS, the user explicitly requests a Notes.app copy for iPhone/iCloud sync. | llm-wiki |
| `AT-07/S2` | `systematic-debugging` | The user explicitly requires a root-cause-first red/green debugging process; debugger stepping is not yet needed. | python-debugpy, github-issue-to-pr |
| `AT-07/S3` | `github-issue-to-pr` | A confirmed root cause becomes a live GitHub issue that must be implemented through a verified PR. | github-code-review |

## Coverage

The 10 added targets are:

1. `document-to-action-items`
2. `notion`
3. `ocr-and-documents`
4. `himalaya`
5. `email-inbox-triage`
6. `obsidian`
7. `llm-wiki`
8. `apple-notes`
9. `systematic-debugging`
10. `github-issue-to-pr`

Together with the existing three positive runtime targets, `stage-transition-v0.3.jsonl`
contains **13 positive post-S1 transitions** across **9 cases**.

This is materially better for the field-ablation and paraphrase-robustness experiments,
but it is still below the current formal explanatory threshold of 20 target transitions.
Treat 13 as an intermediate evaluation set rather than the final statistical set.

## Special audit decisions

### `systematic-debugging`

Do **not** label this Skill required merely because a bug exists. In the new case it
becomes required only after the user explicitly asks for the root-cause-first process:
tight red/green feedback loop, investigation before fixes, and no symptom patching.

### `python-debugpy`

It is intentionally a hard negative in `AT-07/S2`. The failure is already deterministic
and reproducible, and the user has not asked to step through state, attach to a process,
or inspect mutation interactively. That preserves the earlier distinction between
general systematic debugging and debugger-specific capability.

### Connector vs workflow Skills

`himalaya` and `email-inbox-triage` are separated deliberately:

- Himalaya owns terminal mailbox operations.
- Inbox Triage owns prioritization, thread classification, and safe reply-drafting policy.

This creates reciprocal transitions that test whether the representation captures
the difference between a provider/connector capability and a higher-level workflow.

### `obsidian` vs `llm-wiki`

These are also separated deliberately:

- Obsidian owns filesystem-first vault note operations and wikilinks.
- LLM Wiki owns persistent knowledge-base maintenance: schema, index, log, provenance,
  cross-references, contradiction handling, and compounding updates.

The `AT-05` chain tests exactly when the need crosses from ordinary note editing into
knowledge-base maintenance.
