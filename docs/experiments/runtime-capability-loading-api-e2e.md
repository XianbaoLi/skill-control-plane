# Runtime capability loading — live API acceptance

> **Historical / eval-only.** This document preserves evidence or an earlier design; it is not authoritative for the V1 runtime.


## Outcome

The implementation and offline transitions are testable, but live resolver acceptance is blocked by provider HTTP 429. No mock/replay decision was substituted. All three initial runtime states remained unchanged.

## Provenance

- Model: `glm-5.3-flash`, using existing `BigModelChatClient` and configured `.env`.
- Corpus: frozen current87, 87 skills; snapshot `f670e9d5ecdbb299a5c5ac4fcc48ece35e0adadcfb7e24b96850a4ec5379e509`. Existing manifest/card validation passed.
- Initial sandbox attempt failed with URLError. The approved network-enabled run reached the provider and received HTTP 429 for Dense and each of three chat requests.
- Retrieval: real BM25 over existing RetrievalCard representations, k=10; current active bundle BM25, n=3. Dense embedding-3 was attempted and failed with 429.
- No candidate or action was manually injected. Example IDs github/git-read/github-actions are absent from this corpus; the fixture uses actual github-code-review/github-auth members.
- github-pr-workflow is the real CI-equivalent candidate: its SKILL.md includes monitoring checks, GitHub Actions check runs, and fixing CI failures.
- Raw run: `local_artifacts/runtime-capability-loading/live-jky9zreg/report.json`. The separate sandbox-failure run is retained in `live-vzivi20b`.
- No historical output files were edited.

## Intended DIRECT

Need: Read this PDF, extract the useful text, and turn it into a presentation for this one task.

| Rank | Skill candidate | BM25 score | Evidence |
|---|---|---:|---|
| 1 | baoyu-infographic | 16.807071 | matched_terms: this, the, text, and, turn, it, into, a, for |
| 2 | powerpoint | 16.173214 | matched_terms: read, pdf, extract, the, text, and, a, presentation, for |
| 3 | ocr-and-documents | 14.591584 | matched_terms: read, pdf, extract, text, and, a |
| 4 | pdf | 14.471222 | matched_terms: read, pdf, extract, text, and, a |
| 5 | document-to-action-items | 12.620685 | matched_terms: extract, text, and, turn, into, a, for |
| 6 | humanizer | 11.408445 | matched_terms: this, text, and, it, a, for |
| 7 | claude-design | 10.384140 | matched_terms: this, and, it, a, presentation, for |
| 8 | nano-pdf | 9.715278 | matched_terms: pdf, the, text, a, for |
| 9 | box | 9.257109 | matched_terms: extract, the, text, and, it, a, one |
| 10 | spike | 9.071157 | matched_terms: this, the, and, it, into, a, for |

Active bundle candidates:

None (empty runtime).

- API result: HTTP 429; no raw/validated decision, selected IDs, target/new bundle or model reason was available.
- Resulting state (unchanged):

```json
{
  "active_bundles": [],
  "direct_skills": []
}
```

Retrieval assessment: PDF extraction and presentation are present (pdf, ocr-and-documents, powerpoint). Infographic/design candidates are weak alternatives and must not all become required skills.

Resolver assessment: unavailable due to provider failure. State safety: unchanged on failure. Hallucination/schema assessment: unavailable because the model returned no decision text.

## Intended EXTEND

Need: Inspect why the GitHub Actions checks for this pull request are failing.

| Rank | Skill candidate | BM25 score | Evidence |
|---|---|---:|---|
| 1 | github-pr-workflow | 18.869540 | matched_terms: the, github, checks, for, pull, request |
| 2 | github-issue-to-pr | 14.738166 | matched_terms: the, github, for, this, pull, request |
| 3 | github-code-review | 14.335366 | matched_terms: github, for, this, pull, request |
| 4 | github-repo-management | 10.337981 | matched_terms: github, actions |
| 5 | github-auth | 10.282484 | matched_terms: the, github, for, pull |
| 6 | codex | 9.243197 | matched_terms: the, github, pull, request |
| 7 | inspecting-hermes-desktop-dom | 8.568383 | matched_terms: inspect, why, the, for, this |
| 8 | github-issues | 8.551843 | matched_terms: the, github, request |
| 9 | node-inspect-debugger | 8.425402 | matched_terms: inspect, why, the, this |
| 10 | computer-use | 7.007977 | matched_terms: the, actions, this, are |

Active bundle candidates:

- `github-review`; purpose: Review and inspect GitHub pull requests; members: github-code-review, github-auth; score=3.583969; evidence: matched_terms: inspect, the, github, for, this, pull, request.

- API result: HTTP 429; no raw/validated decision, selected IDs, target/new bundle or model reason was available.
- Resulting state (unchanged):

```json
{
  "active_bundles": [
    {
      "bundle_id": "github-review",
      "purpose": "Review and inspect GitHub pull requests",
      "skill_ids": [
        "github-code-review",
        "github-auth"
      ]
    }
  ],
  "direct_skills": []
}
```

Retrieval assessment: github-pr-workflow ranks first and covers CI checks/failure handling. Existing github-review is the top bundle candidate. No critical capability omission is evident from the corpus.

Resolver assessment: unavailable due to provider failure. State safety: unchanged on failure. Hallucination/schema assessment: unavailable because the model returned no decision text.

## Intended CREATE

Need: Search my email, triage the relevant messages, and prepare replies.

| Rank | Skill candidate | BM25 score | Evidence |
|---|---|---:|---|
| 1 | email-inbox-triage | 17.876141 | matched_terms: my, email, triage, the, messages, and |
| 2 | himalaya | 12.220975 | matched_terms: search, email, the, messages, and |
| 3 | github-issues | 10.377736 | matched_terms: search, triage, the, and |
| 4 | google-workspace | 9.828408 | matched_terms: search, email, messages, and |
| 5 | xurl | 8.235626 | matched_terms: search, my, the, messages, and |
| 6 | research-paper-writing | 6.137120 | matched_terms: the, and, prepare |
| 7 | imessage | 5.488139 | matched_terms: the, messages, and |
| 8 | meeting-action-items | 5.431740 | matched_terms: email, the, messages, and |
| 9 | obsidian | 5.412722 | matched_terms: search, my, the, and |
| 10 | session-librarian | 4.823669 | matched_terms: search, my, the, and |

Active bundle candidates:

- `github-review`; purpose: Review and inspect GitHub pull requests; members: github-code-review, github-auth; score=1.512869; evidence: matched_terms: my, the, and.

- API result: HTTP 429; no raw/validated decision, selected IDs, target/new bundle or model reason was available.
- Resulting state (unchanged):

```json
{
  "active_bundles": [
    {
      "bundle_id": "github-review",
      "purpose": "Review and inspect GitHub pull requests",
      "skill_ids": [
        "github-code-review",
        "github-auth"
      ]
    }
  ],
  "direct_skills": []
}
```

Retrieval assessment: email-inbox-triage and himalaya rank first/second; google-workspace provides another email option. The unrelated GitHub bundle was retrieved only through generic words my/the/and.

Resolver assessment: unavailable due to provider failure. State safety: unchanged on failure. Hallucination/schema assessment: unavailable because the model returned no decision text.

## Semantic boundaries

- DIRECT now permits multiple skills; action is no longer based on candidate count or coverage thresholds.
- The definition of a maintained cluster still depends on need context. The email request may reasonably produce DIRECT or CREATE; a matching expected action alone is not acceptance.
- Lexical active-bundle search can admit unrelated bundles through common words. This is a narrowing limitation, not a final EXTEND decision.
- Unit validation rejects invented/out-of-candidate skill or bundle IDs and illegal schemas with at most one repair. This is not proof that a real model will choose semantically correct organizations.
- Live acceptance must be rerun after provider access recovers. Current status is implementation complete / live resolver behavior unverified.

## Tests

Targeted runtime/legacy-loading/client tests: **37 passed**. Final full suite: **164 passed in 1.37s**. `git diff --check` passed. No commit or push was performed.
