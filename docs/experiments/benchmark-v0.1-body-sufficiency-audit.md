# Benchmark v0.1 Skill-body sufficiency audit

All 36 benchmark tasks are marked `body_sufficient: true`. The audit examined the
16 distinct required Skills in their frozen S128 packages. No frozen package was
changed and no task requires Native-only access to `references/`, `scripts/`, or
`assets/` to satisfy its verifier.

| Skill | Task-scoped guidance present in `SKILL.md` |
|---|---|
| requirements-analysis | requirement types, ambiguity, assumptions, acceptance criteria |
| source-driven-development | source authority, conflict resolution, claim citations |
| fine-tuning-expert | splits, leakage prevention, evaluation, rollback |
| threejs-webgl | scene, camera, renderer, lighting, animation, resize |
| gsap-react | scoped context and lifecycle cleanup |
| api-design | resources, errors, authentication, idempotency |
| fastify | schemas, validation, route status behavior |
| rag | deduplication, embeddings, ranking, citations |
| create-website | semantic/responsive page construction |
| terraform-engineer | provider pins, validation, encryption, outputs |
| monitoring | SLIs/SLOs, burn-rate alerts, ownership, runbooks |
| vite | environment prefix, proxy, production chunking |
| seo | canonical/description metadata and Product structured data |
| golang-pro | cancellation, goroutine lifecycle, error wrapping |
| playwright-expert | resilient locators, state waits, page objects, retry traces |
| storybrand-messaging | customer, problem, guide, plan, stakes, call to action |

Some packages link optional deep references (notably API design, Go, Vite,
Terraform, browser testing, fine-tuning, website creation, and messaging). Each
selected task deliberately stays within the actionable rules and examples already
present in the body. The verifiers do not demand a template, script, or asset that
exists only in those linked resources.
