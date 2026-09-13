# RT-T02 EXTEND failure diagnosis (v0.1)

Scope: Round 1 control-plane run `RT-T02` (`a1357be5-ddfc-4f0a-8383-b2b515c7afbd`),
using its persisted Pi session, trace, and result log.  The normal cell log only
contains the summary row; the session file contains the model-visible tool error
payloads.

## Reconstructed requests and results

| Trace event | Request delta | Result |
| --- | --- | --- |
| 50 (result 52) | `EXTEND`, `skill_ids=["playwright-expert"]`, `reason`, `purpose`, and one coverage claim; **no** `target_bundle_id` | `target_bundle_id must be a non-empty string` |
| 53 (result 55) | Same request, now with `target_bundle_id="cap-66fab1fdb2ba4e589b21832cc3bf1fce"` | `EXTEND cannot set purpose` |
| 56 (result 58) | Same request with `purpose` removed; target bundle retained | `Operation aborted` followed by the provider-level `This operation was aborted` stop error |

The target bundle is real and contains the resident `vite` Skill.  The preceding
search (event 49) ranked `playwright-expert` first, so neither candidate coverage
nor the Skill ID was at fault.  No apply result, memory commit, activation, or
body load followed any request.

## Root cause classification

The first two failures are distinct schema/contract validation errors:

1. `EXTEND` requires a target bundle, but the model-facing Pi tool schema makes
   `target_bundle_id` optional.  The runtime policy shown at turn start had no
   Bundle Card; the newly created bundle ID was not reprojected as a fresh policy
   message before the later EXTEND.
2. The same broad schema exposes `purpose` for all actions, while the sidecar
   accepts it only for `CREATE`.  The model reasonably retained the purpose from
   its successful CREATE when constructing the EXTEND request.

The third request satisfies the protocol decoder.  Its `Operation aborted` is
outside schema validation, bundle lookup, coverage validation,
`DiscoverySession.apply`, `CapabilityMemory.commit`, and adapter serialization:
the provider aborted the model operation immediately after receiving the tool
result.  It therefore did not produce an application-level failure payload.

Conclusion: this is not a `playwright-expert` or benchmark-specific parameter
mistake.  It is a runtime/API-contract usability defect: an action-dependent
contract was represented to the model as one permissive object, and the first
error did not contain the actionable bundle target needed to recover.  The repair
should make `CREATE`, `EXTEND`, and `DIRECT` schemas mutually explicit and retain
clear server-side validation for non-Pi callers.
