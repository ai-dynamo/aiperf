# Diagram Node Auto-Sizing Review

## Verdict

**Changes requested.** The shared metrics, bottom-up resolution, rail reflow, and
verifier parity paths are present and their focused tests pass, but the
implementation does not yet cover several capability contracts required by the
spec.

## Findings

### Important — `sdk.note` text never participates in intrinsic sizing

**Status:** Confirmed
**Evidence:** `sdk.note` emits `core.note` with `props: { text }`
([generic/chrome.ts:890-904](../../../../apps/explainers/src/flow/sdk/generic/chrome.ts#L890-L904)).
The renderer treats that `text` as the visible title
([capabilities/chrome.ts:95-100](../../../../apps/explainers/src/core/diagram/capabilities/chrome.ts#L95-L100)),
but `resolvePanelLayout`, which is registered for `core.note`, considers only
`title`/`label` and `detail`/`caption`
([capabilities/layout.ts:342-372](../../../../apps/explainers/src/core/diagram/capabilities/layout.ts#L342-L372)).
Consequently a long ordinary `sdk.note` remains at its authored width and
overflows, contrary to the note-caption requirement. The regression test uses
an unproduced `{ title, caption }` note shape
([capabilities/layout.test.ts:223-238](../../../../apps/explainers/src/core/diagram/capabilities/layout.test.ts#L223-L238)),
so it misses the SDK path.

### Important — header and text/label layout boxes remain fixed-size

**Status:** Confirmed
**Evidence:** `core.header` is native semantic chrome
([capabilities/chrome.ts:70-86](../../../../apps/explainers/src/core/diagram/capabilities/chrome.ts#L70-L86))
and SDK headers emit title/caption properties
([generic/chrome.ts:480-495](../../../../apps/explainers/src/flow/sdk/generic/chrome.ts#L480-L495)).
Likewise `sdk.label` emits a `core.text` node with the visible text and font
size ([generic/chrome.ts:959-972](../../../../apps/explainers/src/flow/sdk/generic/chrome.ts#L959-L972)).
Neither `core.header` nor `core.text` appears in `LAYOUT_CAPABILITIES`
([capabilities/layout.ts:965-980](../../../../apps/explainers/src/core/diagram/capabilities/layout.ts#L965-L980)),
so both fall back to identity layout. This omits two explicitly scoped
capabilities and lets long header titles/captions and labels overflow.

### Important — clip/hidden-overflow nodes still enlarge

**Status:** Confirmed
**Evidence:** The identity resolver preserves authored dimensions when
`overflow: "hidden"` or `clip: true`
([capabilities/layout.ts:294-313](../../../../apps/explainers/src/core/diagram/capabilities/layout.ts#L294-L313)),
but both intrinsic leaf resolvers grow unconditionally
([capabilities/layout.ts:315-374](../../../../apps/explainers/src/core/diagram/capabilities/layout.ts#L315-L374)).
Thus a clipped `core.chip`, `core.panel`, or `core.note` violates the stated
exception that authored dimensions are minimums only unless overflow is
explicitly clipped/hidden.

## Validation and residual risks

- Passed: `npm --prefix apps/explainers test -- src/core/diagram/text-metrics.test.ts src/core/diagram/capabilities/layout.test.ts src/core/diagram/resolution/resolve-scene.test.ts src/flow/dev-tools/verify-geometry.test.ts` (24 tests).
- No diagnostics were reported for the reviewed files.
- Residual risk after the findings: tests cover manually assembled node shapes
  more than actual SDK factory output; add factory-to-resolution assertions for
  notes, headers, labels, clipped leaves, and a container reflow using each.
