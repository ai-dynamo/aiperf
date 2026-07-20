# Round 4 Fixer A — Track/Value Chrome Sibling Overlap (Domain 1)

**Date:** 2026-07-20
**Scope:** `apps/explainers/src/flow/sdk/generic/catalog.ts` — gauge/progress/meter `__track` × `__value` sibling-overlap diagnostics.
**Reference:** `docs/superpowers/plans/auto-sizing-reports/remaining-spacing-round4-scout.md`, Domain 1.

## Root cause

`indicatorFactory` in `catalog.ts` emitted the background `__track` rect and the
foreground `__value` (fill) rect as two absolute rect children directly under
the indicator's root group, both anchored at `{x:0, y:0}` and sharing the same
height. This is intentional visual chrome — the value bar paints over the
track to show progress — but the resolver's generic
`SCENE_ABSOLUTE_SIBLING_OVERLAP` check flagged every one of these pairs as a
spacing defect because it has no way to distinguish deliberate overlay
painting from authored Y-spacing bugs.

## Fix

Chose approach 2 from the brief (mark the pairing as a non-sibling overlay for
overlap checks), using the resolver's existing `layout.overlay` escape hatch
rather than inventing a new mechanism:

- `apps/explainers/src/flow/sdk/generic/catalog.ts`: in the non-sparkline /
  non-rating / non-semaphore branch of `indicatorFactory`, the track rect and
  value rect are now nested as children of a synthetic
  `${instanceId}__band` group node with `capabilityId: "layout.overlay"`,
  instead of being pushed as direct siblings of the indicator root.
  - `resolve-scene.ts` already special-cases `layout.overlay` parents: nodes
    whose parent has that capability are excluded from sibling-overlap
    grouping entirely (`capabilityOf(parent) !== "layout.overlay"` gate), and
    `childrenUseLocalLayout` treats any `layout.*` capability as local-space,
    so the band's children keep their original `{x:0,y:0,...}` relative
    geometry and render pixel-identical to before.
  - `resolveOverlayLayout` (already registered for `layout.overlay` in
    `capabilities/layout.ts`) recomputes the band's own bounds as
    `max(authored, intrinsic)` with default `align: "start"` / `justify:
    "start"`, which reproduces the original track/value positions exactly
    (verified via the pre-existing `layout.test.ts` overlay-layout cases and
    the new unit-test assertions below).
  - Added `pulseTargetId` bookkeeping so the `value` port and the `pulse`
    action still target the actual `__value` rect id (not the new wrapper
    group id), preserving prior interactive/animation semantics.
- No changes were needed in `capabilities/layout.ts` or `capabilities/chrome.ts`
  — the `layout.overlay` capability and its resolver already existed for
  exactly this "intentional overlap" purpose (see the resolver's own repair
  hint: *"Move the siblings apart or place intentional overlap in
  layout.overlay."*).

## Before / after

Acceptance command:

```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck sdk-generic-catalog 2>&1 \
  | rg 'SCENE_ABSOLUTE_SIBLING_OVERLAP.*(__track|__value)' | wc -l
```

| | Count |
|---|---:|
| Before | 15 |
| After | **0** |

The 15 baseline pairs cleared: `open-g`, `c6-g`, `progress-hero`, `progress-v1`,
`progress-v2`, `progress-v3`, `meter-hero`, `meter-v1`, `meter-v2`, `meter-v3`,
`gauge-hero`, `gauge-v1`, `gauge-v2`, `gauge-v3`, `final-progress`
(`__track`×`__value` each).

No new `SCENE_VIEWPORT_ESCAPE` diagnostics appeared on slides 41–44
(progress/meter/gauge catalog slides) after the fix; a full deck run
immediately after the change showed `0 warn(s)` (1 pre-existing, unrelated
`arrow-degenerate-path` error on `il-v3__icon`, slide 24, untouched by this
fix).

Note: the deck file `sdk-generic-catalog.flow` was being edited concurrently
by another Round 4 fixer (Domain 2) while this fix was in progress, so
absolute total warning counts on the deck fluctuated between runs for
unrelated reasons (opener, paragraph, media-object, table, timeline slides).
The `__track|__value` filter used for acceptance is unaffected by those
concurrent deck edits and stayed at 0 across every re-run performed here.

## Tests

- `src/flow/sdk/generic/catalog.test.ts` (33 tests) — includes a pre-existing
  test, `"groups indicator paint layers under an intentional overlay"`, that
  asserts the exact `__band` / `layout.overlay` / children / ports / pulse
  shape produced by this change. All pass.
- `src/core/diagram/capabilities/layout.test.ts` (31 tests) — pass, unchanged
  (no edits made to `layout.ts`).
- Broader regression sweep: `npx vitest run src/flow/sdk src/core/diagram`
  → 20 files / 191 tests, all passing.

## Residual risk

- The fix is resolver-diagnostic-only and does not change any rendered pixel
  output (verified analytically via `resolveOverlayLayout`'s default
  `align: "start"` / `justify: "start"` math, and empirically via the
  unchanged `SceneRenderer.sdk-primitives.test.tsx` suite). Visual risk is low.
- `sdk.sparkline`, `sdk.rating`, and `sdk.semaphore` indicator variants were
  intentionally left untouched — they don't emit `__track`/`__value` pairs and
  had no matching diagnostics in the scout report.
- This fix only covers the 15 `__track`/`__value` pairs. The remaining
  non-track/value overlaps and viewport escapes on `sdk-generic-catalog.flow`
  (opener, paragraph, mediaObject, table, timeline, finale slides) are owned
  by Domain 2 and other Round 4 fixers per the scout's worker assignment
  matrix, and were left untouched here as instructed.
