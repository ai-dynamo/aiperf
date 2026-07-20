# Task 2 Report — Intrinsic Node Sizing

**Date:** 2026-07-20

## Result

- Added intrinsic, minimum-preserving layout for `core.chip`, `core.panel`, and
  `core.note`.
- Moved stepper layout and chrome to shared scale-aware text metrics.
- Changed rails to preserve each child's intrinsic size and authored gap while
  distributing only surplus container space.
- Added a bottom-up child layout pass shared by SceneRenderer indexing and
  rendering, including application of each child's resolved grandchild
  geometries before its parent resolves.

## TDD evidence

### RED

Command:

```text
npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts
```

Result: exit 1. Three intended failures confirmed the missing behavior:

- stepper width was `279`, expected scale-aware `265`;
- long chip width remained authored `84`;
- panel width remained authored `100`.

Five pre-existing/control assertions passed. The rail assertion initially
passed only because unresolved chips both remained `84`; after intrinsic chip
sizing was implemented, it exercised and required heterogeneous-width rail
placement.

### GREEN

Required focused command:

```text
npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts src/core/diagram/SceneRenderer.sdk-primitives.test.tsx src/core/diagram/text-metrics.test.ts
```

Result: exit 0 — 3 files passed, 19 tests passed.

Additional verification:

```text
npm --prefix apps/explainers run build
```

Result: exit 0 — TypeScript check and Vite production build passed. The build
reported only the existing large-chunk advisory.
