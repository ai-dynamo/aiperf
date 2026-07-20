# Round-2 Fixer A Report

## Scope completed

- Added intrinsic layout for every catalog capability ID: `diagram.actor`, `diagram.compute`, `diagram.storage`, `diagram.messaging`, `diagram.network`, `diagram.control`, `diagram.boundary`, and `diagram.symbol`.
- Diagram title/detail sizing mirrors semantic chrome font sizes, glyph gutter, boundary inset, and title/detail bands. Authored dimensions remain minimums; `clip: true` and `overflow: hidden` preserve authored bounds.
- Added `core.group` presentation sizing for `code-block`, `quote`, and `icon-label`. Code blocks measure their longest line and line count; non-presentation groups retain identity behavior. Avatar remains identity-sized because its chrome has no props-driven text.
- Raised lane title, frame title, and frame detail bands from `28/28/48` to `32/32/52`.

## TDD and verification

- Added regression coverage for all eight diagram IDs, presentation width/height growth, clipping exceptions, and updated lane/frame band placement.
- Confirmed the new tests failed before implementation for missing diagram/presentation resolvers and old band values.
- Passed:
  `npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts src/core/diagram/text-metrics.test.ts`
  (`30` tests, `0` failures).
- IDE lint diagnostics report no errors in the two edited TypeScript files.

No commit was created. No `.flow`, CSS, or generic chrome files were edited.
