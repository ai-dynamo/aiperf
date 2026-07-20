# Spacing Fix A Report

**Status:** Complete.

## Changes

- Added shared `SUBTITLE_HEIGHT` and aligned semantic subtitle chrome to it.
- Extended panel intrinsic sizing to include subtitle width and height while preserving authored clipping.
- Added and registered `core.callout` intrinsic sizing from its padded, scale-aware label metrics while preserving authored clipping.
- Made `sdk.legend` grow authored/default widths to fit scale-aware title and entry labels.
- Added regression coverage for subtitle panels, clipped panels, callouts, clipped callouts, and legend factory sizing.

## Verification

- TDD red run: `layout.test.ts` failed the new panel, callout, and legend assertions before implementation.
- `npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts src/core/diagram/text-metrics.test.ts` — 26 tests passed.
- `./apps/explainers/node_modules/.bin/tsc -p apps/explainers/tsconfig.json --noEmit` — passed.
- IDE diagnostics for all changed implementation and test files — no errors.
