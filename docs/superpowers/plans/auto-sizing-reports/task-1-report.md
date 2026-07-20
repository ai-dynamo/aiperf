# Task 1 Report: Shared Scale-Aware Text Metrics

**Date:** 2026-07-20
**Plan:** `docs/superpowers/plans/2026-07-20-diagram-node-auto-sizing.md` — Task 1 only
**Status:** DONE

## Summary

Introduced a shared `text-metrics` module as the single source of truth for scene text scale and deterministic width estimation. Removed duplicate constants and helpers from `SceneRenderer.tsx`; renderer now imports `scaledSceneFontSize` from the new module. All specified unit and regression tests pass.

## Files Changed

| File | Action |
|---|---|
| `apps/explainers/src/core/diagram/text-metrics.ts` | Created |
| `apps/explainers/src/core/diagram/text-metrics.test.ts` | Created |
| `apps/explainers/src/core/diagram/SceneRenderer.tsx` | Modified — removed local scale helpers; import from `text-metrics.js` |

## TDD Steps

### Step 1: Failing tests

Added `text-metrics.test.ts` with four cases from the plan:

- `SCENE_TEXT_SCALE` equals `0.9`
- `scaledSceneFontSize(20)` → `18`; `scaledSceneFontSize(undefined)` → `12.6`
- `estimateTextWidth("authoritative", 11, "bold")` → `Math.ceil(13 * 6.2 * 0.9)`
- `stepperChipWidth("layout", 0)` → `Math.max(72, Math.ceil("1. layout".length * 6.2 * 0.9) + 24)`

### Step 2: Confirm failure

```bash
npm --prefix apps/explainers test -- src/core/diagram/text-metrics.test.ts
```

**Result:** FAIL — `Failed to resolve import "./text-metrics.js"` (module not found). As expected.

### Step 3: Implementation

Created `text-metrics.ts` with:

- Constants: `SCENE_TEXT_SCALE`, `DEFAULT_SCENE_FONT_SIZE`, `CHAR_WIDTH`, `BOLD_CHAR_WIDTH`, `INSET`, `TITLE_HEIGHT`, `DETAIL_HEIGHT`, `CHIP_PAD_X`, `STEPPER_MIN_CHIP_WIDTH`, `STEPPER_CHIP_HEIGHT`, `STEPPER_CHIP_PAD`
- Functions: `scaledSceneFontSize`, `estimateTextWidth`, `stepperChipWidth`

All logic matches the plan verbatim. No DOM `measureText`; metrics are deterministic.

### Step 4: SceneRenderer integration

Removed from `SceneRenderer.tsx`:

- `const SCENE_TEXT_SCALE = 0.9`
- `const DEFAULT_SCENE_FONT_SIZE = 14`
- Local `scaledSceneFontSize` function

Added:

```ts
import { scaledSceneFontSize } from "./text-metrics.js";
```

Existing call sites (`scaledSceneFontSize(part.fontSize)`, `scaledSceneFontSize(node.style?.fontSize)`) unchanged.

### Step 5: Verification

```bash
npm --prefix apps/explainers test -- src/core/diagram/text-metrics.test.ts src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

**Result:** PASS — 2 test files, 10 tests (4 metrics + 6 renderer SDK primitives).

## Constraints Observed

- No commits (per user instruction)
- No `.flow` deck edits
- No shell CSS changes
- `SCENE_TEXT_SCALE = 0.9` enforced
- NVIDIA SPDX Apache-2.0 headers on new source files

## Concerns

None. Task 1 scope is complete. Downstream tasks (2–3) will consume `estimateTextWidth`, `stepperChipWidth`, and layout constants from this module.

## Next Steps (out of scope for Task 1)

- Task 2: Wire shared metrics into `capabilities/layout.ts` and `chrome.ts` for intrinsic chip/panel/note/stepper sizing
- Task 3: Align `verify-geometry.ts` with capability layout resolution
