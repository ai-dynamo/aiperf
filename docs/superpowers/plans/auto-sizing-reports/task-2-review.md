# Task 2 Review — Intrinsic Chip / Panel / Note Sizing

**Date:** 2026-07-20
**Reviewer:** Cursor agent (Composer)
**Plan:** `docs/superpowers/plans/2026-07-20-diagram-node-auto-sizing.md` (Task 2)
**Implementer report:** `docs/superpowers/plans/auto-sizing-reports/task-2-report.md`

---

## Verdict

| Dimension | Result |
|---|---|
| **Spec compliance** | **PASS** |
| **Code quality** | **APPROVED** |

All six verification checkpoints pass. Tests run green. No blocking defects found.

---

## Verification checklist

### 1. `resolveChipLayout` / `resolvePanelLayout` registered for `core.chip`, `core.panel`, `core.note`

**PASS.** `layout.ts` exports both resolvers and registers them in `LAYOUT_CAPABILITIES`:

```516:523:apps/explainers/src/core/diagram/capabilities/layout.ts
export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  // ...
  { capabilityId: "core.chip", resolveLayout: resolveChipLayout },
  { capabilityId: "core.panel", resolveLayout: resolvePanelLayout },
  { capabilityId: "core.note", resolveLayout: resolvePanelLayout },
```

`registry.ts` consumes `LAYOUT_CAPABILITIES` unchanged; dispatch is wired.

### 2. `layout.ts` and `chrome.ts` use `text-metrics` (no duplicate `STEPPER_CHAR_WIDTH`)

**PASS.** Both files import shared constants and helpers from `../text-metrics.js`. Grep over `capabilities/` finds no `STEPPER_CHAR_WIDTH` or local `stepWidth`. Stepper sizing in layout uses `stepperChipWidth`; chrome uses the same import.

**Note (non-blocking, out of Task 2 scope):** `desugar-scene-primitives.ts` still defines unscaled `STEPPER_CHAR_WIDTH`. Scout flagged this; plan Task 2 file list excludes it. Track for a follow-up convergence pass.

### 3. `indexSceneNodes` bottom-up resolves child layouts before parents

**PASS.** New helper `resolveLayoutChildren` recursively resolves grandchildren first, applies each child's intrinsic `layout.bounds`, then merges grandchild placements from the child's `childGeometries`. `resolveContainerLayout` calls it before `resolveCapabilityLayout` on the parent. Both `indexSceneNodes` (line ~675) and the render path (line ~3701) use `resolveContainerLayout`, so indexing and painting share the same bottom-up contract.

### 4. Authored sizes treated as minimums

**PASS.**

- **Chip:** `Math.max(authored.width, estimateTextWidth(...) + CHIP_PAD_X)` and `Math.max(authored.height, STEPPER_CHIP_HEIGHT)`.
- **Panel / note:** `Math.max(authored.width, titleWidth, detailWidth)` and `Math.max(authored.height, contentHeight)`.
- **Stepper:** `Math.max(authored.width, intrinsicWidth)`.
- **Rail:** container `width`/`height` are `Math.max(authored, minContent)`; surplus space is distributed evenly across children without shrinking below intrinsic widths.

Short-label chip test confirms authored `84` is retained when content fits; long-label chip grows above `84`.

### 5. Tests cover scale-aware stepper, chip growth, panel growth, rail reflow

**PASS.** `layout.test.ts` includes all plan-specified cases plus note coverage:

| Case | Test | Status |
|---|---|---|
| Scale-aware stepper | `"expands a semantic stepper using scale-aware chip widths"` | Uses `stepperChipWidth` sum + gaps; expected width 265 (not legacy 279) |
| Chip growth / minimum | `"grows a chip to fit its label while treating authored size as a minimum"` | Exact `estimateTextWidth` assertion |
| Panel growth | `"grows panel and note chrome to fit title and detail bands"` | Panel + note (caption via `resolvePanelLayout`) |
| Rail reflow | `"reflows a rail after chips auto-grow, preserving gap"` | Heterogeneous widths, gap 8 preserved |

`SceneRenderer.sdk-primitives.test.tsx` stepper assertion updated to `String(stepperChipWidth("layout", 0))` (was hard-coded `"80"`).

**Test run (reviewer):**

```text
npm --prefix apps/explainers test -- \
  src/core/diagram/capabilities/layout.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx \
  src/core/diagram/text-metrics.test.ts
```

Result: 3 files, 19 tests, exit 0.

### 6. No `.flow` or CSS edits

**PASS.** `git diff --name-only` for Task 2 touch points lists only:

- `capabilities/layout.ts`
- `capabilities/chrome.ts`
- `capabilities/layout.test.ts`
- `SceneRenderer.sdk-primitives.test.tsx`
- `SceneRenderer.tsx`

No `.flow` deck files or `index.css` changes in this scope.

---

## Code quality notes

### Strengths

- Clean separation: intrinsic leaf resolvers in `layout.ts`, chrome placement consumes the same `text-metrics` module — layout boxes and painted stepper chips stay paired.
- `resolveLayoutChildren` is the right abstraction: one fix shared by indexer and render path, avoiding drift.
- `resolveRailLayout` refactor correctly moves from equal-slot stretching to intrinsic-width placement with gap preservation and optional surplus distribution — aligns with the design spec's "reflow later children after a node grows."
- TDD evidence in implementer report matches observed behavior (RED failures on stepper 279, chip 84, panel 100 before implementation).

### Advisory (non-blocking)

1. **Rail reflow test is resolver-level, not indexer-level.** The rail test manually pre-resolves chip geometries before calling `resolveCapabilityLayout(rail, …)`. Production paths use `resolveLayoutChildren` inside `resolveContainerLayout`, which is correct, but there is no dedicated test asserting `indexSceneNodes` world bounds for a rail of auto-growing chips. Consider a small SceneRenderer or exported-index helper test in Task 3 or as a follow-up.

2. **Compiler desugar metrics still unscaled.** IR geometry emitted at compile time may differ from runtime layout until `desugar-scene-primitives.ts` converges on `text-metrics.ts`. Documented deferral; not a Task 2 regression.

3. **`core.header` not registered.** Plan listed it as optional ("if title-driven height needs a minimum"). No header-specific sizing test or registry entry — acceptable for Task 2 scope.

---

## Plan step coverage

| Plan step | Status |
|---|---|
| Step 1: Failing layout tests | Done — scale-aware stepper, chip, panel, rail cases present |
| Step 2: Confirm RED | Corroborated by implementer report |
| Step 3: Intrinsic leaf layouts + registry + indexer fix | Done |
| Step 4: Align `chrome.ts` | Done |
| Step 5: Run layout + renderer tests | Done — reviewer re-ran, all pass |

---

## Critical findings

None. Task 2 meets plan and design-spec requirements. Safe to proceed to Task 3 (verifier layout parity).
