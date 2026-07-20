# Roomier Explainer Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give shared explainer diagrams more viewport room while keeping subtitles, title, and the complete wrapped lede visible without cutoff.

**Architecture:** Keep the existing two-row cinematic shell and use shared responsive CSS for viewport fit. Compact the footer, make SVG containment explicit, add a short-height breakpoint, and apply one shared font scale to both semantic and regular SVG text in the scene renderer.

**Tech Stack:** CSS, React shell markup, Vitest, Vite

## Global Constraints

- Preserve the full-screen presentation model without normal page scrolling.
- Keep subtitles, title, eyebrow, and the full wrapped lede visible.
- Render all scene SVG text at 90% of its authored or default size.
- Do not add per-deck scaling or alter scene geometry.

---

### Task 1: Make the shared stage responsive and cutoff-safe

**Files:**
- Modify: `apps/explainers/src/index.css:730-742`
- Modify: `apps/explainers/src/index.css:1209-1307`
- Modify: `apps/explainers/src/index.css:1479-1516`
- Test: `apps/explainers/src/core/ExplainerShell.subtitles.test.tsx`

**Interfaces:**
- Consumes: existing `.ex-cinematic-stage`, `.ex-stage-hero`, `.ex-stage-footer`, `.ex-stage-copy`, and `.ex-lede` markup.
- Produces: shared responsive stage sizing with no component API changes.

- [ ] **Step 1: Run the focused shell test before editing**

Run:

```bash
cd apps/explainers && npm test -- src/core/ExplainerShell.subtitles.test.tsx
```

Expected: both existing subtitle/footer structure tests pass.

- [ ] **Step 2: Compact the footer and make SVG containment explicit**

Update the shared rules so the footer uses smaller responsive gaps and padding, the stage has balanced safe insets, and SVGs preserve aspect ratio within their available box:

```css
.ex-stage-footer {
  gap: clamp(8px, 1.2vh, 12px);
  padding: 0 clamp(18px, 4vw, 72px) clamp(48px, 6vh, 68px);
}

.ex-stage-hero.ex-content-card__diagram {
  overflow: visible;
  padding: clamp(54px, 7vh, 76px) clamp(24px, 4vw, 72px) 8px;
}

.ex-stage-hero.ex-content-card__diagram svg {
  max-width: 100%;
  max-height: 100%;
  overflow: visible;
}
```

Tighten the stage-copy gap, title margin, and lede line height while leaving the lede unconstrained in height.

- [ ] **Step 3: Add short-height responsive compaction**

Add a `@media (max-height: 760px)` rule that reduces hero/footer padding, copy gaps, title size, lede size, and subtitle padding. Do not hide, clamp, truncate, or absolutely position footer text.

- [ ] **Step 4: Preserve mobile fit**

Adjust the existing `max-width: 720px` rules to use the compact footer and hero insets. Keep the mobile lede at a readable size and full wrapping.

- [ ] **Step 5: Run focused and package verification**

Run:

```bash
cd apps/explainers
npm test -- src/core/ExplainerShell.subtitles.test.tsx
npm run build
```

Expected: tests and build exit zero.

- [ ] **Step 6: Check edited-file diagnostics**

Check `apps/explainers/src/index.css` and `apps/explainers/src/core/ExplainerShell.subtitles.test.tsx`; resolve any newly introduced diagnostics.

### Task 2: Reduce all scene SVG text consistently

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Test: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`

**Interfaces:**
- Consumes: authored and default `fontSize` values for semantic chrome and `core.text` nodes.
- Produces: SVG `font-size` attributes equal to 90% of those source values, with automatic line height based on the scaled size.

- [ ] **Step 1: Add a failing renderer test**

Render one semantic panel and one pair of regular text nodes: one with `fontSize: 20` and one using the default. Assert their rendered font sizes are `12.6`, `10.35`, `18`, and `12.6` respectively.

- [ ] **Step 2: Run the focused test and confirm the expected failure**

Run:

```bash
npm --prefix apps/explainers test -- src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: the new assertions fail because the renderer still emits unscaled font sizes.

- [ ] **Step 3: Apply a shared scene-text scale**

Add a module-level `SCENE_TEXT_SCALE = 0.9` constant and a small finite-number scaling helper. Use it for semantic chrome `part.fontSize` and regular `core.text` authored/default font sizes. Keep explicit authored line heights unchanged; derive automatic line height from the scaled font size.

- [ ] **Step 4: Run focused verification**

Run:

```bash
npm --prefix apps/explainers test -- src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
npm --prefix apps/explainers run build
```

Expected: the focused test and build exit zero.

- [ ] **Step 5: Check diagnostics**

Check `SceneRenderer.tsx` and `SceneRenderer.sdk-primitives.test.tsx`; resolve newly introduced diagnostics.
