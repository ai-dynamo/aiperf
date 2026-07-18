# Flow Core Geometry & Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a broader generic geometry + animation vocabulary for explainer `@scene` authoring, wire it through language → compiler → SceneIr → SceneRenderer, and migrate all eight decks onto it.

**Architecture:** Hybrid lowering — desugar simple macros (`panel`, `header`, `circle`, …) into existing IR; keep first-class IR + renderer support for relative layout, `layout.stack`/`grid`, node-anchored `connector`/`elbow`, `motion.signal`/`pulse`, compact `stagger`, and `fade`/`exit`. Dual dialect: package `capability` ids and native cinematic keywords stay in sync.

**Tech Stack:** TypeScript (aiperf-flow language/compiler/schema), React SVG SceneRenderer (`apps/explainers`), `.flow` package scenes, DeckPackage JSON.

**Spec:** `docs/superpowers/specs/2026-07-18-flow-core-geometry-animation-design.md`

## Global Constraints

- Generic geometry/animation only — no AIPerf-domain `viz.*` metaphors
- No second render path; extend existing pipeline only
- Foundation nodes (`core.rect`, `core.text`, `core.path`, `core.line`, `core.dot`) remain valid
- **No new tests** — do not add test files or TDD cycles; if existing tests break from API changes, fix them minimally so suites still pass
- **Do not create git commits** unless the controller explicitly asks
- Preserve NVIDIA SPDX headers on new/edited source files
- Work from repo root: `/home/anthony/nvidia/projects/aiperf/ajc/rust`

---

## File map

| Area | Primary files |
|---|---|
| Schema | `apps/aiperf-flow/packages/schema/src/ir.ts`, `apps/aiperf-flow/packages/schema/src/index.ts` |
| Language package | `apps/aiperf-flow/packages/language/src/embedded-scene.ts` |
| Language native | `apps/aiperf-flow/packages/language/src/tokens.ts`, `grammar/explainer.ts`, `parser.ts`, `ast.ts`, `index.ts` |
| Compiler | `apps/aiperf-flow/packages/compiler/src/lower-explainer-scene.ts` (+ new helper module if needed under `compiler/src/`) |
| Renderer | `apps/explainers/src/core/diagram/SceneRenderer.tsx`, `FlowArrow.tsx`, `MotionSignal.tsx` |
| Decks | `apps/explainers/decks-flow/*.flow` |
| Packages | rebuild via existing `apps/explainers` / aiperf-flow build scripts |

---

### Task 1: Schema IR extensions

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/ir.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts` (re-exports if needed)

**Interfaces:**
- Produces: extended `FoundationCapabilityId` / open capability strings; layout props on groups; connector/elbow fields; stagger cue shape; easing / fade / exit actions

- [ ] **Step 1:** Extend foundation / known capability unions to include:
  - `core.circle`, `core.ellipse`, `core.panel`, `core.header`, `core.arrow`, `core.elbow`, `core.bracket`, `core.callout`, `core.group`
  - `layout.stack`, `layout.grid`, `layout.pad`
  - `motion.signal`, `motion.pulse`
- [ ] **Step 2:** Extend `TimelineCueAction` / cue schema to allow `fade`, `exit`, `emphasis`, `emphasize`, `pulse`, `reveal`, `trace` plus optional:
  ```ts
  easing?: "linear" | "ease-in" | "ease-out" | "ease-in-out"
  ```
- [ ] **Step 3:** Add compact stagger record support on scenes. Prefer either:
  - cues with `action: "stagger"` and `targets: string[]`, `step: number`, or
  - parallel `staggers: StaggerIr[]` on `SceneIr`
  
  Choose **cue-shaped stagger** for minimal SceneIr churn:
  ```ts
  type TimelineCueIr = {
    id: string;
    at: number;
    duration: number;
    target: string;           // primary / group id; may be "" when targets[] used
    action: TimelineCueAction | "stagger" | "enter-children";
    targets?: readonly string[];
    step?: number;
    easing?: "linear" | "ease-in" | "ease-out" | "ease-in-out";
    sourceMap: SourceRange;
  };
  ```
- [ ] **Step 4:** On `GroupNodeIr` / component-like groups, allow optional layout props via `style` or explicit fields:
  ```ts
  // Prefer style keys for package compatibility, documented:
  // layout.stack: style.direction = "row"|"column", style.gap = number
  // layout.grid: style.cols = number, style.gap = number
  ```
  Keep `ConnectorEndpointIr.anchor` as string (document allowed values). Add optional `via?: PointIr` / `axis?: "x"|"y"` on connector nodes via style or optional fields if Zod allows without breaking packages.
- [ ] **Step 5:** Update Zod schemas in the same file to accept the new fields without rejecting existing packages.
- [ ] **Step 6:** Fix any schema unit tests that fail solely due to union exhaustiveness — no new tests.

**Done when:** Schema types + Zod accept the new vocabulary; existing package JSON still parses.

---

### Task 2: Language — package + native surface

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/embedded-scene.ts`
- Modify: `apps/aiperf-flow/packages/language/src/tokens.ts`
- Modify: `apps/aiperf-flow/packages/language/src/grammar/explainer.ts` (and/or scene grammar files)
- Modify: `apps/aiperf-flow/packages/language/src/parser.ts`
- Modify: `apps/aiperf-flow/packages/language/src/ast.ts`
- Modify: `apps/aiperf-flow/packages/language/src/index.ts` if exports needed

**Interfaces:**
- Consumes: Task 1 capability names and cue fields
- Produces: AST / package scene objects that carry new capabilities, layout props, stagger cues

- [ ] **Step 1:** Ensure package-form parser already accepts arbitrary object keys; document/normalize known props for panel (`title`, `detail`), header (`title`, `caption`), circle (`r` / center), elbow (`from`/`to`/`via`/`axis`), stack/grid (`direction`/`cols`/`gap`), motion.signal (`d` or from/to), stagger cues (`targets`, `step`, `easing`).
- [ ] **Step 2:** Add native tokens + grammar for: `panel`, `header`, `circle`, `ellipse`, `arrow`, `elbow`, `bracket`, `callout`, `stack`, `grid`, `pad`, `signal` (motion), and timeline forms for `stagger`, `enter-children`, `fade`, `exit`, optional `easing`.
- [ ] **Step 3:** Extend AST node types and parser production to emit structures the compiler can lower (mirror package field names where possible).
- [ ] **Step 4:** Keep existing `rect` / `connector` / `timeline` `reveal`/`trace` working.
- [ ] **Step 5:** Fix broken language tests minimally if any; no new tests.

**Done when:** Package scenes with new fields round-trip through capture; native keywords parse into AST.

---

### Task 3: Compiler desugar + first-class lowering

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/lower-explainer-scene.ts`
- Create if useful: `apps/aiperf-flow/packages/compiler/src/desugar-scene-primitives.ts`

**Interfaces:**
- Consumes: package/native scene AST with Task 2 shapes
- Produces: SceneIr with desugared macros + first-class layout/motion/stagger preserved

- [ ] **Step 1:** Implement desugarers:
  - `core.circle` / `ellipse` → rect geometry + radius style (`r`/`rx`/`ry`)
  - `core.panel` → group/rect with relative title (+ optional detail) text children
  - `core.header` → rect + left title + right caption texts
  - `core.bracket` → path approximating a brace along a span
  - `core.callout` → text + stem path toward target point/anchor
  - `layout.pad` → child with inset offset
  - `core.arrow` with absolute geometry → connector + arrowhead style defaults
- [ ] **Step 2:** Pass through first-class:
  - `layout.stack` / `layout.grid` as `kind: "group"` with capability preserved + direction/cols/gap
  - `core.connector` / `core.elbow` as `kind: "connector"` with from/to/anchors; set style or capability so renderer can distinguish elbow
  - `motion.signal` / `motion.pulse` as connector/rect-like nodes with capability retained
- [ ] **Step 3:** Timeline: accept `stagger` / `enter-children` / `fade` / `exit` / `easing`; for `enter-children`, either expand to stagger cue targeting children’s ids (if known at lower time) or emit `action: "enter-children"` for renderer expansion.
- [ ] **Step 4:** Prefer **relative child layouts** when desugaring panel/header (title at local y offsets inside parent box).
- [ ] **Step 5:** Update `capabilityKind()` mapping; fix broken compiler tests minimally.

**Done when:** A miniature hand scene using panel + elbow + stagger lowers to valid SceneIr without throwing.

---

### Task 4: SceneRenderer — geometry & layout

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Possibly: `apps/explainers/src/core/diagram/FlowArrow.tsx`

**Interfaces:**
- Consumes: SceneIr from Task 3
- Produces: Correct SVG placement for relative children, stack/grid, anchors, elbows

- [ ] **Step 1:** Relative children — if parent is group-like and child geometry fits inside parent (or child has explicit local coords convention), offset by parent origin. Establish clear rule: **children of `core.panel` / `layout.*` / `core.group` use local coordinates** unless `style.coordinateSpace === "absolute"` (or geometry clearly world-absolute — prefer explicit local for new primitives; keep absolute working for legacy nodes by detecting absolute coords that already match world space — simplest: new desugared children always local; legacy absolute scenes unchanged because they don’t use local parents).
- [ ] **Step 2:** Expand anchors to `n|s|e|w|ne|nw|se|sw|center` plus `top|bottom|left|right` aliases.
- [ ] **Step 3:** `core.elbow` / elbow-capable connectors: build orthogonal path `M x1 y1 H/V mid H/V x2 y2` (use `via` or midpoint heuristic; `axis` prefers first bend direction).
- [ ] **Step 4:** `layout.stack`: place children along row/column with `gap`; size parent from children if needed.
- [ ] **Step 5:** `layout.grid`: place children in row-major cells with `cols` + `gap`.
- [ ] **Step 6:** Render circles/ellipses (from desugared rect+radius or capability) as SVG `rect` rx/ry or `ellipse`.
- [ ] **Step 7:** Fix broken scene-renderer tests minimally if API assumptions change.

**Done when:** Panel with local title text, stack of boxes, and elbow between two nodeIds render correctly in the explainers preview path.

---

### Task 5: SceneRenderer — animation

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/diagram/MotionSignal.tsx` if needed

**Interfaces:**
- Consumes: stagger / motion / fade / exit / easing cues
- Produces: Playback matching existing enter/draw/emphasis contracts

- [ ] **Step 1:** Expand compact `stagger` / `enter-children` at play time into per-target enter (or specified action) cues with `at + i*step`.
- [ ] **Step 2:** First-class `motion.signal`: treat capability as motion guide + traveling tip (reuse MotionSignal); no id-heuristic required when capability is set.
- [ ] **Step 3:** `motion.pulse`: honor `pulse` cues and continuous pulse for `motion.pulse` nodes without relying only on `pulse-*` ids.
- [ ] **Step 4:** Implement `fade` / `exit` opacity ramp down; after exit completes, treat as hidden for hit/opacity.
- [ ] **Step 5:** Apply cue `easing` to progress mapping (simple cubic approximations OK).
- [ ] **Step 6:** Reduced-motion: still jump to end; omit traveling dots.

**Done when:** A scene with stagger enter-children + motion.signal + fade plays coherently.

---

### Task 6: Migrate all eight decks + rebuild packages

**Files:**
- Modify: all under `apps/explainers/decks-flow/*.flow`
- Regenerate: `apps/explainers/src/decks-generated/*.package.json`

**Rewrite patterns (apply systematically):**

| Old pattern | New |
|---|---|
| `core.rect` + 2× `core.text` title/detail children | `core.panel` with `title` / `detail` |
| Header strip rect + left/right texts | `core.header` with `title` / `caption` |
| Hand `d:` between boxes when axis-aligned | `core.elbow` or `core.connector` with `from`/`to` `{ nodeId, anchor }` |
| Path + draw + motion-sig id/label heuristic | `motion.signal` + `draw` (or built-in play) |
| Long identical enter cue lists | `stagger` or `enter-children` |
| Pulse overlay empty rects with pulse-* ids | `motion.pulse` where appropriate |

- [ ] **Step 1:** Migrate decks in this order (smaller → larger risk): `dynosim`, `segment-pools`, `slurm-velo`, `velo-deep-dive`, `rust-architecture`, `rust-architecture-atlas`, `cellular-internals`, `cellular-algorithms`.
- [ ] **Step 2:** Rebuild packages with the project’s existing explainer package build script (check `apps/explainers/package.json` / Makefile targets such as `build:packages` / `assert-deck-packages`).
- [ ] **Step 3:** Smoke: load hub / one deck route if scripts available; at minimum ensure compile/pack exits 0 for all eight.
- [ ] **Step 4:** Leave low-level nodes only where a higher primitive does not fit (curved custom paths, one-off art).

**Done when:** All eight `.flow` files use the new catalog for common patterns; regenerated packages compile cleanly.

---

## Self-review (plan)

1. Spec coverage: architecture, geometry catalog, animation catalog, migration, dual dialect, hybrid lowering, no-tests — all mapped to Tasks 1–6.
2. No placeholder TBD steps.
3. Types: stagger cue shape and capability names consistent across tasks.
