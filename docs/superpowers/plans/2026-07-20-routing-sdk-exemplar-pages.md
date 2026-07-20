<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routing SDK Exemplar Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nine complete routing cookbook slides—including all 81 source/target anchor combinations—to the Flow SDK examples deck.

**Architecture:** The implementation remains deck-only: public `sdk.*` calls compile into ordinary Scene IR and exercise the existing renderers. A verifier contract pins the 19-slide sequence and checks that the matrix contains every ordered anchor pair before the deck content is added.

**Tech Stack:** Flow DSL, TypeScript/Node geometry verifier, Vite, Playwright

## Global Constraints

- Keep all product implementation in `apps/explainers/decks-flow/flow-sdk-examples.flow`.
- Preserve every existing slide and append the nine routing slides after “Topology patterns.”
- Use only public `sdk.*` authoring; do not add raw SVG, package-form Scene IR, or `freeform`.
- Add a non-empty timeline to every page and an explicit draw/trace cue for every directed edge.
- Use anchor order `center`, `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, `sw`.
- Update all deck fractions to `x of 19` and preserve existing IDs.
- Do not modify router behavior as part of the exemplar work.

---

### Task 1: Pin the routing-cookbook contract

**Files:**
- Modify: `apps/explainers/scripts/flow-verifier/ir.mjs`
- Test: `npm run flow-verifier:ir`

**Interfaces:**
- Consumes: compiled package `pkg.id`, `pkg.slides`, `slide.render.scene.roots`.
- Produces: `verifyRoutingSdkExamples(pkg, findings)`, called by `verifyPackageIr`.

- [ ] **Step 1: Add the failing package contract**

Add a verifier helper that:

```js
const ROUTING_EXEMPLAR_TITLES = [
  "Complete 9×9 curve matrix",
  "Cardinal curves",
  "Corner and center curves",
  "Same-side links and self-loops",
  "Obstacle avoidance",
  "Parallel lanes",
  "Bundling",
  "Anchor-safe orthogonal routing",
  "Routing controls reference",
];

function verifyRoutingSdkExamples(pkg, findings) {
  if (pkg?.id !== "flow-sdk-examples") return;
  const slides = Array.isArray(pkg.slides) ? pkg.slides : [];
  if (slides.length !== 19) {
    findings.push(finding("error", pkg.id, "*", "routing-exemplar-slide-count", `expected 19 slides, got ${slides.length}`));
  }
  const titles = new Set(slides.map((slide) => slide?.title));
  for (const title of ROUTING_EXEMPLAR_TITLES) {
    if (!titles.has(title)) {
      findings.push(finding("error", pkg.id, "*", "routing-exemplar-missing-slide", `missing "${title}"`));
    }
  }
  const matrix = slides.find((slide) => slide?.title === "Complete 9×9 curve matrix");
  const roots = Array.isArray(matrix?.render?.scene?.roots) ? matrix.render.scene.roots : [];
  const pairs = new Set(
    roots
      .filter((node) => node?.style?.route === "curve")
      .map((node) => `${node?.from?.anchor ?? "center"}:${node?.to?.anchor ?? "center"}`),
  );
  const anchors = ["center", "n", "s", "e", "w", "ne", "nw", "se", "sw"];
  for (const from of anchors) {
    for (const to of anchors) {
      if (!pairs.has(`${from}:${to}`)) {
        findings.push(finding("error", pkg.id, "curve-matrix", "routing-anchor-pair-missing", `missing ${from} → ${to}`));
      }
    }
  }
}
```

Call it once near the start of `verifyPackageIr`.

- [ ] **Step 2: Run the focused verifier and confirm RED**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run flow-verifier:ir
```

Expected: failure reports 10 slides, nine missing routing slides, and missing anchor pairs.

---

### Task 2: Add the 9×9 matrix and focused curve pages

**Files:**
- Modify: `apps/explainers/decks-flow/flow-sdk-examples.flow`
- Test: `npm run flow-verifier:ir`

**Interfaces:**
- Consumes: `sdk.Header`, `sdk.Label`, `sdk.Shape`, `sdk.Panel`, `sdk.Edge(mode = "curve")`.
- Produces: five slides titled exactly as the first five entries in `ROUTING_EXEMPLAR_TITLES`.

- [ ] **Step 1: Update deck metadata and slide numbering**

Change hub copy to mention the routing cookbook. Change existing eyebrow fractions to `of 19`. Existing slides 1–4 retain their ordinals; existing slides 5–10 become 14–19 after insertion.

- [ ] **Step 2: Generate the complete matrix scene**

Insert “Complete 9×9 curve matrix” after “Topology patterns.” Use a 9-column × 9-row grid in the 620×250 content area. For each ordered pair:

```flow
sdk.Shape(id = "s5-center-n-src", variant = "circle", x = 54, y = 94, width = 8, height = 8)
sdk.Shape(id = "s5-center-n-dst", variant = "circle", x = 92, y = 110, width = 8, height = 8)
sdk.Edge(
  id = "s5-center-n-edge",
  mode = "curve",
  from = { nodeId: "s5-center-n-src", anchor: "center" },
  to = { nodeId: "s5-center-n-dst", anchor: "n" },
  style = { stroke: theme(accent.primary), strokeWidth: 1.0, curvature: 0.32 }
)
```

Generate coordinates mechanically from row/column indices, keeping every node and curve inside the stage. Add source row labels and target column labels. Trace all 81 stable edge IDs in row-grouped timeline cues.

- [ ] **Step 3: Add cardinal and corner/center pages**

Use sparse 2×2 and compass layouts. Each edge must name its source and target anchor in a nearby `sdk.Label`. Trace edges after revealing endpoint panels.

- [ ] **Step 4: Add same-side/self-loop and obstacle pages**

The same-side page contains `n → n`, `w → w`, and one self-edge with `preferredSide`. The obstacle page contains default, high-clearance, preferred-side, and disabled-avoidance examples on separate vertical lanes.

- [ ] **Step 5: Run the verifier**

Run `npm run flow-verifier:ir`.

Expected: slide-title errors remain only for the final four pages; no findings originate from the five new scenes.

---

### Task 3: Add lanes, bundling, orthogonal, and controls pages

**Files:**
- Modify: `apps/explainers/decks-flow/flow-sdk-examples.flow`
- Test: `npm run flow-verifier:ir`

**Interfaces:**
- Consumes: the existing curved-route style controls and anchor-aware `mode = "route"`.
- Produces: the final four titles in `ROUTING_EXEMPLAR_TITLES` and a 19-slide deck.

- [ ] **Step 1: Add parallel-lane and bundle pages**

Parallel lanes use three same-endpoint curved edges with `parallelGap = 12`. Bundling uses three compatible curved edges with `bundle = true`; labels identify shared corridor and endpoint branches.

- [ ] **Step 2: Add anchor-safe orthogonal routing**

Use tall-displacement west/east and wide-displacement north/south examples. Their geometry must visibly terminate perpendicular to the target side. Add a note reading “terminal leg ⟂ component edge.”

- [ ] **Step 3: Add the six-control reference**

Use six mini-panels for `clearance`, `curvature`, `avoidObstacles`, `preferredSide`, `bundle`, and `parallelGap`. Each mini-panel contains a real edge and a concise authored-value label.

- [ ] **Step 4: Update final slides and checklist**

Renumber the original slides after the insertion, update comments, and extend the checklist copy to direct agents to the routing cookbook.

- [ ] **Step 5: Run the verifier and confirm GREEN**

Run `npm run flow-verifier:ir`.

Expected: `0 error(s)` and no warning attributable to any new routing slide.

---

### Task 4: Full verification and scope inspection

**Files:**
- Verify: `apps/explainers/decks-flow/flow-sdk-examples.flow`
- Verify: `apps/explainers/scripts/flow-verifier/ir.mjs`

**Interfaces:**
- Consumes: complete 19-slide package.
- Produces: build and verifier evidence.

- [ ] **Step 1: Run the production build**

Run `npm run build`.

Expected: TypeScript and Vite complete successfully.

- [ ] **Step 2: Run full-deck playback**

Run `npm run flow-verifier:extended`.

Expected for this deck: navigation reaches slide 19; no new geometry, missing-path, out-of-bounds, or draw-cue findings. Unrelated pre-existing deck failures are reported separately.

- [ ] **Step 3: Check diff hygiene**

Run:

```bash
git diff --check
git status --short -- \
  apps/explainers/decks-flow/flow-sdk-examples.flow \
  apps/explainers/scripts/flow-verifier/ir.mjs \
  docs/superpowers/specs/2026-07-20-routing-sdk-exemplar-pages-design.md \
  docs/superpowers/plans/2026-07-20-routing-sdk-exemplar-pages.md
```

Expected: no whitespace errors; only the routing exemplar implementation, verifier contract, spec, and plan appear in this task’s scope.
