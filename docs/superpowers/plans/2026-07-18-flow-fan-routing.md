<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Fan Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship first-class `core.fan-out` / `core.fan-in` topology through schema → compiler → SceneRenderer, fix arrow/ball semantics platform-wide, and migrate all nine explainer decks onto the new routing model.

**Architecture:** Add a `FanNodeIr` that survives lowering as one node. SceneRenderer resolves trunk + junction + branches once, deduplicates shared segments, paints semantic arrowheads, and drives split/merge ball timing from the same geometry. Extend flow-verifier to understand routes, elbows, and fans. Migrate decks heaviest-first on fan topology, then normalize remaining connectors.

**Tech Stack:** TypeScript (aiperf-flow schema/language/compiler), React SVG SceneRenderer (`apps/explainers`), `.flow` decks, DeckPackage JSON, flow-verifier (IR + Playwright).

**Spec:** `docs/superpowers/specs/2026-07-18-flow-fan-routing-design.md`

## Global Constraints

- Fan representation: first-class topology in package IR and `SceneRenderer` (not desugared to overlapping connectors)
- Deck coverage: all nine files under `apps/explainers/decks-flow/`, including `tstar-warmup.flow`
- Arrowheads: only at semantic destinations (fan branch targets / single fan-in target)
- Fan-out balls: one incoming traveler duplicates onto all outgoing branches at the junction
- Fan-in balls: branch travelers converge at the junction, then one traveler leaves
- Default route: orthogonal, perimeter-anchored, shared trunk rendered once
- Verification: no new or modified tests; use builds, package generation, IR verifier, and Playwright verifier gates
- Generated packages: rebuild from corrected `.flow` sources only
- Work from repo root: `/home/anthony/nvidia/projects/aiperf/ajc/rust`
- Preserve NVIDIA SPDX headers on new/edited source files
- Do not create git commits unless explicitly requested

---

## File map

| Area | Primary files |
|---|---|
| Schema | `apps/aiperf-flow/packages/schema/src/ir.ts`, `capability.ts`, `index.ts` |
| Language | `apps/aiperf-flow/packages/language/src/embedded-scene.ts`, `tokens.ts`, `ast.ts`, `parser.ts`, `grammar/explainer.ts` |
| Compiler | `apps/aiperf-flow/packages/compiler/src/lower-explainer-scene.ts`, `desugar-scene-primitives.ts` |
| Renderer | `apps/explainers/src/core/diagram/SceneRenderer.tsx`, `MotionSignal.tsx`, `FlowArrow.tsx` |
| Verifier | `apps/explainers/scripts/flow-verifier/geometry.mjs`, `ir.mjs`, `play.mjs` |
| Decks | `apps/explainers/decks-flow/*.flow` |
| Packages | `npm run build:explainer-packages` (from repo Makefile or explainers scripts) |

---

### Task 1: Schema — `FanNodeIr`

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/ir.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/capability.ts` (`FOUNDATION_CAPABILITIES`)
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/fan-node.test.ts`

**Interfaces:**
- Produces: `FanNodeIr`, Zod schema, `FoundationCapabilityId` entries `core.fan-out` / `core.fan-in`

- [ ] **Step 1: Write failing schema tests**

```ts
// apps/aiperf-flow/packages/schema/test/fan-node.test.ts
import { describe, expect, it } from "vitest";
import { sceneNodeSchema } from "../src/ir.js";

describe("FanNodeIr", () => {
  it("accepts fan-out with one from and two to endpoints", () => {
    const parsed = sceneNodeSchema.parse({
      id: "dispatch",
      kind: "fan",
      capability: "core.fan-out",
      from: { nodeId: "src", anchor: "e" },
      to: [
        { nodeId: "a", anchor: "w" },
        { nodeId: "b", anchor: "w" },
      ],
      axis: "x",
      layout: { x: 0, y: 0, width: 0, height: 0 },
      style: {},
      sourceMap: { source: "t", start: { offset: 0, line: 1, column: 1 }, end: { offset: 0, line: 1, column: 1 } },
    });
    expect(parsed.kind).toBe("fan");
  });

  it("rejects fan-out with only one destination", () => {
    expect(() =>
      sceneNodeSchema.parse({
        id: "bad",
        kind: "fan",
        capability: "core.fan-out",
        from: { nodeId: "src", anchor: "e" },
        to: [{ nodeId: "a", anchor: "w" }],
        layout: { x: 0, y: 0, width: 0, height: 0 },
        style: {},
        sourceMap: { source: "t", start: { offset: 0, line: 1, column: 1 }, end: { offset: 0, line: 1, column: 1 } },
      }),
    ).toThrow();
  });
});
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `cd apps/aiperf-flow/packages/schema && npm test -- fan-node.test.ts`

- [ ] **Step 3: Implement `FanNodeIr`**

Add to `ir.ts`:

```ts
export type FanNodeIr = RenderNodeBaseIr &
  Readonly<{
    kind: "fan";
    capability: "core.fan-out" | "core.fan-in";
    from: ConnectorEndpointIr | readonly ConnectorEndpointIr[];
    to: ConnectorEndpointIr | readonly ConnectorEndpointIr[];
    axis?: ConnectorAxisIr;
    junction?: Readonly<{ x: number; y: number }>;
  }>;
```

Extend `SceneNodeIr` union, Zod discriminated union, and `FoundationCapabilityId`. Add `.superRefine` cardinality checks (fan-out: scalar `from`, array `to` length ≥ 2; fan-in: array `from` length ≥ 2, scalar `to`).

Register capabilities in `capability.ts` with `nodeKinds: ["fan"]`.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Fix any union exhaustiveness breaks in existing schema tests**

---

### Task 2: Language surface

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/embedded-scene.ts`
- Modify: `apps/aiperf-flow/packages/language/src/ast.ts` (`SCENE_PRIMITIVE_CAPABILITIES`)
- Modify: `apps/aiperf-flow/packages/language/src/tokens.ts`, `grammar/explainer.ts`, `parser.ts`
- Create: `apps/aiperf-flow/packages/language/test/fan-primitives.test.ts`

**Interfaces:**
- Consumes: Task 1 capability ids and endpoint shapes
- Produces: package capture + native parse of `fan-out` / `fan-in` nodes

- [ ] **Step 1: Write failing parser tests** for package-form capture and native keywords `fan-out`, `fan-in` with `from`/`to`/`axis`/`junction`.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd apps/aiperf-flow/packages/language && npm test -- fan-primitives.test.ts`

- [ ] **Step 3: Add tokens + grammar + AST nodes**

Map native `fan-out` → `core.fan-out`, `fan-in` → `core.fan-in`. Package form accepts the fields documented in the spec.

- [ ] **Step 4: Run tests — expect PASS**

---

### Task 3: Compiler lowering

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/desugar-scene-primitives.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/lower-explainer-scene.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/fan-node-lowering.test.ts`

**Interfaces:**
- Consumes: language AST / package nodes from Task 2
- Produces: `FanNodeIr` in SceneIr (not exploded connectors)

- [ ] **Step 1: Write failing lowering test** — minimal fan-out scene lowers to exactly one `kind: "fan"` node with preserved endpoints.

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `capabilityKind()` → `"fan"`** for both capabilities; pass through in `lowerFirstClassPackageNode` without trunk/branch expansion.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Run existing compiler suite**

Run: `cd apps/aiperf-flow/packages/compiler && npm test`

---

### Task 4: SceneRenderer — fan geometry + arrows

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Create: `apps/explainers/src/test/scene-renderer-fan.test.tsx`

**Interfaces:**
- Consumes: `FanNodeIr`
- Produces: `resolveFanGeometry(node, index, layoutOrigin) → { segments, junction, trajectories }`

- [ ] **Step 1: Write failing renderer tests**

Cases:
- horizontal fan-out: one trunk, N branches, arrowheads only on branch ends
- horizontal fan-in: N branches, one trunk, one arrowhead on destination
- authored `junction` honored
- shared trunk segment rendered once (query DOM path count or `data-flow-segment-id`)

- [ ] **Step 2: Run test — expect FAIL**

Run: `cd apps/explainers && npm test -- scene-renderer-fan.test.tsx`

- [ ] **Step 3: Implement fan resolver**

Add helper module or functions inside `SceneRenderer.tsx`:

```ts
type FanSegment = Readonly<{
  id: string;
  d: string;
  directed: boolean;
  showMarker: boolean;
}>;

function resolveFanGeometry(/* … */): Readonly<{
  segments: readonly FanSegment[];
  junction: Readonly<{ x: number; y: number }>;
  trajectories: readonly Readonly<{ d: string; role: "trunk" | "branch" | "merge-trunk" }>[];
}>;
```

Rules:
- Resolve endpoints with existing `resolveEndpoint` / facing anchors
- Place junction at authored point or corridor midpoint (stable, not timeline-dependent)
- Build orthogonal trunk + branches per `axis`
- Deduplicate identical segment `d` strings before paint
- Paint with `FlowArrow`; `showMarker` true only on semantic destinations

- [ ] **Step 4: Wire fan branch in node render switch** before generic connector handling.

- [ ] **Step 5: Run fan tests — expect PASS**

---

### Task 5: SceneRenderer — topology-aware balls

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/diagram/MotionSignal.tsx` (only if multi-dot API needed)
- Extend: `apps/explainers/src/test/scene-renderer-fan.test.tsx`

**Interfaces:**
- Consumes: `resolveFanGeometry` trajectories + timeline cues (`draw`, `trace`)
- Produces: split/merge ball playback using painted paths (no straight-line override)

- [ ] **Step 1: Add failing tests** for fan-out split timing and fan-in merge timing at `trace` progress 0.25 / 0.5 / 0.75.

- [ ] **Step 2: Implement fan playback**

- `draw` on fan id reveals segments via existing stroke-reveal path
- `trace` on fan id drives balls:
  - fan-out: trunk ball 0→junction; at junction spawn branch balls; each branch 0→1
  - fan-in: normalize branch progress so all reach junction together; single outgoing ball

- [ ] **Step 3: Fix single-edge motion** — when `motion.signal` references the same endpoints as a visible connector/elbow/fan segment, reuse resolved geometry instead of `boundaryOnlyMotionPath` straight override.

- [ ] **Step 4: Reduced motion** — final static fan visible, zero traveling balls.

- [ ] **Step 5: Run fan + existing scene-renderer tests**

Run: `cd apps/explainers && npm test -- scene-renderer`

---

### Task 6: Flow verifier alignment

**Files:**
- Modify: `apps/explainers/scripts/flow-verifier/geometry.mjs`
- Modify: `apps/explainers/scripts/flow-verifier/ir.mjs`
- Modify: `apps/explainers/scripts/flow-verifier/play.mjs`

- [ ] **Step 1: Extend `ARROW_CAPS` / `ARROW_KINDS`** with `core.route`, `core.elbow`, `core.fan-out`, `core.fan-in`, kind `fan`.

- [ ] **Step 2: Port elbow path resolution** — share logic with renderer or duplicate `elbowPathData` for verifier path points.

- [ ] **Step 3: Add fan checks in `ir.mjs`**

New codes:
- `fan-invalid-cardinality` (error)
- `fan-disconnected-junction` (error)
- `fan-missing-trace-cue` (warn when fan present but no `trace` and no staggered branch motion)

- [ ] **Step 4: Add `/#/tstar-warmup` to `DECK_ROUTES`** (9/9 decks).

- [ ] **Step 5: Run IR verifier baseline**

Run: `cd apps/explainers && npm run flow-verifier:ir -- --from-flow`

Expected after later deck migration: 0 errors.

---

### Task 7: Migrate topology-heavy decks

**Files:**
- Modify: `apps/explainers/decks-flow/slurm-velo.flow`
- Modify: `apps/explainers/decks-flow/velo-deep-dive.flow`

**Rewrite targets:**

| Slide / region | Current pattern | Target |
|---|---|---|
| `slurm-velo` task fork ~332–415 | trunk + bar + drops + 4 motion signals | one `core.fan-out` + `draw`/`trace` |
| `slurm-velo` rank fan-out ~1717–1860 | `s13-br-*` path tree + staggered motion | `core.fan-out` from rank0 → cells via Velo hub junction |
| `slurm-velo` fan-in ~1963–2100 | mirror paths | `core.fan-in` |
| `velo-deep-dive` slide 9 ~980–1035 | four curved paths | `core.fan-in` with authored `junction` if auto placement insufficient |

- [ ] **Step 1: Replace fork slide** — delete redundant `motion.signal` branch duplicates; one fan node owns topology.

- [ ] **Step 2: Replace rank fan-out / fan-in slides** similarly.

- [ ] **Step 3: Rebuild packages for these two decks only** and run IR verifier scoped:

Run: `cd apps/explainers && node scripts/flow-verifier.mjs --from-flow --deck slurm-velo`

- [ ] **Step 4: Manual visual check** of fork + fan slides at junction frame.

---

### Task 8: Normalize modern decks

**Files:**
- Modify: `apps/explainers/decks-flow/segment-pools.flow`, `dynosim.flow`, `tstar-warmup.flow`

- [ ] **Step 1:** Replace manual `motion.signal` `d:` paths with anchored endpoints matching adjacent `core.connector` / `core.route`.

- [ ] **Step 2:** Convert any obvious 1→N or N→1 slide segments to fan nodes (if present).

- [ ] **Step 3:** IR verify three decks.

---

### Task 9: Normalize legacy decks

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture.flow`, `rust-architecture-atlas.flow`, `cellular-algorithms.flow`, `cellular-internals.flow`

- [ ] **Step 1:** Convert absolute `core.line` edges whose endpoints snap within 36px to two panels → `core.connector` or `core.route`.

- [ ] **Step 2:** Replace pipeline `core.path` chains with connectors where axis-aligned.

- [ ] **Step 3:** Introduce fan nodes on exporter/controller merge slides where copy implies fan-out/in (atlas metrics ~959, cellular algorithms controller fanout panel region).

- [ ] **Step 4:** `cellular-internals` last — slide-by-slide path reduction (68 paths).

---

### Task 10: Rebuild, verify, report

**Files:**
- Regenerate: `apps/explainers/src/decks-generated/*.package.json`
- Update: `.superpowers/sdd/task-6-report.md` or new fan-routing report section

- [ ] **Step 1: Rebuild all nine packages**

Run from repo root (use existing target): `make assert-explainer-packages` or explainers build script referenced in Makefile.

- [ ] **Step 2: IR gate**

Run: `cd apps/explainers && npm run flow-verifier:ir -- --from-flow`
Expected: **0 errors**

- [ ] **Step 3: Play gate**

Run: `cd apps/explainers && npm run flow-verifier`
Expected: **0 errors** on all 9 routes including `tstar-warmup`

- [ ] **Step 4: Unit tests**

Run:
- `cd apps/aiperf-flow/packages/schema && npm test`
- `cd apps/aiperf-flow/packages/compiler && npm test`
- `cd apps/explainers && npm test`

- [ ] **Step 5: Document post-migration counts** (fan nodes, remaining manual paths, motion anchored vs manual).

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| First-class `FanNodeIr` | 1, 3 |
| Native + package authoring | 2 |
| Shared trunk, deduped segments | 4 |
| Semantic arrowheads | 4 |
| Split/merge ball timing | 5 |
| `trace` vs `draw` | 5 |
| Verifier fan + route/elbow coverage | 6 |
| All nine decks migrated | 7–9 |
| Tests + IR + Play gates | 1–6, 10 |
| Play includes tstar-warmup | 6 |

No placeholder steps remain. Types consistent: `FanNodeIr`, `resolveFanGeometry`, capabilities `core.fan-out` / `core.fan-in`.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-07-18-flow-fan-routing.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach do you want?
