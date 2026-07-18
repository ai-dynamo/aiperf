# Explainer Canvas Ports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add seven narrated website decks that port every AIPerf product canvas while preserving the existing three explainer routes.

**Architecture:** Each port is a self-contained `DeckDefinition` under `apps/explainers/src/decks/`, with co-located slide copy, deck-local SVG scenes, and shared playback through `ExplainerShell`. The central registry drives routing and the hub; tests enforce deck identity, route uniqueness, slide counts, and narration completeness.

**Tech Stack:** React 19, TypeScript 5.8 strict mode, Vite 6, React Router hash routes, Vitest, React Testing Library, SVG/CSS animation, Web Speech API.

## Global Constraints

- Port the seven product canvases; exclude `canvas-repo-layout.canvas.tsx`.
- Retain existing `rust-architecture`, `slurm-velo`, and `dynosim` routes.
- Label overlapping ports as “atlas” or “deep dive.”
- Keep narration approximately 20–45 seconds per slide.
- Use the existing 700×400 responsive SVG viewport and reduced-motion behavior.
- Ground architecture claims in current executable Rust and manifests.
- Do not remove or replace source canvases.
- Do not reproduce all interactive simulators or all algorithm pages verbatim.

---

### Task 1: Registry contract and failing content tests

**Files:**
- Modify: `apps/explainers/src/test/registry.test.ts`
- Modify: `apps/explainers/src/core/deck-registry.ts`

**Interfaces:**
- Consumes: `DeckDefinition` from `src/core/types.ts`
- Produces: ten entries in `DECK_REGISTRY`, addressable through `deckByRoute(route)`

- [ ] **Step 1: Add a failing registry test**

Assert these IDs and slide counts:

```typescript
const expected = new Map([
  ["rust-architecture", 16],
  ["slurm-velo", 16],
  ["dynosim", 18],
  ["rust-architecture-atlas", 11],
  ["velo-deep-dive", 10],
  ["cellular-internals", 20],
  ["cellular-algorithms", 16],
  ["dynosim-offline", 7],
  ["segment-pools", 6],
  ["mock-server", 10],
]);
```

For every deck assert the exact slide count, unique ID/route, non-empty title,
narration, caption, and points.

- [ ] **Step 2: Run the registry test and verify it fails**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm test -- src/test/registry.test.ts
```

Expected: FAIL because the seven IDs are absent.

- [ ] **Step 3: Add imports and registry entries as each deck lands**

Keep `DECK_REGISTRY` ordered as overview decks first, then deep dives:
Rust overview, Rust atlas, Segment Pools, Mock Server, SLURM/Velo, Velo deep
dive, Cellular internals, Cellular algorithms, Dynosim overview, Dynosim
offline.

### Task 2: Shared diagram primitives

**Files:**
- Create: `apps/explainers/src/core/diagram/SceneBox.tsx`
- Create: `apps/explainers/src/core/diagram/FlowArrow.tsx`
- Create: `apps/explainers/src/core/diagram/MotionSignal.tsx`
- Test: `apps/explainers/src/test/diagram.test.tsx`

**Interfaces:**
- Produces: `SceneBox(props)`, `FlowArrow(props)`, and `MotionSignal(props)`
- Consumes: `useHostTheme()` and SVG-native attributes

- [ ] **Step 1: Test primitives render semantic SVG output**

Render each primitive inside `<svg>` and assert its title/detail text, path
data, marker reference, and motion class.

- [ ] **Step 2: Run the test and verify missing-module failure**

- [ ] **Step 3: Implement neutral shared primitives**

`SceneBox` accepts coordinates, title, detail, and optional category accent.
`FlowArrow` accepts `d`, marker ID, and optional color. `MotionSignal` accepts a
path, color, duration, and delay. Keep deck-specific fork/join choreography
inside deck modules.

- [ ] **Step 4: Run diagram tests**

Expected: PASS.

### Task 3: Rust architecture atlas and Segment Pools decks

**Files:**
- Create: `apps/explainers/src/decks/rust-architecture-atlas/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Create: `apps/explainers/src/decks/segment-pools/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Read source: `docs/canvases/rust-aiperf-architecture.canvas.tsx`
- Read source: `docs/canvases/segment-pools-and-body-plans.canvas.tsx`

**Interfaces:**
- Produces: `rustArchitectureAtlasDeck`, `segmentPoolsDeck`

- [ ] **Step 1: Author eleven atlas slides**

Use canvas order: system, processes, runtime, protocol, scheduled, graph,
endpoints, metrics, cellular, builds, seams. Distinguish this deck from the
existing overview through source-file labels and implementation vocabulary.

- [ ] **Step 2: Author six Segment Pools slides**

Use canvas order: overview, interning, payload domains, BodyPlan splicing,
prefix addressing, dispatch precedence.

- [ ] **Step 3: Implement one SVG scene per slide**

Use shared primitives and deck-local scene data. Preserve canvas relationships,
not canvas component code.

- [ ] **Step 4: Register both decks and run registry/build checks**

Expected: registry still fails only for the five remaining decks; TypeScript
build passes.

### Task 4: Velo deep dive and Cellular internals decks

**Files:**
- Create: `apps/explainers/src/decks/velo-deep-dive/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Create: `apps/explainers/src/decks/cellular-internals/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Read source: `docs/canvases/velo-in-aiperf.canvas.tsx`
- Read source: `docs/canvases/cellular-architecture.canvas.tsx`

**Interfaces:**
- Produces: `veloDeepDiveDeck`, `cellularInternalsDeck`

- [ ] **Step 1: Author ten Velo mechanism slides**

Connection, registration, START, MessagePack, heartbeat, partition, merge,
phaser, dataset distribution, aggregator hierarchy.

- [ ] **Step 2: Author twenty Cellular internals slides**

Preserve the five chapters and page order: Launch 5, Distribute 5, Execute 5,
Reduce 3, Scale 2.

- [ ] **Step 3: Implement scenes with control/load/result lane consistency**

Green is request/load flow, purple is results, yellow is coordination authority,
and blue is timing/runtime context.

- [ ] **Step 4: Register both decks and run tests/build**

### Task 5: Cellular algorithm workbook deck

**Files:**
- Create: `apps/explainers/src/decks/cellular-algorithms/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Read source: `docs/canvases/cellular-algorithm-workbook.canvas.tsx`

**Interfaces:**
- Produces: `cellularAlgorithmsDeck`

- [ ] **Step 1: Distill sixteen narrated slides**

Cover workbook orientation, eligibility, ownership, control, distribution,
execution, capture, merge, artifacts, composition, and decisions. Preserve
algorithm names and evidence vocabulary, but do not create 100 slides.

- [ ] **Step 2: Implement chapter-map and decision-flow scenes**

Show status semantics (built, partial, feature-gated) with labels in addition to
color. Final slide points to `docs/canvases/cellular-algorithm-workbook.canvas.tsx`
for exhaustive lookup.

- [ ] **Step 3: Register and verify**

### Task 6: Dynosim offline and Mock Server decks

**Files:**
- Create: `apps/explainers/src/decks/dynosim-offline/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Create: `apps/explainers/src/decks/mock-server/{content.ts,MentalModel.tsx,styles.ts,index.tsx}`
- Read source: `docs/canvases/dynosim-offline-flow.canvas.tsx`
- Read source: `docs/canvases/mock-server-architecture.canvas.tsx`
- Verify claims: `rust/runtime/src/dynosim.rs`
- Verify claims: `rust/runtime/src/graph/runtime.rs`
- Verify claims: `rust/mock-server/src/`

**Interfaces:**
- Produces: `dynosimOfflineDeck`, `mockServerDeck`

- [ ] **Step 1: Author seven offline Dynosim slides**

Overview, launch/preflight, two seams, simulation loop, request/token dispatch,
parity gate, topology builder.

- [ ] **Step 2: Author ten Mock Server chapter slides**

Orientation, ingress, LLM protocols, specialized endpoints, gRPC/Riva, timing,
scheduler/cache, faults/semantics, observability/deployment, proof/boundaries.

- [ ] **Step 3: Implement dual-clock, token, protocol, and timing scenes**

Keep the offline deck visually distinct from broad Dynosim and the mock deck
focused on test-author mental models rather than listing every flag.

- [ ] **Step 4: Register both decks**

Expected: registry test now reports only Segment/other tasks already resolved;
once all tasks are complete, ten decks pass.

### Task 7: Hub labeling, regression tests, and publication

**Files:**
- Modify: `apps/explainers/src/routes/Hub.tsx`
- Modify: `apps/explainers/src/test/registry.test.ts`
- Modify: `apps/explainers/README.md`
- Use: `apps/explainers/scripts/deploy-pages.sh`

**Interfaces:**
- Consumes: complete ten-deck registry
- Produces: published Pages artifact with all routes

- [ ] **Step 1: Add hub grouping**

Group cards under Overview, Runtime & data, Distributed execution, and Deep
references. Keep registry as route source; hub grouping may derive from a
`hub.group` field added to `DeckHubMeta`.

- [ ] **Step 2: Run all tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm test
```

Expected: all tests PASS.

- [ ] **Step 3: Run strict production build**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run build
```

Expected: `tsc --noEmit` and Vite build PASS.

- [ ] **Step 4: Preview and smoke-test hash routes**

Verify all ten routes render a start gate, first slide, narration text, and
working Home link. Verify Back restarts the selected slide and reduced-motion
styles suppress animated signals.

- [ ] **Step 5: Publish Pages**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
bash scripts/deploy-pages.sh
```

Expected: tests/build pass and `gh-pages` reports `Published`.
