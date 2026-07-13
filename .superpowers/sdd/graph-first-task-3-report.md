# Graph-First Task 3 Consolidated Report

## Scope

Integrated the three parallel Task 3 slices into a single graph-first route runtime centered on `features/graph/graph-scene.tsx`, replacing active guided/card rendering on scene routes and wiring shell/router shared state to Task 2 derivation, layout, timeline, canvas, evidence, and accessibility outline behavior.

## Integrated Files

### Router and shell integration

- `apps/architecture-atlas/src/routes/router.tsx`

### Graph scene orchestration and feature surfaces

- `apps/architecture-atlas/src/features/graph/graph-scene.tsx` (new)
- `apps/architecture-atlas/src/features/graph/graph-canvas.tsx`
- `apps/architecture-atlas/src/features/graph/graph-nodes.tsx`
- `apps/architecture-atlas/src/features/graph/graph-edges.tsx`
- `apps/architecture-atlas/src/features/graph/accessibility-outline.tsx`

### Test coverage (TDD integration/route focus)

- `apps/architecture-atlas/src/routes/graph-scene.integration.test.tsx` (new)
- `apps/architecture-atlas/src/features/graph/graph-canvas.test.tsx`

## Delivered Behavior

- Default `/` now renders a real graph-first runtime scene backed by Task 2 derivation and layout input.
- All nine canonical scene routes render graph scene runtime content with shared shell controls.
- Audience/primary flavor/compare flavor/search affect derived visible topology via `deriveGraphDerivation`.
- Selection updates directed node-path highlighting and opens evidence for node/edge entities.
- Expansion/collapse/select/isolate/inspect callbacks from accessibility outline update shared graph state.
- Evidence drawer Escape/close focus restoration uses visible entity trigger fallback behavior.
- Crate drill-down remains available through shell graph search exact crate-name routing.
- Active scene routes no longer render the previous guided/card `AtlasView` content.
- Canvas duplicate edge-control list was removed in favor of outline-driven accessibility interactions.

## Validation and Test Runs

### Focused integration/tests

- `npm --prefix apps/architecture-atlas test -- src/routes/graph-scene.integration.test.tsx src/features/graph/graph-canvas.test.tsx src/features/graph/evidence-drawer.test.tsx src/features/graph/accessibility-outline.test.tsx`
  - Result: pass (`4 files`, `23 tests`).

### Requested full validation commands

- `npm --prefix apps/architecture-atlas run validate:content`
  - Fails due to pre-existing catalog evidence path mismatch:
  - `evidence file does not exist: crates/aiperf/src/dynamo_offline.rs`
- `npm --prefix apps/architecture-atlas run typecheck`
  - Fails due pre-existing non-Task-3 issues:
  - `src/features/crates/crate-reference.tsx` search type mismatch (`selected` not in route search type)
  - `src/features/guided/guided-view.tsx` stale `GuidedRoute` import
- `npm --prefix apps/architecture-atlas run lint`
  - Pass.
- `npm --prefix apps/architecture-atlas test`
  - Fails only on same pre-existing content-integrity mismatch as `validate:content`.
- `npm --prefix apps/architecture-atlas run build`
  - Fails on same pre-existing typecheck issues listed above.

## Notes / Concerns

- Task 3 implementation and focused graph-scene integration tests are green.
- Full-suite blockers are unrelated, pre-existing app/content drift outside Task 3 integration scope and should be addressed in follow-up cleanup.

## Full-Gate Follow-Up — 2026-07-12

The source and route drift above was resolved before review:

- Replaced every Atlas reference to `crates/aiperf/src/dynamo_offline.rs` with `crates/aiperf/src/dynosim.rs`.
- Re-anchored graph source evidence to current exact symbols and ranges:
  - `OfflineEngineConfig::build_native` at lines 556-628.
  - `finish_shared_metrics_enforcing` at lines 915-984.
  - `run_scheduled_backend_online` at lines 4124-4156.
- Replaced obsolete crate-related-component `selected` links with graph-first scene, search, and encoded focused-state links.
- Fixed the route-transition state race by reading pathname and search from one router-location snapshot.
- Preserved invalid shared-state recovery notices after canonical URL normalization.
- Removed the unreferenced `guided-view.tsx` implementation and its obsolete `GuidedRoute` dependency.

Failing-first regressions were added for exact DynoSim evidence and focused crate-to-graph drill-down. The existing invalid-state recovery test caught and now protects the route-transition notice behavior.

Final gate results:

- `npm --prefix apps/architecture-atlas run validate:content` — pass; 25 components, 20 edges, 23 crates.
- `npm --prefix apps/architecture-atlas run typecheck` — pass.
- `npm --prefix apps/architecture-atlas run lint` — pass.
- `npm --prefix apps/architecture-atlas test` — pass; 19 files, 180 tests.
- `npm --prefix apps/architecture-atlas run build` — pass; Vite production build completed.

The build retains Vite's advisory warning that the worker and main chunks exceed 500 kB; this is non-failing and remains a Task 4/performance-polish concern.

## Review Integration Follow-Up — 2026-07-12

Integrated the remaining Task 3 review slices:

- The default runtime scene now includes the complete Tier-0 Python-to-result journey and its canonical edges alongside the runtime seam branches.
- The command bar emits typed `GraphFitViewCommand` values through router context and `GraphScene` into `GraphCanvas`; the canvas executes each command through the React Flow `fitView` API.
- `GraphScene` now passes `derivation.overlay` directly into the canvas, which classifies shared, primary-only, and comparison-only nodes and edges.
- Built and planned state, overlay class, and directed path state reach both custom node presentation and the underlying `BaseEdge` path styling.
- Evidence focus restoration now targets the one visible command-bar graph search input; the temporary hidden duplicate input was removed.
- No accidental `artifacts/task-3-report.md` exists; this file remains the canonical Task 3 report.

The fit/overlay route test was corrected before orchestrator wiring and failed on both missing typed fit commands and missing overlay propagation. After integration, the focused review suite passed 46 tests across five files.

Final gate results after review integration:

- `npm --prefix apps/architecture-atlas run validate:content` — pass; 25 components, 20 edges, 23 crates.
- `npm --prefix apps/architecture-atlas run typecheck` — pass.
- `npm --prefix apps/architecture-atlas run lint` — pass.
- `npm --prefix apps/architecture-atlas test` — pass; 21 files, 187 tests.
- `npm --prefix apps/architecture-atlas run build` — pass; Vite production build completed.

The existing non-failing Vite warning remains: the layout worker and main application chunks exceed 500 kB.

## Final Review Follow-Up — 2026-07-12

- `GraphCanvas` now consumes each typed fit request ID exactly once. Layout, scene, flavor, search, and selection rerenders cannot replay a handled request; a new monotonically increasing request ID remains the only trigger.
- `FlavorOverlay` is now required by `GraphCanvasProps`. Node and edge classification accepts no missing overlay, and an entity absent from all overlay partitions fails loudly rather than defaulting to primary-only.
- A failing-first canvas regression proved request ID 1 replayed after a layout change, then verified the fix and that request ID 2 still executes.

Final verification:

- Focused fit/overlay tests — pass; 3 files, 8 tests.
- `npm --prefix apps/architecture-atlas run typecheck` — pass.
- `npm --prefix apps/architecture-atlas run lint` — pass.
- `npm --prefix apps/architecture-atlas test` — pass; 22 files, 188 tests.
- `npm --prefix apps/architecture-atlas run build` — pass.

The non-failing Vite large-chunk warning remains unchanged.

## Fit Lifecycle Follow-Up — 2026-07-12

- `GraphCanvas` acknowledges a completed fit through `GraphScene` to the root route.
- The root clears a command only when the acknowledged request ID matches the currently live command, so stale completions cannot erase newer work.
- Fit request sequencing is independent from live-command storage; clearing request 1 does not reuse its ID, and the next request is 2.
- Local canvas dedup remains as a second guard while route-level acknowledgement prevents replay after scene unmount/remount.
- A failing-first route regression acknowledges request 1, advances to request 2, acknowledges it, navigates scenes, and verifies the remounted canvas receives no completed command.

Final verification:

- Focused lifecycle tests — pass; 2 files, 3 tests.
- `npm --prefix apps/architecture-atlas run validate:content` — pass; 25 components, 20 edges, 23 crates.
- `npm --prefix apps/architecture-atlas run typecheck` — pass.
- `npm --prefix apps/architecture-atlas run lint` — pass.
- `npm --prefix apps/architecture-atlas test` — pass; 22 files, 188 tests.
- `npm --prefix apps/architecture-atlas run build` — pass.

The non-failing Vite large-chunk warning remains unchanged.
