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
