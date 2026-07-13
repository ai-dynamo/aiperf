## Graph-First Task 4 Integration Report

### Scope integrated

- Integrated slices 4a-4d through `graph-scene.tsx`, router shared-state wiring, and typed graph state schema.
- Added integration coverage for:
  - drag/waypoint persistence into URL state,
  - reset behavior preserving non-layout scene state,
  - share URL clipboard copy + decode validation,
  - accessibility outline isolate parity with in-node trace controls,
  - pulse overlay + reduced-motion semantics.
- Added waypoint coordinate-conversion coverage to ensure viewport-space pointer input converts into flow coordinates before persistence.

### Key implemented behavior

1. **Manual visual persistence**
   - Node drag completions now persist into `GraphState.nodePositions`.
   - Edge waypoint add/move/remove/reset now persist through `GraphState.edgeWaypoints`.
   - State remains visual-only (`edgeId + points`), without semantic endpoint/fact persistence.

2. **Layout reset and authority model**
   - Reset now clears only manual layout overrides (`nodePositions`, `edgeWaypoints`) while preserving scene/flavor/expansion/focus timeline context.
   - URL remains authoritative; local storage remains synchronized from effective graph state.

3. **Expand/collapse and cleanup**
   - Collapse now removes descendant layout overrides (node positions and descendant-connected waypoint overrides), keeping collapsed topology state clean.

4. **Trace and accessibility parity**
   - In-node upstream/downstream/isolate controls now flow through shared graph state (`traceMode` + focus).
   - Accessibility outline isolate/inspect/select/expand/collapse callbacks now drive the same state model.
   - Path states visibly respond to selected trace mode.

5. **Pulse integration**
   - Pulse controls and overlay are now mounted in `GraphScene`.
   - Playback state is deterministic over Task 2 timeline semantics.
   - Exact edge references are used when present; reduced-motion keeps semantic ordering/states while disabling motion.

6. **Sharing**
   - Share now copies a compressed URL containing current graph state through clipboard when available.

### Verification

- Focused tests:
  - `vitest run src/routes/fit-request.test.tsx src/routes/graph-scene.integration.test.tsx src/features/graph/edge-waypoints.test.tsx` ✅
- Full requested validation pipeline:
  - `npm run validate:content` ✅
  - `npm run typecheck` ✅
  - `npm run lint` ✅
  - `npm test` ✅ (218 tests passed)
  - `npm run build` ✅

### Notes / residual concerns

- Vite reports an existing large chunk-size warning during production build; this predates Task 4 integration and does not block correctness.
- `npm` emits a workspace environment warning for `devdir`; it is non-blocking and does not affect Atlas test/build results.

### Collapse cleanup follow-up

- Added a failing integration regression test before the fix. The RED result showed collapse removed the collapsed node's own manual position (`expected node.runtime-composition, received []`).
- Split collapse cleanup into:
  - descendants-only node IDs for removing hidden descendant positions,
  - descendant-connected edge IDs for removing hidden waypoint overrides,
  - hidden-subtree entity IDs for focus and trace cleanup.
- The collapsed node's manual position and its non-descendant-connected waypoint overrides now remain intact.
- Focused integration verification passes: `graph-scene.integration.test.tsx` (17 tests).

### Pulse-path and waypoint review fixes

- Added an end-to-end failing scene regression before integration. The RED result showed the live graph canvas had no exact active edge metadata for `edge.runtime.dispatch.metrics`.
- `GraphScene` now derives one typed pulse-edge overlay from the Task 2 timeline and passes it to `GraphCanvas`; exact edge references take precedence over channel/flavor fallback.
- Runtime graph edges receive active/completed edge and channel state. Animated particles follow the rendered path, including persisted waypoint segments.
- Added a second red-green regression proving edges that merely share an active channel do not animate when an exact edge reference selects another edge.
- Reduced-motion rendering uses the same active/completed identities and channels, replacing motion with a static marker.
- Waypoint pointer dragging now captures the pointer and releases it on pointer up or cancellation.
- Focused Task 4 review tests pass: 5 files, 42 tests.

### Completed pulse finalization

- Added a failing normal-motion completed-edge regression before the fix. The RED result showed completed edges incorrectly reported `data-motion="animated"` and contained infinite `animateMotion`.
- Only `active` edge phases now render infinite waypoint-aware `animateMotion`.
- Completed edges preserve completed edge/channel identity and render a static green midpoint marker in both normal and reduced-motion modes.
- Removed the dead `.graph-edge-dynamo-online` CSS selector; planned Dynamo-online paths remain dashed coral through `.graph-edge-planned`.
- Focused pulse verification passes: 3 files, 26 tests.
