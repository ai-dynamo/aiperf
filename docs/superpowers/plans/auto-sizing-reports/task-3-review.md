# Task 3 Review — Verifier Layout Parity

**Date:** 2026-07-20
**Reviewer:** Cursor agent (Composer)
**Plan:** `docs/superpowers/plans/2026-07-20-diagram-node-auto-sizing.md` (Task 3)
**Implementer report:** `docs/superpowers/plans/auto-sizing-reports/task-3-report.md`
**Scout:** `docs/superpowers/plans/auto-sizing-reports/task-2-3-scout.md`

---

## Verdict

| Dimension | Result |
|---|---|
| **Spec compliance** | **PASS** |
| **Code quality** | **APPROVED** (with Important follow-ups) |

Task 3 delivers verifier layout parity for the TypeScript path. Focused tests pass. No Critical defects. Two Important maintainability gaps should be tracked, not re-litigated in this task.

---

## Verification checklist

### 1. `verify-geometry.ts` resolves capability layout bottom-up

**PASS.** `verify-geometry.ts` re-exports `resolveSceneWorldGeometry` as `indexResolvedWorldGeometry` from `capabilities/resolved-geometry.js`. That module:

1. Recursively resolves each child via `resolveCapabilityLayout` before the parent runs (`resolveChildren`).
2. Applies parent `childGeometries` as overrides during the world-space visit.
3. Translates local children using the same `childrenAreLocal` heuristics as SceneRenderer’s `resolve-scene.ts` (coordinateSpace, layout.*, panel/lane/stepper capabilities, fit-parent probe, group fallback).

This matches the plan’s required bottom-up contract and the scout’s recommended visit order.

### 2. `verify-deck.ts` consumes resolved world bounds

**PASS.** `verifyPackageIr` now:

```282:289:apps/explainers/src/flow/dev-tools/verify-deck.ts
    const authoredNodes = walkNodes(roots);
    const worldGeometryById = indexResolvedWorldGeometry(roots);
    const nodes = authoredNodes.map((node) => {
      const geometry = worldGeometryById.get(node.id);
      return geometry === undefined
        ? node
        : ({ ...node, geometry } as RenderNodeIr);
    });
```

Downstream box collection, connector snapping, fan resolution, dot centers, and viewport checks iterate `nodes` with materialized world geometry. Connector endpoints and obstacle boxes therefore see intrinsic expansion and rail reflow.

### 3. Regression tests

**PASS.** New `verify-geometry.test.ts` (3 cases):

| Case | Covers |
|---|---|
| Rail + auto-grown chips | Intrinsic child width, gap-8 reflow, world translation at `(20, 30)` |
| Stack + semantic circle | Column reflow and `core.circle` intrinsic bounds in world space |
| Node verifier integration | `materializeSceneWorldGeometry` + `ir.mjs` `verifyPackageIr` — no zero-area / missing-geometry / out-of-viewport findings |

Exceeds plan minimum (dedicated file vs. layout-contract comment only).

**Reviewer run:**

```text
npm --prefix apps/explainers test -- src/flow/dev-tools/verify-geometry.test.ts
```

Result: 3/3 pass, exit 0.

### 4. Full explainer test + build gates

**PARTIAL (non-blocking for Task 3 scope).** Implementer report documents exit 1/2 on full suite and build due to unrelated in-progress managed-layout / semantic-IR work. Task 3 files themselves lint clean (`git diff --check` exit 0). Acceptable given orthogonal failures; re-run full gates once sibling work lands.

### 5. `.mjs` flow-verifier deferral

**PASS (correctly deferred).** `scripts/flow-verifier/geometry.mjs` and `ir.mjs` remain authored-geometry-only, as the plan allows when TS registry import would require a large rewrite.

Mitigation added outside Task 3 file list: `compile-decks.ts` now pipes verifier output through `materializeSceneWorldGeometry`, so the Node CLI verifier receives pre-resolved roots. The third regression test exercises that bridge.

---

## Findings

### Critical

None.

### Important

1. **Duplicate geometry indexers — drift risk.** SceneRenderer resolves via `resolution/resolve-scene.ts` (`resolveScene`). The verifier imports a parallel copy in `capabilities/resolved-geometry.ts` (`resolveSceneWorldGeometry`). The visit loops and `resolveChildren` / `resolveLayoutChildren` helpers are near-identical (~200 lines duplicated). `capabilityOf` fallbacks differ slightly (`node.kind ?? "core.group"` vs. `core.${kind}`). Today’s rail/chip and stack/circle tests pass on both paths, but future layout heuristic edits must be applied twice until consolidated (scout Option C: extract shared indexer).

2. **Node verifier API still assumes materialized IR.** `ir.mjs` / `geometry.mjs` call raw `geomOf`. Parity holds when callers use `compile-decks.ts` (CI) or pre-materialize roots; direct `verifyPackageIr(unmaterializedPkg)` from `ir.mjs` alone would still verify authored-only bounds. Documented follow-up per plan; not a Task 3 regression, but the gap remains until `.mjs` imports shared logic or always materializes.

### Minor

1. **`resolveEndpoint` / `resolveFanEndpoint` in `verify-geometry.ts`** still read `geomOf(node)` rather than accepting a world-geometry index. Safe today because `verify-deck.ts` materializes nodes before building `nodesById`; direct helper callers without materialization would snap to authored boxes.

2. **Unrelated diff hunk** in `verify-deck.ts` (`hasAbsoluteConnector` optional chaining on `from`/`to`) — harmless defensive fix, outside Task 3 scope.

3. **Elbow obstacle wiring** added to `verify-geometry.ts` `arrowPathData` — improves routing parity, not required by Task 3 plan; low risk.

---

## Spec coverage (Task 3)

| Plan requirement | Status |
|---|---|
| Verifier uses same layout rules as SceneRenderer | ✅ TS path via `indexResolvedWorldGeometry` |
| Pass pre-resolved child geometries into parents | ✅ `resolveChildren` bottom-up |
| World-coordinate index for nested local children | ✅ visit + `childrenAreLocal` |
| Optional `.mjs` parity or documented deferral | ✅ Deferred; compile-decks bridge documented |
| Focused regression assertion | ✅ `verify-geometry.test.ts` |
| Full test + build | ⚠️ Task 3 tests green; full suite blocked by unrelated work |

---

## Summary

Task 3 meets the plan: browser/TS verification now resolves intrinsic chip/panel sizing and container reflow before collecting bounds, matching SceneRenderer’s bottom-up layout contract. `verify-deck.ts` was correctly updated. Node `.mjs` deferral is justified and partially bridged via compile-time materialization. Consolidate `resolved-geometry.ts` with `resolve-scene.ts` in a follow-up to eliminate duplication drift.
