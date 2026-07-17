# Cellular Architecture Actual-Code Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the cellular architecture canvas so every topology, status, source link, symbol, and proof reflects the current Rust implementation.

**Architecture:** Preserve the existing 20-page story and interactive full atlas, while evolving its data model and layout for newly built surfaces. Treat `rust/runtime/src/engine/`, `rust/runtime/src/cellular/`, `rust/runtime/src/transport/`, `rust/cli/`, and current e2e tests as authoritative.

**Tech Stack:** TypeScript, React, `cursor/canvas`, inline SVG, Rust source/test evidence.

## Global Constraints

- Modify only the existing canvas and this implementation plan.
- Import only from `cursor/canvas`; keep all data inline.
- Use host theme tokens only; no gradients, emojis, or shadows.
- Mark runtime-supported behavior without product e2e proof as partial, not built.
- Keep barrier-free START distinct from the planned T3 external sink.
- Do not alter Rust code or tests.

---

### Task 1: Repair the evidence catalog

**Files:**
- Modify: `docs/canvases/cellular-architecture.canvas.tsx`

**Interfaces:**
- Consumes: current `NODES`, `EDGES`, `STORY_STEPS`, and `ABILITIES` constants.
- Produces: source-resolvable paths, symbols, and proof strings used by the inspector and story margin.

- [x] Replace every `rust/runtime/src/runner_protocol/` path with `rust/runtime/src/engine/`.
- [x] Replace `rust/runtime/src/transport_http/` with `rust/runtime/src/transport/http/`.
- [x] Correct renamed symbols: `ProfileFlags`, `DatasetInputAdapterResolver`, and `GrpcExecutionFactory`.
- [x] Align artifact, phaser, dataset fan-out, and cell-entry symbols with their defining files.
- [x] Replace unrelated proof references with the current cellular unit or e2e tests.

### Task 2: Add current built architecture

**Files:**
- Modify: `docs/canvases/cellular-architecture.canvas.tsx`

**Interfaces:**
- Consumes: the corrected evidence catalog from Task 1.
- Produces: atlas nodes/edges and story copy for current control, data, and artifact planes.

- [x] Add shared timing origin (`cell_origin.rs`, `AIPERF_CELL_SHARED_ORIGIN`) as a built control-plane capability.
- [x] Add native Kubernetes `controller` / `cell` / `aggregator` roles as built entry points with partial cluster-level proof.
- [x] Split the artifact plane into record lane, Stage E artifact shipping, and Stage G dataset serving.
- [x] Clarify graph+gRPC, graph duration, graph adaptive, K8s tree, and side-telemetry proof boundaries.
- [x] Keep the external no-central-merge sink planned and visually separate from barrier-free START.

### Task 3: Reflow and validate the interactive story

**Files:**
- Modify: `docs/canvases/cellular-architecture.canvas.tsx`

**Interfaces:**
- Consumes: updated nodes, edges, story steps, and ability rows.
- Produces: a valid 20-page progressive story and full atlas with no orphaned references.

- [x] Reposition added nodes and route edges without obscuring existing lanes.
- [x] Update story pages to introduce the new built surfaces without increasing the page count.
- [x] Update route derivation and warnings for the revised topology.
- [x] Run the canvas TypeScript diagnostic and fix all errors.
- [x] Run repository lint diagnostics for the canvas.
- [x] Verify every built node opens an existing source path and every named proof exists.

### Task 4: Final verification

**Files:**
- Verify: `docs/canvases/cellular-architecture.canvas.tsx`

**Interfaces:**
- Consumes: completed canvas.
- Produces: evidence that the artifact compiles and its factual catalog is internally consistent.

- [x] Run:

```bash
source "/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate"
python - <<'PY'
from pathlib import Path

canvas = Path("docs/canvases/cellular-architecture.canvas.tsx").read_text()
assert "src/runner_protocol/" not in canvas
assert "src/transport_http/" not in canvas
assert "NativeGrpcExecutionBackendFactory" not in canvas
assert "RunnerDatasetInputAdapterResolver" not in canvas
PY
```

- [x] Confirm the canvas retains clear visual hierarchy, varied composition, and none of the forbidden slop patterns.

Verification notes:
- `esbuild` parsed the TSX successfully.
- Workspace diagnostics only reported the repository's missing `cursor/canvas` and JSX ambient types; no canvas-specific syntax error was present.
- A post-edit filesystem audit verified all 30 atlas source paths/symbols, named proofs, and story visibility invariants.
