/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";

// Ported from docs/canvases/segment-pools-and-body-plans.canvas.tsx `PageOverview`
// (rust/aiperf/src/dataset/{segment.rs, model.rs, dataset.rs, compose.rs},
// rust/aiperf/src/body_plan.rs). Three bands: BUILD (mutable) -> FREEZE ->
// DISPATCH (hot path).

const nodes: Node[] = [
  {
    id: "band-build",
    type: "header",
    position: { x: 0, y: 0 },
    data: { title: "BUILD (mutable)" },
  },
  {
    id: "band-freeze",
    type: "header",
    position: { x: 460, y: 0 },
    data: { title: "FREEZE" },
  },
  {
    id: "band-dispatch",
    type: "header",
    position: { x: 760, y: 0 },
    data: { title: "DISPATCH (hot path)" },
  },

  {
    id: "dataset-source",
    type: "panel",
    position: { x: 0, y: 80 },
    data: { title: "Dataset source", detail: "JSON / CSV / HF / trace" },
  },
  {
    id: "composer-compose",
    type: "panel",
    position: { x: 0, y: 200 },
    data: { title: "Composer.compose", detail: "intern rows → pool" },
  },
  {
    id: "apply-common-contexts",
    type: "panel",
    position: { x: 0, y: 320 },
    data: { title: "apply_common_contexts", detail: "system / user_context" },
  },

  {
    id: "segment-pool",
    type: "card",
    position: { x: 300, y: 190 },
    data: { title: "SegmentPool", subtitle: "arena: Vec<Segment>", detail: "ids: HashMap<Id,Handle>" },
  },

  {
    id: "in-memory-segment-store",
    type: "card",
    position: { x: 560, y: 200 },
    data: { title: "InMemorySegmentStore", subtitle: "Box<[Segment]> (frozen)", detail: "ids map dropped" },
  },

  {
    id: "dataset",
    type: "card",
    position: { x: 800, y: 80 },
    data: { title: "Dataset", subtitle: "Arc<dyn SegmentStore>", detail: "+ body_plans cache" },
  },
  {
    id: "precompute-body-plans",
    type: "panel",
    position: { x: 800, y: 200 },
    data: { title: "precompute_body_plans", detail: "BodyPlan per static turn" },
  },
  {
    id: "json-body-materializer",
    type: "panel",
    position: { x: 800, y: 320 },
    data: { title: "JsonBodyMaterializer", detail: "splice handles → Bytes" },
  },
  {
    id: "transport",
    type: "panel",
    position: { x: 800, y: 440 },
    data: { title: "Transport", detail: "HTTP / gRPC dispatch" },
  },
];

const edges: Edge[] = [
  { id: "e-source-compose", source: "dataset-source", target: "composer-compose", type: "flow" },
  { id: "e-compose-contexts", source: "composer-compose", target: "apply-common-contexts", type: "flow" },
  { id: "e-compose-pool", source: "composer-compose", target: "segment-pool", type: "flow" },
  { id: "e-contexts-pool", source: "apply-common-contexts", target: "segment-pool", type: "flow" },
  {
    id: "e-pool-store",
    source: "segment-pool",
    target: "in-memory-segment-store",
    type: "flow",
    label: ".freeze()",
    data: { speed: "slow" },
  },
  { id: "e-store-dataset", source: "in-memory-segment-store", target: "dataset", type: "flow" },
  { id: "e-dataset-precompute", source: "dataset", target: "precompute-body-plans", type: "flow" },
  { id: "e-precompute-materializer", source: "precompute-body-plans", target: "json-body-materializer", type: "flow" },
  { id: "e-materializer-transport", source: "json-body-materializer", target: "transport", type: "flow" },
  {
    id: "e-store-materializer",
    source: "in-memory-segment-store",
    target: "json-body-materializer",
    type: "flow",
    label: "store.get(handle) → wire bytes",
    data: { speed: "slow" },
  },
];

/**
 * Overview page of the Segment Pools & Body Plans explainer deck.
 *
 * Ports `PageOverview` from `docs/canvases/segment-pools-and-body-plans.canvas.tsx`
 * onto aiperf-flow's node/edge vocabulary: the hand-drawn SVG boxes become
 * `panel`/`card` nodes and the animated SVG paths become `flow` edges. Shows
 * the same three-band pipeline — dataset row intern into a mutable
 * `SegmentPool`, `.freeze()` into an `InMemorySegmentStore`, then
 * `BodyPlan` materialization splicing handles into wire bytes on dispatch.
 */
export function OverviewPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">The pipeline: rows in → wire bytes out</h2>
        <p className="mt-1 max-w-3xl text-sm text-[var(--color-ink-secondary)]">
          Every request body AIPerf sends starts as a dataset row and ends as pre-spliced bytes. Two data structures
          own the middle: the <strong>SegmentPool</strong> (content-addressed, deduplicated storage) and the{" "}
          <strong>BodyPlan</strong> (a shape that says which handles fill which JSON fields). The design invariant is{" "}
          <strong>serialize content once, splice bytes forever</strong>.
        </p>
      </div>

      <div style={{ height: 560 }}>
        <ReactFlow nodeTypes={nodeTypes} edgeTypes={edgeTypes} nodes={nodes} edges={edges} fitView>
        </ReactFlow>
      </div>

      <div className="rounded-none border border-[var(--color-stroke-secondary)] px-4 py-3 text-sm">
        <div className="font-semibold">The two representations</div>
        <p className="mt-1 text-[var(--color-ink-secondary)]">
          A <code>Handle(u32)</code> is the public address — a dense arena index. A <code>SegmentId([u8;32])</code> is
          the blake3 content hash used only to deduplicate while the pool is mutable. When the pool freezes into an{" "}
          <code>InMemorySegmentStore</code>, the hash→handle map is thrown away; handles stay valid.
        </p>
      </div>
    </div>
  );
}
