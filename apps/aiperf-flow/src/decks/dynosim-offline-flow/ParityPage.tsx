/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Callout } from "../../prose/Callout.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `ParityPage`: AIPerf accumulates
//! its own summary from the observer stream; Dynamo produces its own from the engine. The run is
//! rejected unless the two serialize to identical bytes.

const nodes: Node[] = [
  { id: "aiperf", type: "card", position: { x: 0, y: 0 }, data: { title: "AIPerf", subtitle: "Collector" } },
  { id: "dynamo", type: "card", position: { x: 0, y: 120 }, data: { title: "Dynamo", subtitle: "take_report_at" } },
  {
    id: "comparator",
    type: "panel",
    position: { x: 300, y: 60 },
    data: { title: "canonical_shared_metric_bytes", detail: "finish_shared_metrics → verify_parity" },
  },
  {
    id: "outcome",
    type: "card",
    position: { x: 620, y: 60 },
    data: { title: "byte-equal", subtitle: "74 fields (+3 goodput)" },
  },
];

const edges: Edge[] = [
  { id: "e-aiperf-comparator", source: "aiperf", target: "comparator", type: "flow" },
  { id: "e-dynamo-comparator", source: "dynamo", target: "comparator", type: "flow" },
  { id: "e-comparator-outcome", source: "comparator", target: "outcome", type: "flow" },
];

const FIELD_COLUMNS: TableColumn[] = [
  { key: "field", label: "Field" },
  { key: "source", label: "Source" },
];

const FIELD_ROWS: TableRow[] = [
  { field: "ttft", source: "AIPerf" },
  { field: "itl", source: "AIPerf" },
  { field: "e2e", source: "AIPerf" },
  { field: "throughput", source: "AIPerf" },
  { field: "sessions", source: "AIPerf" },
  { field: "tokens", source: "AIPerf" },
  { field: "prefill_s", source: "Dynamo (engine)" },
  { field: "gpu_hours", source: "Dynamo (engine)" },
];

/**
 * The verification gate page: AIPerf's own summary vs Dynamo's own summary, compared field by
 * field, rejecting the run unless the two serialize to identical bytes (69 own fields, 5 from the
 * engine, +3 goodput fields when an SLA is set).
 */
export function ParityPage({ level }: { level: Level }): React.JSX.Element {
  const dev = atLeast(level, "developer");
  const maint = atLeast(level, "maintainer");
  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>The verification gate</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          AIPerf accumulates its own summary from the observer stream; Dynamo produces its own
          from the engine. The run is rejected unless the two serialize to identical bytes.
        </p>
      </div>

      <div style={{ height: 260 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      {maint && (
        <div>
          <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>
            Representative compared fields (69 own + 5 from engine)
          </h3>
          <Table columns={FIELD_COLUMNS} rows={FIELD_ROWS} />
        </div>
      )}

      {dev && (
        <Callout tone="warning" title="Mismatch → the run bails">
          Any differing field fails <strong>finish_shared_metrics</strong> in the library and
          again at <strong>verify_parity</strong> in <strong>offline_execution</strong>.
          Per-request rows are excluded from the compare; goodput adds 3 fields when an SLA is
          set.
        </Callout>
      )}
    </Stack>
  );
}
