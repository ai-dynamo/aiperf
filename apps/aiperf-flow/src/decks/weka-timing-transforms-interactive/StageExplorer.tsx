/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pipeline stage-by-stage explorer, ported from `StageExplorer`/`STAGES` in
//! `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`. The nine stages form a real
//! linear pipeline, so — unlike the time-scaled Gantt views in this deck — this is exactly the
//! shape React Flow's node/edge vocabulary is for: a chain of `panel` nodes joined by `flow`
//! edges, clickable to select the active stage shown in the detail card below.

import { useMemo, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";

export type Stage = {
  id: string;
  name: string;
  timing: boolean;
  symbol: string;
  detail: string;
};

export const STAGES: Stage[] = [
  {
    id: "flatten",
    name: "Flatten requests",
    timing: false,
    symbol: "_flatten_requests",
    detail:
      "DFS over requests. Each n/s becomes a _Node with its spawner, joined_causes, chain_prev, and async_ancestors. Structural context only — no clock math yet.",
  },
  {
    id: "content",
    name: "Content trie",
    timing: false,
    symbol: "_resolve_content_parents",
    detail:
      "Hash-id prefix tree resolves each node's content_parent (longest full/partial prefix). Drives prompt/KV lineage and segment ids — explicitly NOT a timing dependency.",
  },
  {
    id: "warp",
    name: "Idle-gap warp",
    timing: true,
    symbol: "_ActiveIdleWarp.map",
    detail:
      "Collapses true dead air over the union of active intervals to at most `cap` seconds (default 60s, from synthesis.idle_gap_cap_seconds; null disables it). Stamps warped_start on every node. Never cuts inside a request, so warped_end = warped_start + api_time always holds.",
  },
  {
    id: "ranks",
    name: "Compute ranks",
    timing: true,
    symbol: "_compute_ranks",
    detail:
      "Global total order by (warped start, end, node_id). This rank breaks ties and defines which of two overlapping turns is 'earlier' for edge orientation.",
  },
  {
    id: "edges",
    name: "Interval-order edges",
    timing: true,
    symbol: "_build_interval_edges",
    detail:
      "A -> B iff A finished-before B (raw clock), rank(A) < rank(B), and B is not an async subtree exclusion. Binding cause carries delay_after_predecessor_us = max(0, B.start - A.end); other frontier causes are AND-fan-in waits (delay 0).",
  },
  {
    id: "arrival",
    name: "Arrival offsets",
    timing: true,
    symbol: "arrival_offset_us",
    detail:
      "Each node's warped start in microseconds. Anchors ordinal sort, snapshot partition, and START-rooted min_start_delay_us for concurrent turns.",
  },
  {
    id: "persist",
    name: "Persist ordinals",
    timing: false,
    symbol: "trie_node_ordinals",
    detail:
      "Build plane writes per-ordinal delta envelopes. Ordinals share the exact (arrival_offset_us, id) sort key with the schedule plane, so build and replay agree.",
  },
  {
    id: "chop",
    name: "t* snapshot chop",
    timing: true,
    symbol: "chop_trie_at_tstar",
    detail:
      "Runtime: drop nodes with arrival_offset_us < t* (warmed, not profiled). Survivors that lost all predecessors re-root to START at min_start_delay = arrival - t*. Prompt path is kept whole.",
  },
  {
    id: "burst",
    name: "Burst collapse",
    timing: true,
    symbol: "_burst_collapse_leading_offsets",
    detail:
      "Runtime overlay: zeros leading min_start_delay_us to synchronize phase starts. Does NOT touch inter-turn delay_after_predecessor_us gaps.",
  },
];

// Module-level (stable identity) ELK options — the stages form a left-to-right linear chain.
const STAGE_ELK_OPTS: ElkOptions = { direction: "RIGHT" };

function buildNodes(selected: string): Node[] {
  return STAGES.map((stage) => ({
    id: stage.id,
    type: "panel",
    // Placeholder position (ignored by ELK); satisfies the React Flow `Node` type only.
    position: { x: 0, y: 0 },
    data: {
      title: stage.name,
      detail: stage.timing ? "timing transform" : "structural",
      strokeRole: stage.id === selected ? "primary" : "secondary",
    },
  }));
}

const FLOW_EDGES: Edge[] = STAGES.slice(1).map((stage, i) => ({
  id: `e-${STAGES[i]!.id}-${stage.id}`,
  source: STAGES[i]!.id,
  target: stage.id,
  type: "flow",
}));

/**
 * Nine-stage pipeline chain rendered as a real React Flow graph (`panel` nodes + `flow` edges).
 * Clicking a stage updates the detail card below with its symbol and description.
 */
// Inner graph: runs the shared ELK layout and re-applies the ELK-computed positions onto the
// live (per-selection) node `data` so the selected-stage highlight stays live. Must be inside a
// `ReactFlowProvider`.
function StageGraph({
  sel,
  onSelect,
}: {
  sel: string;
  onSelect: (id: string) => void;
}): React.JSX.Element {
  const nodes = useMemo(() => buildNodes(sel), [sel]);
  const { nodes: laid, laidOut } = useElkLayout(nodes, FLOW_EDGES, STAGE_ELK_OPTS);
  const posById = useMemo(() => new Map(laid.map((n) => [n.id, n.position])), [laid]);
  const positioned = useMemo(
    () => nodes.map((n) => ({ ...n, position: posById.get(n.id) ?? n.position })),
    [nodes, posById],
  );
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={positioned}
      edges={FLOW_EDGES}
      onNodeClick={(_, node) => onSelect(node.id)}
      nodesDraggable={false}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

export function StageExplorer(): React.JSX.Element {
  const [sel, setSel] = useState<string>("warp");
  const active = STAGES.find((s) => s.id === sel) ?? STAGES[0]!;

  return (
    <Stack gap={12}>
      <div style={{ height: 140 }}>
        <ReactFlowProvider>
          <StageGraph sel={sel} onSelect={setSel} />
        </ReactFlowProvider>
      </div>

      <div>
        <Row align="center" gap={10}>
          <h3 className={`text-sm font-semibold ${inkClassName("primary")}`}>{active.name}</h3>
          <span className={`text-xs font-semibold uppercase tracking-wide ${inkClassName(active.timing ? "link" : "tertiary")}`}>
            {active.timing ? "timing transform" : "structural"}
          </span>
        </Row>
        <div className="mt-2">
          <Code inline>{active.symbol}</Code>
        </div>
        <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>{active.detail}</p>
      </div>
    </Stack>
  );
}
