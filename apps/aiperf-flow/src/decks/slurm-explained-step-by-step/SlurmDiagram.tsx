/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-step React Flow diagram for the SLURM + Velo walkthrough. Each of the 16 canvas
//! `stepIndex` SVG scenes is re-authored here as a real node/edge graph using aiperf-flow's
//! Header/Panel/Card/Chip node vocabulary and animated `flow` edges — the hand-drawn SVG boxes,
//! labels, and connectors from `slurm-explained-step-by-step.canvas.tsx` translated into the
//! app's real diagram primitives (no hand-computed pixel motion; category color emphasis and the
//! animated `FlowEdge` dashes replace the canvas's SVG pulse/motion flourishes).

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import type { CategoryRole } from "../../theme/tokens.js";

// Accent border emphasis for a highlighted node. Kept as complete literal strings so Tailwind's
// JIT scanner emits every class (see the aiperf-flow-diagrams SKILL "Tailwind JIT trap").
const HIGHLIGHT_BORDER: Record<CategoryRole, string> = {
  green: "border-2 border-category-green",
  yellow: "border-2 border-category-yellow",
  purple: "border-2 border-category-purple",
  blue: "border-2 border-category-blue",
  red: "border-2 border-category-red",
  orange: "border-2 border-category-orange",
  cyan: "border-2 border-category-cyan",
  gray: "border-2 border-category-gray",
};

// Category color CSS variables for edge strokes.
const EDGE_COLOR: Record<CategoryRole, string> = {
  green: "var(--color-category-green)",
  yellow: "var(--color-category-yellow)",
  purple: "var(--color-category-purple)",
  blue: "var(--color-category-blue)",
  red: "var(--color-category-red)",
  orange: "var(--color-category-orange)",
  cyan: "var(--color-category-cyan)",
  gray: "var(--color-category-gray)",
};

function hl(role: CategoryRole): string {
  return HIGHLIGHT_BORDER[role];
}

type Diagram = { nodes: Node[]; edges: Edge[] };

// --- Small node factories, so each scene reads as its box list. ---------------------------------

function header(id: string, x: number, y: number, title: string, caption?: string): Node {
  return { id, type: "header", position: { x, y }, data: { title, caption } };
}
function panel(
  id: string,
  x: number,
  y: number,
  title: string,
  detail?: string,
  accent?: CategoryRole,
): Node {
  return {
    id,
    type: "panel",
    position: { x, y },
    data: { title, detail, className: accent ? hl(accent) : undefined },
  };
}
function card(
  id: string,
  x: number,
  y: number,
  title: string,
  subtitle?: string,
  detail?: string,
  accent?: CategoryRole,
): Node {
  return {
    id,
    type: "card",
    position: { x, y },
    data: { title, subtitle, detail, className: accent ? hl(accent) : undefined },
  };
}
function chip(id: string, x: number, y: number, label: string, accent?: CategoryRole): Node {
  return {
    id,
    type: "chip",
    position: { x, y },
    data: { label, className: accent ? hl(accent) : undefined },
  };
}
function flow(
  id: string,
  source: string,
  target: string,
  label?: string,
  accent?: CategoryRole,
): Edge {
  return {
    id,
    source,
    target,
    type: "flow",
    label,
    data: accent ? { color: EDGE_COLOR[accent] } : undefined,
  };
}

const COL = 300;

/**
 * The 16 per-step diagrams, indexed by `stepIndex` (0..15). Each entry ports the corresponding
 * canvas SVG scene into real nodes/edges. Positions are declared hints; React Flow's `fitView`
 * reflows them to the viewport, so the layout intent (left→right pipelines, fan-out/fan-in) is
 * preserved without pixel-matching the SVG coordinates.
 */
export const DIAGRAMS: readonly Diagram[] = [
  // 0 — BENCHMARK GOAL: many load generators → inference server.
  {
    nodes: [
      panel("gen", 0, 60, "Many load generators", "more machines = more traffic", "green"),
      card("server", COL, 50, "Inference server", "the AI model endpoint", undefined, "green"),
    ],
    edges: [flow("e", "gen", "server", "REQUESTS", "green")],
  },
  // 1 — SLURM RESERVES RESOURCES: scheduler → allocation of 4 task machines.
  {
    nodes: [
      panel("slurm", 0, 90, "SLURM scheduler", "finds and reserves machines", "yellow"),
      header("alloc", COL, 0, "Your allocation"),
      chip("t0", COL, 60, "Task 0 · machine"),
      chip("t1", COL + 150, 60, "Task 1 · machine"),
      chip("t2", COL, 120, "Task 2 · machine"),
      chip("t3", COL + 150, 120, "Task 3 · machine"),
    ],
    edges: [flow("e", "slurm", "alloc", undefined, "green")],
  },
  // 2 — ONE COMMAND, MANY TASKS: identical command forks to N tasks by PROCID.
  {
    nodes: [
      header("cmd", 180, 0, "aiperf slurm run --config benchmark.yaml", "one identical command on every task"),
      card("t0", 0, 120, "Task 0", "SLURM_PROCID = 0", "rank decides the role", "green"),
      card("t1", COL, 120, "Task 1", "SLURM_PROCID = 1", "rank decides the role", "green"),
      card("t2", 2 * COL, 120, "Task 2", "SLURM_PROCID = 2", "rank decides the role", "green"),
    ],
    edges: [
      flow("e0", "cmd", "t0", undefined, "green"),
      flow("e1", "cmd", "t1", undefined, "green"),
      flow("e2", "cmd", "t2", undefined, "green"),
    ],
  },
  // 3 — RANKS BECOME AIPERF ROLES: rank 0 controller vs ranks 1..N cells.
  {
    nodes: [
      card("ctrl", 0, 60, "Rank 0", "CONTROLLER", "coordinates the benchmark", "yellow"),
      card("cells", COL, 40, "Ranks 1, 2, 3…", "LOAD CELLS", "cell 0 · cell 1 · cell 2… send benchmark requests", "green"),
    ],
    edges: [flow("e", "ctrl", "cells", "SAME RUN", "gray")],
  },
  // 4 — EVERY CELL FINDS RANK 0: nodelist → derived coordinate → every cell dials.
  {
    nodes: [
      panel("nodelist", 0, 60, "SLURM nodelist", "node01, node02… + port 9500"),
      card("coord", COL, 55, "Controller coordinate", "tcp://node01:9500", undefined, "green"),
      panel("dial", 2 * COL, 50, "Every cell dials", "the same address and reaches rank 0 — no discovery service", "green"),
    ],
    edges: [
      flow("e0", "nodelist", "coord", undefined, "green"),
      flow("e1", "coord", "dial", undefined, "green"),
    ],
  },
  // 5 — VELO = CONTROL MESSAGING: controller ↔ Velo ↔ load cells.
  {
    nodes: [
      card("ctrl", 0, 40, "Controller", "rank 0", "binds Velo listener on port 9500", "yellow"),
      card("velo", COL, 55, "VELO", "control walkie-talkie", undefined, "green"),
      card("cells", 2 * COL, 40, "Load cells", "ranks 1…N", "dial controller over Velo", "green"),
    ],
    edges: [
      flow("e0", "ctrl", "velo", "SEND", "green"),
      flow("e1", "velo", "cells", "REPLY", "green"),
    ],
  },
  // 6 — VELO HANDSHAKE: cell knows addr → velo.connect → peers registered → ready.
  {
    nodes: [
      card("k", 0, 60, "Cell knows", undefined, "tcp://node01:9500"),
      card("c", COL, 60, "velo.connect", undefined, "hello handshake", "green"),
      card("p", 2 * COL, 60, "Peers registered", undefined, "controller ↔ cell"),
      card("r", 3 * COL, 60, "Ready to talk", undefined, "named handlers"),
    ],
    edges: [
      flow("e0", "k", "c", undefined, "green"),
      flow("e1", "c", "p", undefined, "green"),
      flow("e2", "p", "r", undefined, "green"),
      flow("e3", "r", "k", "HELLO REPLY RETURNS TO THE CELL", "green"),
    ],
  },
  // 7 — REGISTER + START OVER VELO: connect → register → slice → await → START.
  {
    nodes: [
      card("connect", 0, 60, "Connect", undefined, "Velo hello", "green"),
      card("register", COL, 60, "Register", undefined, "aiperf.cell.register", "green"),
      card("slice", 2 * COL, 60, "Get slice", undefined, "sliced envelope", "green"),
      card("await", 3 * COL, 60, "Await START", undefined, "EventHandle", "green"),
      card("start", 4 * COL, 60, "START!", undefined, "all cells release", "green"),
    ],
    edges: [
      flow("e0", "connect", "register", undefined, "green"),
      flow("e1", "register", "slice", undefined, "green"),
      flow("e2", "slice", "await", undefined, "green"),
      flow("e3", "await", "start", undefined, "green"),
    ],
  },
  // 8 — HOT PATH IS NOT VELO: load cells → inference server over HTTP/gRPC; Velo heartbeats only.
  {
    nodes: [
      card("cells", 0, 40, "Load cells", "send requests", "measure replies", "green"),
      card("server", COL, 40, "Inference server", "AI endpoint", undefined, "green"),
      panel("hb", COL / 2, 190, "Velo heartbeats only", "progress summaries → rank 0", "green"),
    ],
    edges: [
      flow("e", "cells", "server", "HTTP / gRPC · NOT Velo", "green"),
      flow("hb", "cells", "hb", undefined, "green"),
    ],
  },
  // 9 — THREE TRAFFIC PLANES: three stacked lanes.
  {
    nodes: [
      panel("velo", 0, 0, "VELO — control", "register · START · heartbeat · partition", "green"),
      panel("http", 0, 110, "HTTP / gRPC — load", "benchmark requests to the AI server", "green"),
      panel("bulk", 0, 220, "HTTP/1 + zstd — bulk files", "large per-record artifact uploads", "purple"),
    ],
    edges: [],
  },
  // 10 — RESULTS OVER VELO: cells → velo ship → rank-0 controller → one report.
  {
    nodes: [
      card("cells", 0, 60, "Load cells", undefined, "finish local work", "green"),
      card("ship", COL, 60, "Velo ship", undefined, "partition / store", "green"),
      card("ctrl", 2 * COL, 60, "Rank-0 controller", undefined, "receives + merges", "green"),
      card("report", 3 * COL, 60, "One report", undefined, "global benchmark", "purple"),
    ],
    edges: [
      flow("e0", "cells", "ship", undefined, "green"),
      flow("e1", "ship", "ctrl", undefined, "green"),
      flow("e2", "ctrl", "report", undefined, "green"),
    ],
  },
  // 11 — BULK ARTIFACTS SEPARATE: Velo path vs HTTP artifact path.
  {
    nodes: [
      panel("velo", 0, 60, "VELO PATH", "small messages · register · START · heartbeat · partition", "green"),
      panel("http", COL, 60, "HTTP ARTIFACT PATH", "large files · per-record exports · zstd upload + concat", "purple"),
    ],
    edges: [],
  },
  // 12 — CONTROLLER ROLE VS NODE: dedicated role vs dedicated node.
  {
    nodes: [
      panel("role", 0, 60, "Dedicated ROLE", "always useful · keeps merge + sync off the measured path", "yellow"),
      panel("node", COL, 60, "Dedicated NODE", "optional · default script uses one task per node"),
    ],
    edges: [],
  },
  // 13 — FAN-OUT: rank 0 global plan → Velo → cell slices.
  {
    nodes: [
      card("plan", 0, 90, "Rank 0", "GLOBAL PLAN", "requests 0…N", "yellow"),
      chip("velo", COL, 100, "Velo", "green"),
      panel("c0", 2 * COL, 20, "Cell 0 · slice 0", "disjoint owned work", "green"),
      panel("c1", 2 * COL, 100, "Cell 1 · slice 1", "disjoint owned work", "green"),
      panel("c2", 2 * COL, 180, "Cell 2 · slice 2", "disjoint owned work", "green"),
    ],
    edges: [
      flow("ev", "plan", "velo", undefined, "green"),
      flow("e0", "velo", "c0", undefined, "green"),
      flow("e1", "velo", "c1", undefined, "green"),
      flow("e2", "velo", "c2", undefined, "green"),
    ],
  },
  // 14 — FAN-IN: cell partitions → Velo → rank 0 merge.
  {
    nodes: [
      panel("c0", 0, 20, "Cell 0 · partition 0", "finished local result", "green"),
      panel("c1", 0, 100, "Cell 1 · partition 1", "finished local result", "green"),
      panel("c2", 0, 180, "Cell 2 · partition 2", "finished local result", "green"),
      chip("velo", COL, 100, "Velo", "green"),
      card("merge", 2 * COL, 90, "Rank 0", "MERGE", "one global report", "purple"),
    ],
    edges: [
      flow("e0", "c0", "velo", undefined, "green"),
      flow("e1", "c1", "velo", undefined, "green"),
      flow("e2", "c2", "velo", undefined, "green"),
      flow("em", "velo", "merge", undefined, "purple"),
    ],
  },
  // 15 — FROM CONFIG TO REPORT: config → generate → submit → SLURM+Velo → report.
  {
    nodes: [
      card("config", 0, 60, "Config v2", undefined, "benchmark.yaml"),
      card("generate", COL, 60, "Generate", undefined, "job.sbatch"),
      card("submit", 2 * COL, 60, "Submit", undefined, "sbatch job.sbatch"),
      card("run", 3 * COL, 60, "SLURM + Velo", undefined, "roles + messaging"),
      card("report", 4 * COL, 60, "AIPerf report", undefined, "merged result", "purple"),
    ],
    edges: [
      flow("e0", "config", "generate", undefined, "green"),
      flow("e1", "generate", "submit", undefined, "green"),
      flow("e2", "submit", "run", undefined, "green"),
      flow("e3", "run", "report", undefined, "green"),
    ],
  },
];

/**
 * Renders the React Flow diagram for a single walkthrough step. Keyed by `stepIndex` upstream so
 * React Flow refits the viewport whenever the active scene changes.
 */
export function SlurmDiagram({ stepIndex }: { stepIndex: number }): React.JSX.Element {
  const diagram = DIAGRAMS[stepIndex] ?? DIAGRAMS[0];
  return (
    <div style={{ height: 360 }} className="border border-stroke-secondary bg-surface-elevated">
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={diagram.nodes}
        edges={diagram.edges}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
      </ReactFlow>
    </div>
  );
}
