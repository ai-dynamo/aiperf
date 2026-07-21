/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `docs/canvases/slurm-architecture.canvas.tsx` (a real, hand-authored Cursor Canvas)
//! onto aiperf-flow's component vocabulary. Single-view canvas — no in-deck page tabs — so this
//! is one composed component: a rank ribbon, a React Flow architecture diagram with a
//! source-grounded inspector, a lifecycle strip, a traffic-plane table, and deployment
//! invariants. The hand-drawn SVG diagram becomes a real `Header`/`Panel`/`Card` node graph with
//! `FlowEdge` edges; clicking a rank or a diagram node updates the inspector, mirroring the
//! canvas's `useCanvasState` selection.

import { useMemo, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { Code } from "../../prose/Code.js";
import { inkClassName, surfaceClassName, strokeClassName } from "../../theme/tokens.js";
import { Eyebrow } from "../../prose/Eyebrow.js";

type Focus = "all" | "launch" | "control" | "results";
type Selection = "allocation" | "dispatcher" | "controller" | "cells" | "velo" | "artifacts";

type SourceItem = {
  label: string;
  path: string;
  detail: string;
};

// Verbatim from `docs/canvases/slurm-architecture.canvas.tsx`'s `SOURCES` record.
const SOURCES: Record<Selection, SourceItem> = {
  allocation: {
    label: "SLURM allocation",
    path: "rust/runtime/src/engine/slurm_topology.rs",
    detail:
      "Resolves rank, task count, controller host, dense cell IDs, and the shared tcp://HOST:PORT coordinate from SLURM_*.",
  },
  dispatcher: {
    label: "Native rank dispatch",
    path: "rust/cli/src/slurm.rs",
    detail:
      "Every task enters the same command. Rank 0 projects Config v2 as the controller; every other rank enters --cell.",
  },
  controller: {
    label: "Cellular controller",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    detail:
      "Binds Velo, slices the benchmark envelope, waits for registrations, releases START, and merges terminal results.",
  },
  cells: {
    label: "Autonomous cells",
    path: "rust/runtime/src/engine/cellular_cell.rs",
    detail:
      "Each cell fetches its slice over Velo, owns scheduling and transport state, dispatches load, and ships its partition.",
  },
  velo: {
    label: "Velo cellular transport",
    path: "rust/runtime/src/cellular/transport/velo_transport.rs",
    detail:
      "Carries registration, sliced envelopes, START synchronization, heartbeats, and terminal record or folded-store partitions.",
  },
  artifacts: {
    label: "Bulk artifact plane",
    path: "rust/runtime/src/engine/artifact_shipping.rs",
    detail: "Per-record artifact bytes use a separate HTTP/1 + zstd path; they are not carried on Velo.",
  },
};

// Diagram node id -> the `Selection` it represents in the inspector. Several diagram nodes
// (e.g. the two rank-space role boxes) map to the same selection key, mirroring the canvas's
// per-node `onClick={() => onSelect(...)}` handlers.
const NODE_SELECTION: Record<string, Selection> = {
  "sbatch-srun": "allocation",
  "slurm-topology": "allocation",
  dispatcher: "dispatcher",
  "controller-role": "controller",
  "cells-role": "cells",
  "controller-box": "controller",
  "cell-box": "velo",
  "controller-merge": "controller",
  "artifact-upload": "artifacts",
};

// Band membership for the TRACE focus filter, mirroring the canvas's launch/control/results
// opacity gating.
const LAUNCH_NODES = new Set(["header-launch", "sbatch-srun", "slurm-topology", "dispatcher", "controller-role", "cells-role"]);
const CONTROL_NODES = new Set(["header-control", "controller-box", "cell-box", "inference-servers"]);
const RESULTS_NODES = new Set(["header-results", "controller-merge", "authoritative-outputs", "artifact-upload"]);

function nodeFocusBand(id: string): Focus {
  if (LAUNCH_NODES.has(id)) return "launch";
  if (CONTROL_NODES.has(id)) return "control";
  if (RESULTS_NODES.has(id)) return "results";
  return "all";
}

function isNodeActive(id: string, focus: Focus): boolean {
  if (focus === "all") return true;
  return nodeFocusBand(id) === focus;
}

const BASE_NODES: Node[] = [
  { id: "header-launch", type: "header", position: { x: 0, y: 0 }, data: { title: "ALLOCATION + ROLE DISPATCH" } },
  {
    id: "sbatch-srun",
    type: "panel",
    position: { x: 0, y: 90 },
    data: { title: "sbatch / srun", detail: "N identical tasks" },
  },
  {
    id: "slurm-topology",
    type: "panel",
    position: { x: 260, y: 90 },
    data: { title: "SLURM_* topology", detail: "rank · ntasks · nodelist" },
  },
  {
    id: "dispatcher",
    type: "panel",
    position: { x: 520, y: 90 },
    data: { title: "aiperf slurm run", detail: "native rank dispatch" },
  },
  {
    id: "controller-role",
    type: "card",
    position: { x: 800, y: 60 },
    data: { title: "rank 0 → controller", detail: "profile --cells N−1" },
  },
  {
    id: "cells-role",
    type: "card",
    position: { x: 800, y: 170 },
    data: { title: "ranks 1…N−1 → cells", detail: "--cell · no config read" },
  },

  { id: "header-control", type: "header", position: { x: 0, y: 300 }, data: { title: "CELLULAR CONTROL + EXECUTION" } },
  {
    id: "controller-box",
    type: "panel",
    position: { x: 0, y: 390 },
    data: {
      title: "CONTROLLER",
      detail: "rank 0 · bind Velo :9500 · slice envelope · await registrations · trigger START · merge results",
    },
  },
  {
    id: "cell-box",
    type: "panel",
    position: { x: 340, y: 380 },
    data: {
      title: "CELL k / N−1",
      detail: "separate process · autonomous runtime · connect → register → fetch slice → await START → heartbeat → ship partition",
    },
  },
  {
    id: "inference-servers",
    type: "card",
    position: { x: 720, y: 400 },
    data: { title: "Inference servers", subtitle: "HTTP / gRPC endpoints", detail: "Velo is never on this path" },
  },

  { id: "header-results", type: "header", position: { x: 0, y: 560 }, data: { title: "TERMINAL RESULTS" } },
  {
    id: "controller-merge",
    type: "panel",
    position: { x: 0, y: 650 },
    data: { title: "Controller merge", detail: "records or folded stores · one partition per cell" },
  },
  {
    id: "authoritative-outputs",
    type: "card",
    position: { x: 260, y: 650 },
    data: { title: "Authoritative outputs", subtitle: "report · JSON / CSV / Parquet", detail: "global cellular result" },
  },
  {
    id: "artifact-upload",
    type: "panel",
    position: { x: 560, y: 650 },
    data: { title: "Bulk artifact upload", detail: "HTTP/1 + zstd · separate plane · per-record files, then concat" },
  },
];

const BASE_EDGES: Edge[] = [
  { id: "e-sbatch-topology", source: "sbatch-srun", target: "slurm-topology", type: "flow" },
  { id: "e-topology-dispatcher", source: "slurm-topology", target: "dispatcher", type: "flow" },
  { id: "e-dispatcher-controller-role", source: "dispatcher", target: "controller-role", type: "flow" },
  { id: "e-dispatcher-cells-role", source: "dispatcher", target: "cells-role", type: "flow" },
  {
    id: "e-controller-register",
    source: "controller-box",
    target: "cell-box",
    type: "flow",
    label: "register + peer info",
  },
  {
    id: "e-controller-envelope",
    source: "controller-box",
    target: "cell-box",
    type: "flow",
    label: "sliced envelope + START",
    data: { speed: "slow" },
  },
  {
    id: "e-cell-heartbeat",
    source: "cell-box",
    target: "controller-box",
    type: "flow",
    label: "heartbeat + terminal partition",
  },
  { id: "e-cell-servers", source: "cell-box", target: "inference-servers", type: "flow", label: "request hot path" },
  { id: "e-cell-merge", source: "cell-box", target: "controller-merge", type: "flow" },
  {
    id: "e-cell-artifacts",
    source: "cell-box",
    target: "artifact-upload",
    type: "flow",
    data: { speed: "slow" },
  },
  { id: "e-merge-outputs", source: "controller-merge", target: "authoritative-outputs", type: "flow" },
];

const RANKS: readonly [rank: string, role: string, detail: string, selection: Selection][] = [
  ["RANK 0", "Controller", "reads Config v2", "controller"],
  ["RANK 1", "Cell 0", "cell_id = rank − 1", "cells"],
  ["RANK 2", "Cell 1", "cell_id = rank − 1", "cells"],
  ["RANK N−1", "Cell N−2", "dense cell IDs", "cells"],
];

function RankRibbon({ onSelect }: { onSelect: (selection: Selection) => void }): React.JSX.Element {
  return (
    <Stack gap={8}>
      <Row align="center" gap={8}>
        <span className={`text-sm font-semibold ${inkClassName("tertiary")}`}>SLURM TASK SPACE</span>
        <span className={`text-sm ${inkClassName("quaternary")}`}>cell_count = SLURM_NTASKS − 1</span>
      </Row>
      <Grid columns={4} gap={8}>
        {RANKS.map(([rank, role, detail, selection]) => (
          <button
            key={rank}
            type="button"
            onClick={() => onSelect(selection)}
            className={`rounded-none border border-t-2 px-3 py-2.5 text-left ${surfaceClassName("elevated")} ${strokeClassName("secondary")}`}
          >
            <div className={`text-xs font-semibold ${inkClassName("tertiary")}`}>{rank}</div>
            <div className={`mt-1 text-sm font-semibold ${inkClassName("primary")}`}>{role}</div>
            <div className={`mt-0.5 text-xs ${inkClassName("secondary")}`}>{detail}</div>
          </button>
        ))}
      </Grid>
    </Stack>
  );
}

function Inspector({ selection }: { selection: Selection }): React.JSX.Element {
  const source = SOURCES[selection];
  return (
    <div className={`rounded-none border p-4 ${surfaceClassName("elevated")} ${strokeClassName("secondary")}`}>
      <Eyebrow>Selected architecture unit</Eyebrow>
      <Stack gap={10} className="mt-3">
        <div className={`text-sm font-semibold ${inkClassName("primary")}`}>{source.label}</div>
        <div className={`text-sm ${inkClassName("secondary")}`}>{source.detail}</div>
        <div className={`text-xs font-semibold ${inkClassName("tertiary")}`}>IMPLEMENTATION</div>
        <Code inline>{source.path}</Code>
      </Stack>
    </div>
  );
}

const LIFECYCLE_STEPS: readonly [number: string, title: string, detail: string][] = [
  ["1", "Resolve", "SLURM_* → role + coordinate"],
  ["2", "Register", "cells connect over Velo"],
  ["3", "Release", "START after readiness barrier"],
  ["4", "Execute", "cells drive HTTP / gRPC load"],
  ["5", "Reduce", "partitions merge into one report"],
];

function LifecycleStrip(): React.JSX.Element {
  return (
    <div className={`border-y ${strokeClassName("secondary")}`}>
      <Grid columns={5} gap={0}>
        {LIFECYCLE_STEPS.map(([number, title, detail], index) => (
          <div
            key={number}
            className={index === 0 ? "px-3.5 py-3" : `border-l px-3.5 py-3 ${strokeClassName("tertiary")}`}
          >
            <Row gap={7} align="center">
              <span className="text-sm font-bold text-accent-primary">{number}</span>
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</span>
            </Row>
            <div className={`mt-1 text-sm ${inkClassName("tertiary")}`}>{detail}</div>
          </div>
        ))}
      </Grid>
    </div>
  );
}

const PLANE_COLUMNS: TableColumn[] = [
  { key: "transport", label: "Transport" },
  { key: "direction", label: "Direction" },
  { key: "responsibility", label: "Responsibility" },
];

const PLANE_ROWS: TableRow[] = [
  {
    transport: "Velo",
    direction: "Cell ↔ controller",
    responsibility: "register · envelope · START · heartbeat · partition/store",
  },
  {
    transport: "HTTP / gRPC",
    direction: "Cell ↔ inference server",
    responsibility: "per-request and per-token benchmark hot path",
  },
  {
    transport: "HTTP/1 + zstd",
    direction: "Cell → controller",
    responsibility: "bulk per-record artifact files and completion barrier",
  },
];

const INVARIANTS: readonly [title: string, detail: string][] = [
  [
    "SLURM launches; AIPerf coordinates",
    "SlurmLauncher creates no processes. It expects sibling srun tasks and uses controller timeouts as a backstop.",
  ],
  [
    "One fact bootstraps every cell",
    "All tasks derive the same rank-0 host plus AIPERF_CONTROLLER_PORT coordinate; no discovery service is required.",
  ],
  [
    "Cells never parse the authored config",
    "Only rank 0 reads Config v2. Cells fetch strict, already-sliced protocol-v2 envelopes from the controller.",
  ],
];

const FOCUS_OPTIONS: readonly [Focus, string][] = [
  ["all", "End to end"],
  ["launch", "Rank dispatch"],
  ["control", "Control + execution"],
  ["results", "Results"],
];

/**
 * Ports `docs/canvases/slurm-architecture.canvas.tsx` — the AIPerf SLURM cellular architecture
 * explainer — onto aiperf-flow's component vocabulary. A single-view canvas: rank ribbon, a
 * React Flow architecture diagram with a source-grounded inspector, a lifecycle strip, a
 * traffic-plane table, and deployment invariants.
 */
export function SlurmArchitectureDeck(): React.JSX.Element {
  const [focus, setFocus] = useState<Focus>("all");
  const [selection, setSelection] = useState<Selection>("controller");

  const nodes = useMemo<Node[]>(
    () =>
      BASE_NODES.map((node) => ({
        ...node,
        style: { opacity: isNodeActive(node.id, focus) ? 1 : 0.3 },
      })),
    [focus],
  );

  const edges = useMemo<Edge[]>(
    () =>
      BASE_EDGES.map((edge) => ({
        ...edge,
        style: { opacity: isNodeActive(edge.source, focus) && isNodeActive(edge.target, focus) ? 1 : 0.15 },
      })),
    [focus],
  );

  const handleNodeClick = (_event: unknown, node: Node): void => {
    const next = NODE_SELECTION[node.id];
    if (next !== undefined) {
      setSelection(next);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="SLURM Cellular Architecture" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={`mx-auto min-h-full max-w-6xl px-10 py-8 ${surfaceClassName("page")}`}>
          <Stack gap={20}>
            <Row align="start" gap={18} wrap>
              <Stack gap={5} className="max-w-[900px]">
                <span className={`text-sm font-semibold ${inkClassName("tertiary")}`}>
                  AIPERF · SLURM CELLULAR ARCHITECTURE
                </span>
                <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
                  One allocation. One controller. Autonomous load cells.
                </h1>
                <p className={`max-w-[900px] text-sm ${inkClassName("secondary")}`}>
                  Every SLURM task runs the same native command. Rank alone selects the controller or a cell, while
                  AIPerf preserves its existing cellular protocol, worker-local hot path, and global result contract.
                </p>
              </Stack>
              <Stack gap={4} className="ml-auto min-w-[260px]">
                <span className={`text-right text-sm ${inkClassName("quaternary")}`}>
                  Source: current Rust implementation + docs/velo.md
                </span>
                <span className={`text-right text-sm ${inkClassName("quaternary")}`}>Audited 2026-07-17</span>
              </Stack>
            </Row>

            <div className={`border-t ${strokeClassName("secondary")}`} />

            <RankRibbon onSelect={setSelection} />

            <Row gap={7} align="center" wrap>
              <span className={`text-sm font-semibold ${inkClassName("tertiary")}`}>TRACE</span>
              {FOCUS_OPTIONS.map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={focus === id}
                  onClick={() => setFocus(id)}
                  className={
                    focus === id
                      ? "rounded-none border border-accent-primary bg-accent-primary px-3 py-1 text-xs font-semibold text-white"
                      : `rounded-none border px-3 py-1 text-xs font-semibold ${strokeClassName("secondary")} ${inkClassName("secondary")}`
                  }
                >
                  {label}
                </button>
              ))}
              <span className={`ml-auto text-sm ${inkClassName("quaternary")}`}>
                Select a node for implementation evidence.
              </span>
            </Row>

            <div className="grid grid-cols-1 gap-[18px] lg:grid-cols-[1.65fr_0.55fr]">
              <div className={`overflow-x-auto rounded-none border ${strokeClassName("secondary")}`} style={{ height: 760 }}>
                <ReactFlow
                  nodeTypes={nodeTypes}
                  edgeTypes={edgeTypes}
                  nodes={nodes}
                  edges={edges}
                  onNodeClick={handleNodeClick}
                  fitView
                  fitViewOptions={{ padding: 0.1 }}
                  proOptions={{ hideAttribution: true }}
                >
                  <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
                </ReactFlow>
              </div>
              <Inspector selection={selection} />
            </div>

            <LifecycleStrip />

            <Stack gap={10}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Three traffic planes</h2>
              <Table columns={PLANE_COLUMNS} rows={PLANE_ROWS} />
            </Stack>

            <Stack gap={10}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Deployment invariants</h2>
              <Grid columns={3} gap={12}>
                {INVARIANTS.map(([title, detail]) => (
                  <div key={title} className={`border-t-2 pt-2.5 ${strokeClassName("primary")}`}>
                    <div className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</div>
                    <div className={`mt-1 text-sm ${inkClassName("secondary")}`}>{detail}</div>
                  </div>
                ))}
              </Grid>
            </Stack>

            <div className={`border-t ${strokeClassName("secondary")}`} />

            <Row gap={10} align="center" wrap>
              <Code inline>controller = rank 0</Code>
              <Code inline>cell_id = rank − 1</Code>
              <Code inline>cell_count = ntasks − 1</Code>
              <span className={`ml-auto text-sm ${inkClassName("quaternary")}`}>
                Minimum allocation: 2 tasks · default Velo bootstrap port: 9500
              </span>
            </Row>
          </Stack>
        </div>
      </div>
    </div>
  );
}
