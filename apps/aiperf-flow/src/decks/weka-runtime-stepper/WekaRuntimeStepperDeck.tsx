/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `docs/canvases/weka-runtime-stepper.canvas.tsx` (a real, hand-authored Cursor Canvas)
//! onto aiperf-flow's component vocabulary. Single-view canvas — an interactive step-through of
//! the async dataflow frontier as the weka TraceExecutor drives one trie trace: a root Step (A),
//! a concurrent fan-out (B, C, no edge between them), and an AND-fan-in join (D) gated on two
//! channel arrivals.

import { useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Button } from "../../prose/Button.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import {
  FRAMES,
  GATE_NEED,
  GRAPH_EDGES,
  GRAPH_NODE_IDS,
  stateLabel,
  stateOf,
  type Frame,
  type NodeState,
} from "./frames.js";
import {
  accentClassName,
  inkClassName,
  strokeClassName,
  surfaceClassName,
} from "../../theme/tokens.js";

// Module-level (stable identity) ELK options — the frontier graph is a top-to-bottom DAG
// (START -> A -> {B, C} fan-out -> D fan-in). Positions are computed from structure + measured
// sizes; the placeholder `NODE_POSITIONS` below only satisfy the React Flow `Node` type.
const FRONTIER_ELK_OPTS: ElkOptions = { direction: "DOWN" };

const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  START: { x: 150, y: 0 },
  A: { x: 150, y: 100 },
  B: { x: 30, y: 220 },
  C: { x: 270, y: 220 },
  D: { x: 150, y: 340 },
};

// Literal per-state class strings — see the Tailwind-JIT trap note in
// apps/aiperf-flow's SKILL.md: a runtime-interpolated `border-stroke-${role}` would be invisible
// to Tailwind's compiler, so every state's classes must appear as complete literal strings here.
const STATE_NODE_CLASSES: Record<NodeState, string> = {
  firing: "border-2 border-accent-primary bg-surface-elevated",
  ready: "border-2 border-dashed border-accent-primary bg-surface-panel",
  done: "border-stroke-secondary bg-surface-panel opacity-70",
  pending: "border-dashed border-stroke-tertiary bg-surface-chrome opacity-50",
};

function buildGraphNodes(frame: Frame): Node[] {
  return GRAPH_NODE_IDS.map((id) => {
    const isStart = id === "START";
    const s = stateOf(frame, id);
    const position = NODE_POSITIONS[id];
    return {
      id,
      type: "card",
      position,
      data: {
        title: isStart ? "START" : `${id}  (LlmNode)`,
        detail: isStart ? undefined : `${stateLabel(s)} · Dispatch`,
        className: STATE_NODE_CLASSES[s],
      },
    } satisfies Node;
  });
}

function buildGraphEdges(frame: Frame): Edge[] {
  return GRAPH_EDGES.map(({ from, to }) => {
    const fromState = stateOf(frame, from);
    const toState = stateOf(frame, to);
    const active = fromState === "done" && (toState === "firing" || toState === "ready");
    return {
      id: `${from}-${to}`,
      source: from,
      target: to,
      type: active ? "flow" : undefined,
    } satisfies Edge;
  });
}

// Runs the shared ELK layout for the frontier graph. The node/edge identity is stable across
// steps (only per-node `data` recolors), so ELK lays out once; each step's fresh node `data` is
// re-applied onto the ELK-computed positions so the recoloring stays live. Must be inside a
// `ReactFlowProvider` (uses the layout hook's React Flow hooks).
function FrontierGraphInner({ frame }: { frame: Frame }): React.JSX.Element {
  const nodes = useMemo(() => buildGraphNodes(frame), [frame]);
  const edges = useMemo(() => buildGraphEdges(frame), [frame]);
  const { nodes: laid, laidOut } = useElkLayout(nodes, edges, FRONTIER_ELK_OPTS);
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
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      nodesDraggable={false}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

function FrontierGraph({ frame }: { frame: Frame }): React.JSX.Element {
  return (
    <div style={{ height: 420 }}>
      <ReactFlowProvider>
        <FrontierGraphInner frame={frame} />
      </ReactFlowProvider>
    </div>
  );
}

function ChannelPanel({ frame, prev }: { frame: Frame; prev: Frame | null }): React.JSX.Element {
  const prevNames = new Set((prev?.channels ?? []).map((c) => c.name));
  if (frame.channels.length === 0) {
    return <p className={`text-sm ${inkClassName("tertiary")}`}>no writes yet</p>;
  }
  return (
    <Stack gap={6}>
      {frame.channels.map((c) => {
        const isNew = !prevNames.has(c.name);
        return (
          <div
            key={c.name}
            className={clsx(
              "border px-3 py-1.5",
              isNew ? "border-accent-primary" : strokeClassName("secondary"),
              isNew ? surfaceClassName("elevated") : surfaceClassName("panel"),
            )}
          >
            <Row align="center" gap={8}>
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{c.name}</span>
              <span className={`ml-auto text-xs ${inkClassName("tertiary")}`}>write_seq {c.seq}</span>
            </Row>
          </div>
        );
      })}
    </Stack>
  );
}

function GateBar({ have, need }: { have: number; need: number }): React.JSX.Element {
  return (
    <Row gap={4}>
      {Array.from({ length: need }, (_, i) => (
        <div
          key={i}
          className={clsx(
            "h-3 flex-1 border",
            i < have ? "bg-accent-primary border-accent-primary" : `border-stroke-secondary ${surfaceClassName("panel")}`,
          )}
        />
      ))}
    </Row>
  );
}

function Controls({
  index,
  total,
  isFirst,
  isLast,
  onNext,
  onBack,
  onReset,
  onJump,
}: {
  index: number;
  total: number;
  isFirst: boolean;
  isLast: boolean;
  onNext: () => void;
  onBack: () => void;
  onReset: () => void;
  onJump: (i: number) => void;
}): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Row gap={8} align="center" wrap>
        <Button variant="secondary" disabled={isFirst} onClick={onBack}>
          Prev
        </Button>
        <Button variant="primary" disabled={isLast} onClick={onNext}>
          Next
        </Button>
        <Button variant="ghost" disabled={isFirst} onClick={onReset}>
          Reset
        </Button>
        <span className={`ml-auto text-xs ${inkClassName("tertiary")}`}>
          step {index + 1} / {total}
        </span>
      </Row>
      <Row gap={6} wrap>
        {FRAMES.map((_, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onJump(i)}
            className={clsx(
              "h-6 w-6 rounded-full border text-xs font-semibold",
              i === index
                ? `border-accent-primary ${accentClassName("primary")} ${surfaceClassName("elevated")}`
                : `${strokeClassName("secondary")} ${inkClassName("tertiary")} ${surfaceClassName("panel")}`,
            )}
          >
            {i + 1}
          </button>
        ))}
      </Row>
    </Stack>
  );
}

const LEGEND_ENTRIES: Array<{ state: NodeState; label: string }> = [
  { state: "firing", label: "firing" },
  { state: "ready", label: "ready" },
  { state: "done", label: "done" },
  { state: "pending", label: "pending" },
];

function StateLegend(): React.JSX.Element {
  return (
    <Row gap={16} wrap>
      {LEGEND_ENTRIES.map((entry) => (
        <Row key={entry.state} gap={6} align="center">
          <div className={clsx("h-3 w-3", STATE_NODE_CLASSES[entry.state])} />
          <span className={`text-sm ${inkClassName("secondary")}`}>{entry.label}</span>
        </Row>
      ))}
    </Row>
  );
}

/**
 * Interactive step-through of the async dataflow frontier: a root Step (A), a concurrent
 * fan-out (B, C), and an AND-fan-in join (D). Steps through {@link FRAMES} via
 * {@link useStepSimulator}, recoloring the trace graph, the VersionedChannelStore write log, and
 * D's AND-fan-in gate at every step.
 */
export function WekaRuntimeStepperDeck(): React.JSX.Element {
  const sim = useStepSimulator(FRAMES, { autoPlayMs: 1400 });
  const frame = sim.current ?? FRAMES[0];
  const prev = sim.index > 0 ? FRAMES[sim.index - 1] : null;

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Runtime Stepper" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl 2xl:max-w-[1728px] bg-surface-page px-10 py-8">
          <Stack gap={20}>
            <Stack gap={8}>
              <Row align="center" gap={10} wrap>
                <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
                  Async dataflow frontier — interactive stepper
                </h1>
                <span
                  className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${accentClassName("primary")}`}
                >
                  weka replay
                </span>
              </Row>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Step through one weka trie trace as the TraceExecutor drives it: a root Step (A),
                a concurrent fan-out (B, C), and an AND-fan-in join (D). Use Next/Prev or click a
                step.
              </p>
            </Stack>

            <Controls
              index={sim.index}
              total={sim.total}
              isFirst={sim.isFirst}
              isLast={sim.isLast}
              onNext={sim.next}
              onBack={sim.back}
              onReset={sim.reset}
              onJump={(i) => {
                // `useStepSimulator` exposes only next/back/reset, not a direct jump, so walk
                // there with a bounded number of calls (mirrors PoolPage.tsx's "Run all" note:
                // next()/back() schedule state updates rather than mutating synchronously, so
                // this must be a fixed-size loop, never a `while` on a stale `index`/`isLast`).
                const steps = i - sim.index;
                if (steps > 0) {
                  for (let n = 0; n < steps; n++) sim.next();
                } else if (steps < 0) {
                  for (let n = 0; n < -steps; n++) sim.back();
                }
              }}
            />

            <Callout tone="info" title={`Step ${sim.index + 1}`}>
              {frame.desc}
            </Callout>

            <Grid columns="minmax(0, 340px) 1fr" gap={20} align="start">
              <div className={`border ${strokeClassName("secondary")}`}>
                <div
                  className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}
                >
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Trace graph</span>
                  <span
                    className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
                  >
                    frontier
                  </span>
                </div>
                <div className="p-3">
                  <FrontierGraph frame={frame} />
                </div>
              </div>

              <Stack gap={16}>
                <Stack gap={8}>
                  <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                    VersionedChannelStore
                  </h2>
                  <ChannelPanel frame={frame} prev={prev} />
                </Stack>
                <Stack gap={8}>
                  <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                    D — AND-fan-in gate
                  </h2>
                  <Stack gap={4}>
                    <Row align="center" gap={8}>
                      <span className={`text-xs ${inkClassName("tertiary")}`}>
                        await_inputs: B_out, C_out (count=1 each)
                      </span>
                      <span className={`ml-auto text-xs font-semibold ${inkClassName("secondary")}`}>
                        {frame.gateHave} / {GATE_NEED} arrived
                      </span>
                    </Row>
                    <GateBar have={frame.gateHave} need={GATE_NEED} />
                  </Stack>
                  <p className={`text-sm ${inkClassName("tertiary")}`}>
                    {frame.gateHave < GATE_NEED
                      ? "Gate is waiting; D stays parked on its asyncio.Event."
                      : "Gate satisfied — D is released to fire."}
                  </p>
                </Stack>
              </Stack>
            </Grid>

            <Divider />
            <StateLegend />
          </Stack>
        </div>
      </div>
    </div>
  );
}
