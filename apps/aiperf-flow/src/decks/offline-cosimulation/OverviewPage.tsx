/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

//! Ported from `offline-cosimulation.canvas.tsx` `ArchitectureDiagram` + overview page: a
//! mode-driven (dynosim_offline | dynosim_online) picture of AIPerf owning orchestration, clock,
//! and measurement while a passive Dynamo engine steps in-process with no sockets. The hand-drawn
//! SVG becomes `header`/`panel`/`card` nodes and `flow` edges.

/** Execution mode selecting the clock and the `transport.type` authored in Config v2. */
export type Mode = "offline" | "online";

/** Builds the architecture graph for the selected mode. Only the request-color path and clock box
 *  change between `dynosim_offline` (SimClock / virtual time) and `dynosim_online` (RealClock). */
function nodes(mode: Mode): Node[] {
  return [
    {
      id: "band-owns",
      type: "header",
      position: { x: 0, y: 0 },
      data: { title: "AIPERF OWNS ORCHESTRATION, CLOCK, AND MEASUREMENT" },
    },

    // Row 1 — request-color path.
    {
      id: "config-v2",
      type: "panel",
      position: { x: 0, y: 60 },
      data: { title: "Config v2", detail: `transport.type: dynosim_${mode}` },
    },
    {
      id: "run-loop",
      type: "card",
      position: { x: 350, y: 60 },
      data: { title: "AIPerf run loop", detail: "schedule · admit · step" },
    },
    {
      id: "clock",
      type: "card",
      position: { x: 700, y: 60 },
      data: {
        title: mode === "offline" ? "SimClock" : "RealClock",
        detail: mode === "offline" ? "integer-ns virtual time" : "wall-clock replay",
      },
    },

    // Row 2 — engine boundary.
    {
      id: "workload",
      type: "panel",
      position: { x: 0, y: 200 },
      data: { title: "Workload", detail: "requests · graph gates" },
    },
    {
      id: "engine-boundary",
      type: "card",
      position: { x: 350, y: 200 },
      data: { title: "Steppable engine boundary", detail: "step_until · next_event_ms" },
    },
    {
      id: "request-observer",
      type: "card",
      position: { x: 700, y: 200 },
      data: { title: "RequestObserver", detail: "shared Level-B contract" },
    },

    // Row 3 — in-process engine + observers.
    {
      id: "no-sockets",
      type: "panel",
      position: { x: 0, y: 340 },
      data: { title: "No sockets", detail: "in-process Dynamo mocker" },
    },
    {
      id: "collector",
      type: "panel",
      position: { x: 350, y: 340 },
      data: { title: "AIPerf collector", detail: "primary observer" },
    },
    {
      id: "trace-tee",
      type: "panel",
      position: { x: 700, y: 340 },
      data: { title: "Optional trace tee", detail: "mocker TraceCollector" },
    },

    // Second band — consumers.
    {
      id: "band-consumers",
      type: "header",
      position: { x: 0, y: 460 },
      data: { title: "THE SAME OBSERVER STREAM POWERS EVERY CONSUMER" },
    },
    {
      id: "report",
      type: "card",
      position: { x: 0, y: 520 },
      data: { title: "Native v2 report", detail: "same report schema" },
    },
    {
      id: "streaming",
      type: "card",
      position: { x: 260, y: 520 },
      data: { title: "Streaming metrics", detail: "TTFT · tokens · terminal" },
    },
    {
      id: "adaptive",
      type: "card",
      position: { x: 520, y: 520 },
      data: { title: "Adaptive windows", detail: "live scale decisions" },
    },
    {
      id: "dashboard",
      type: "card",
      position: { x: 780, y: 520 },
      data: { title: "Live dashboard", detail: "not post-hoc" },
    },
  ];
}

const edges: Edge[] = [
  { id: "e-config-run", source: "config-v2", target: "run-loop", type: "flow", label: "select mode" },
  // Clock owns `now`; it feeds the run loop rather than the reverse.
  { id: "e-clock-run", source: "clock", target: "run-loop", type: "flow", label: "owns now" },
  { id: "e-workload-engine", source: "workload", target: "engine-boundary", type: "flow", label: "token arrays" },
  { id: "e-run-engine", source: "run-loop", target: "engine-boundary", type: "flow", label: "step" },
  { id: "e-clock-engine", source: "clock", target: "engine-boundary", type: "flow", label: "next event" },
  {
    id: "e-engine-observer",
    source: "engine-boundary",
    target: "request-observer",
    type: "flow",
    label: "events during run",
  },
  { id: "e-engine-nosockets", source: "engine-boundary", target: "no-sockets", type: "flow" },
  { id: "e-observer-collector", source: "request-observer", target: "collector", type: "flow" },
  { id: "e-observer-trace", source: "request-observer", target: "trace-tee", type: "flow" },
  // The one observer stream fans out to every consumer.
  { id: "e-collector-report", source: "collector", target: "report", type: "flow" },
  { id: "e-collector-streaming", source: "collector", target: "streaming", type: "flow" },
  { id: "e-collector-adaptive", source: "collector", target: "adaptive", type: "flow" },
  { id: "e-collector-dashboard", source: "collector", target: "dashboard", type: "flow" },
];

/** Two-option exclusive mode switch rendered as a small segmented control. */
function ModeControl({ mode, onChange }: { mode: Mode; onChange: (mode: Mode) => void }): React.JSX.Element {
  const options: ReadonlyArray<{ id: Mode; label: string }> = [
    { id: "offline", label: "dynosim_offline" },
    { id: "online", label: "dynosim_online" },
  ];
  return (
    <Row gap={6} align="center">
      <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>Mode</span>
      {options.map((option) => {
        const active = option.id === mode;
        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.id)}
            className={
              active
                ? "rounded-md border border-accent-primary bg-accent-primary px-3 py-1 text-xs font-medium text-white shadow-md"
                : `rounded-md border px-3 py-1 text-xs font-medium shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`
            }
          >
            {option.label}
          </button>
        );
      })}
    </Row>
  );
}

/**
 * Overview page of the Offline Co-simulation deck.
 *
 * Ports the source canvas's `ArchitectureDiagram`: AIPerf owns the run loop, clock, and
 * measurement while the passive Dynamo engine steps in-process (no sockets). Toggling the mode
 * swaps the authored `transport.type` and the clock (`SimClock` virtual time vs `RealClock`
 * wall-clock replay) while the steppable boundary and observer path stay stable.
 */
export function OverviewPage(): React.JSX.Element {
  const [mode, setMode] = useState<Mode>("offline");

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
          One driver, a passive engine, one observer stream
        </h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          AIPerf owns orchestration, the clock, and measurement. The Dynamo engine is passive: it
          receives a scalar <strong>now</strong> and steps in-process with no sockets. Switch modes
          to see the clock change while the steppable boundary and observer path remain stable.
        </p>
      </div>

      <ModeControl mode={mode} onChange={setMode} />

      <div style={{ height: 680 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes(mode)}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      <div className={`rounded-lg border px-4 py-3 text-sm shadow-sm ${strokeClassName("secondary")}`}>
        <span className={inkClassName("secondary")}>
          first on_token = TTFT + prefill release + graph first-token gate
        </span>
      </div>

      <Grid columns={2} gap={12}>
        <Callout tone="info" title="One driver contract, two clocks">
          The passive engine receives a scalar <strong>now</strong> and a mutable observer. It does
          not own AIPerf&apos;s clock or run loop. Switch modes above to see the clock change while
          the steppable engine and observer path remain stable.
        </Callout>
        <Callout tone="success" title="Acceptance invariant">
          Steppable execution must reproduce the batch path&apos;s <strong>perf_ns</strong> sequence
          bit-for-bit on handoff fixtures, while AIPerf emits its normal native-v2 report.
        </Callout>
      </Grid>
    </Stack>
  );
}
