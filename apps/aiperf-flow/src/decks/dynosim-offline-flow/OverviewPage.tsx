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
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `OverviewPage`: the four-layer
//! native stack (Rust CLI entry point -> execution engine -> aiperf library -> engine/wire),
//! with the two trait seams (Clock, RequestSink<HttpRequest>) called out and the bottom layer
//! forking by transport.

function nodes(maint: boolean): Node[] {
  return [
    { id: "band-cli", type: "header", position: { x: 0, y: 0 }, data: { title: "RUST CLI ENTRY POINT" } },
    { id: "band-engine", type: "header", position: { x: 0, y: 140 }, data: { title: "EXECUTION ENGINE" } },
    { id: "band-lib", type: "header", position: { x: 0, y: 280 }, data: { title: "AIPERF LIBRARY" } },
    { id: "band-wire", type: "header", position: { x: 0, y: 500 }, data: { title: "ENGINE / WIRE" } },

    {
      id: "profile",
      type: "card",
      position: { x: 0, y: 40 },
      data: { title: "aiperf profile", subtitle: maint ? "load.rs / yaml.rs" : "native Config v2" },
    },
    {
      id: "benchmarkrun",
      type: "card",
      position: { x: 300, y: 40 },
      data: { title: "BenchmarkRun", subtitle: maint ? "RunnerRequest execute" : "v2 wire envelope" },
    },

    {
      id: "execute",
      type: "card",
      position: { x: 0, y: 180 },
      data: { title: "aiperf --execute", subtitle: maint ? "execute_mode.rs" : "same binary" },
    },
    {
      id: "runnerapp",
      type: "card",
      position: { x: 300, y: 180 },
      data: { title: "RunnerApplication", subtitle: maint ? "handle_v2" : "frozen registry" },
    },
    {
      id: "transportworkload",
      type: "card",
      position: { x: 600, y: 180 },
      data: { title: "transport + workload", subtitle: maint ? "AIPerfRegistry" : "independent registries" },
    },

    {
      id: "workload",
      type: "card",
      position: { x: 0, y: 320 },
      data: { title: "Workload", subtitle: maint ? "RequestRate · Graph" : "arrival pattern" },
    },
    {
      id: "scheduledruntime",
      type: "card",
      position: { x: 300, y: 320 },
      data: { title: "ScheduledRuntime", subtitle: maint ? "SlotPool · StopChecker" : "schedule + admit" },
    },
    {
      id: "observertee",
      type: "card",
      position: { x: 600, y: 320 },
      data: { title: "ObserverTee", subtitle: maint ? "Collector+Native" : "metrics" },
    },
    {
      id: "clock-seam",
      type: "panel",
      position: { x: 0, y: 420 },
      data: { title: "SEAM · Clock", surfaceRole: "chrome" },
    },
    {
      id: "sink-seam",
      type: "panel",
      position: { x: 300, y: 420 },
      data: { title: "SEAM · RequestSink<HttpRequest>", surfaceRole: "chrome" },
    },

    {
      id: "http-transport",
      type: "card",
      position: { x: 0, y: 560 },
      data: { title: "HttpTransport → real server", subtitle: "Hyper + SSE" },
    },
    {
      id: "engine-host",
      type: "card",
      position: { x: 340, y: 560 },
      data: { title: "EngineHost → SteppableReplay", subtitle: "in-process · no sockets" },
    },
  ];
}

const edges: Edge[] = [
  { id: "e-profile-run", source: "profile", target: "benchmarkrun", type: "flow" },
  { id: "e-run-execute", source: "benchmarkrun", target: "execute", type: "flow" },
  { id: "e-execute-app", source: "execute", target: "runnerapp", type: "flow" },
  { id: "e-app-transport", source: "runnerapp", target: "transportworkload", type: "flow" },
  { id: "e-app-workload", source: "runnerapp", target: "workload", type: "flow" },
  { id: "e-workload-runtime", source: "workload", target: "scheduledruntime", type: "flow" },
  { id: "e-runtime-observer", source: "scheduledruntime", target: "observertee", type: "flow" },
  { id: "e-runtime-clock", source: "scheduledruntime", target: "clock-seam", type: "flow" },
  { id: "e-runtime-sink", source: "scheduledruntime", target: "sink-seam", type: "flow" },
  { id: "e-clock-http", source: "clock-seam", target: "http-transport", type: "flow" },
  { id: "e-sink-http", source: "sink-seam", target: "http-transport", type: "flow" },
  { id: "e-sink-engine", source: "sink-seam", target: "engine-host", type: "flow" },
];

/**
 * Overview page of the Dynosim Offline explainer deck.
 *
 * Ports `OverviewPage` from `docs/canvases/dynosim-offline-flow.canvas.tsx`: the native path is
 * a four-layer stack — the Rust CLI authors Config v2 and re-execs itself in `--execute` mode,
 * the frozen `RunnerApplication` validates and dispatches, the library runs the shared benchmark
 * loop through two trait seams (`Clock`, `RequestSink<HttpRequest>`), and the bottom layer forks
 * into either a real server or Dynamo's in-process engine.
 */
export function OverviewPage({ level }: { level: Level }): React.JSX.Element {
  const maint = atLeast(level, "maintainer");
  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>How it fits together</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The native path is a four-layer stack: the Rust CLI authors Config v2 and re-execs
          itself in <strong>--execute</strong> mode, the frozen <strong>RunnerApplication</strong>{" "}
          validates and dispatches, the library runs the shared benchmark loop, and the bottom
          layer is either a real server or Dynamo&apos;s in-process engine. Flow runs top-down;
          the report exits the side.
        </p>
      </div>

      <div style={{ height: 700 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes(maint)}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Shared above the seam">
          Layers 1–3 are identical for every mode; only the bottom transport fork changes.
        </Callout>
        <Callout tone="info" title="No sockets (dynosim)">
          The right fork feeds token arrays straight to the engine.
        </Callout>
        <Callout tone="success" title="Verified">
          AIPerf&apos;s summary must byte-match Dynamo&apos;s, or the run fails.
        </Callout>
      </Grid>
    </Stack>
  );
}
