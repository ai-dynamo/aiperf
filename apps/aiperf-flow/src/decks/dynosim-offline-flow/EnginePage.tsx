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
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast, SegControl } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `EnginePage`: the offline engine
//! (Dynamo's `SteppableReplay`) is built in one of three topology shapes, each optionally
//! stepping through a KV-affinity router instead of round-robin.

type TopoId = "single" | "aggregated" | "disaggregated";
type RouterId = "round_robin" | "kv";

const TOPO_OPTIONS = [
  { id: "single" as const, label: "single" },
  { id: "aggregated" as const, label: "aggregated" },
  { id: "disaggregated" as const, label: "disaggregated" },
];
const ROUTER_OPTIONS = [
  { id: "round_robin" as const, label: "round robin" },
  { id: "kv" as const, label: "kv" },
];

const ENGINE_NAME: Record<TopoId, string> = {
  single: "SteppableEngine",
  aggregated: "SteppableAgg",
  disaggregated: "SteppableDisagg",
};
const ENGINE_SUB: Record<TopoId, string> = {
  single: "1 worker · no router",
  aggregated: "N workers",
  disaggregated: "prefill + decode pools",
};

function nodes(topo: TopoId, router: RouterId, maint: boolean): Node[] {
  const kv = router === "kv";
  const header: Node = {
    id: "engine-header",
    type: "header",
    position: { x: 260, y: 0 },
    data: { title: ENGINE_NAME[topo], caption: ENGINE_SUB[topo] + (maint ? " · build_native()" : "") },
  };

  if (topo === "single") {
    return [
      header,
      { id: "w0", type: "card", position: { x: 300, y: 100 }, data: { title: "w0" } },
    ];
  }

  if (topo === "aggregated") {
    const workers: Node[] = [0, 1, 2].map((i) => ({
      id: `w${i}`,
      type: "card",
      position: { x: 150 + i * 220, y: 200 },
      data: { title: `w${i}` },
    }));
    const routerNode: Node[] = kv
      ? [
          {
            id: "router",
            type: "panel",
            position: { x: 260, y: 100 },
            data: { title: maint ? "OfflineReplayRouter" : "KV router" },
          },
        ]
      : [];
    return [header, ...routerNode, ...workers];
  }

  // disaggregated
  const prefillWorkers: Node[] = [0, 1].map((i) => ({
    id: `p${i}`,
    type: "card",
    position: { x: i * 220, y: 220 },
    data: { title: `p${i}` },
  }));
  const decodeWorkers: Node[] = [0, 1].map((i) => ({
    id: `d${i}`,
    type: "card",
    position: { x: 520 + i * 220, y: 220 },
    data: { title: `d${i}` },
  }));
  const routers: Node[] = kv
    ? [
        { id: "prefill-router", type: "panel", position: { x: 0, y: 100 }, data: { title: "prefill_router" } },
        { id: "decode-router", type: "panel", position: { x: 520, y: 100 }, data: { title: "decode_router" } },
      ]
    : [];
  const poolLabels: Node[] = [
    { id: "prefill-label", type: "chip", position: { x: 0, y: 40 }, data: { label: maint ? "prefill · Hidden" : "prefill" } },
    { id: "decode-label", type: "chip", position: { x: 520, y: 40 }, data: { label: maint ? "decode · Visible" : "decode" } },
  ];
  return [header, ...poolLabels, ...routers, ...prefillWorkers, ...decodeWorkers];
}

function edges(topo: TopoId, router: RouterId): Edge[] {
  const kv = router === "kv";
  if (topo === "single") {
    return [{ id: "e-header-w0", source: "engine-header", target: "w0", type: "flow" }];
  }
  if (topo === "aggregated") {
    if (kv) {
      return [
        { id: "e-header-router", source: "engine-header", target: "router", type: "flow" },
        ...[0, 1, 2].map((i) => ({ id: `e-router-w${i}`, source: "router", target: `w${i}`, type: "flow" as const })),
      ];
    }
    return [0, 1, 2].map((i) => ({ id: `e-header-w${i}`, source: "engine-header", target: `w${i}`, type: "flow" as const }));
  }
  // disaggregated
  const es: Edge[] = [];
  if (kv) {
    es.push({ id: "e-header-prefill-router", source: "engine-header", target: "prefill-router", type: "flow" });
    es.push({ id: "e-header-decode-router", source: "engine-header", target: "decode-router", type: "flow" });
    es.push({ id: "e-prefill-router-p0", source: "prefill-router", target: "p0", type: "flow" });
    es.push({ id: "e-prefill-router-p1", source: "prefill-router", target: "p1", type: "flow" });
    es.push({ id: "e-decode-router-d0", source: "decode-router", target: "d0", type: "flow" });
    es.push({ id: "e-decode-router-d1", source: "decode-router", target: "d1", type: "flow" });
  } else {
    es.push({ id: "e-header-p0", source: "engine-header", target: "p0", type: "flow" });
    es.push({ id: "e-header-p1", source: "engine-header", target: "p1", type: "flow" });
    es.push({ id: "e-header-d0", source: "engine-header", target: "d0", type: "flow" });
    es.push({ id: "e-header-d1", source: "engine-header", target: "d1", type: "flow" });
  }
  es.push({ id: "e-p0-d0-handoff", source: "p0", target: "d0", label: "handoff" });
  return es;
}

const TOPO_CALLOUT: Record<TopoId, { dev: string; maint: string }> = {
  single: {
    dev: "one ReplayWorkerCore · router_mode ignored",
    maint: "falls back to SteppableAgg(1, RoundRobin) when clock events force step_until",
  },
  aggregated: {
    dev: "N workers, one aggregated component",
    maint: "workers × OfflineWorkerState in one EngineComponent(Aggregated)",
  },
  disaggregated: {
    dev: "two independent pools with handoff",
    maint: "separate MockEngineArgs (WorkerType::Prefill / Decode) + per-pool routers",
  },
};

/**
 * Engine internals — topology builder page. Switching `topo` (single / aggregated /
 * disaggregated) and `router` (round_robin / kv) rewires the worker/router diagram to match,
 * mirroring the source canvas's `EnginePage`.
 */
export function EnginePage({ level }: { level: Level }): React.JSX.Element {
  const [topo, setTopo] = useState<TopoId>("aggregated");
  const [router, setRouter] = useState<RouterId>("kv");
  const dev = atLeast(level, "developer");
  const maint = atLeast(level, "maintainer");
  const kv = router === "kv";

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Engine internals — topology builder</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The offline engine (Dynamo&apos;s <strong>SteppableReplay</strong>) is built in one of
          three shapes. Change the shape and the router and watch workers and routing rewire.
        </p>
      </div>

      <Row gap={20} align="center" wrap>
        <Stack gap={4}>
          <span className={`text-xs ${inkClassName("tertiary")}`}>topology</span>
          <SegControl value={topo} onChange={setTopo} options={TOPO_OPTIONS} />
        </Stack>
        <Stack gap={4}>
          <span className={`text-xs ${inkClassName("tertiary")}`}>router_mode</span>
          <SegControl value={router} onChange={setRouter} options={ROUTER_OPTIONS} />
        </Stack>
      </Row>

      <div style={{ height: 360 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes(topo, router, maint)}
          edges={edges(topo, router)}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      {dev && (
        <Grid columns={2} gap={12}>
          <Callout tone="info" title={kv ? "KV routing" : "Round-robin"}>
            {kv
              ? "Each pool gets an OfflineReplayRouter that places requests by KV-cache affinity."
              : "router is None — a plain round-robin index picks the next worker."}
          </Callout>
          <Callout tone="info" title="Stepping seam">
            {maint
              ? "EngineHost::step calls step_until(next_event) so the engine can't overshoot the virtual clock."
              : "Whatever the shape, EngineHost steps it one bounded slice at a time."}
          </Callout>
        </Grid>
      )}
      {dev && (
        <p className={`text-xs ${inkClassName("tertiary")}`}>
          {maint ? TOPO_CALLOUT[topo].maint : TOPO_CALLOUT[topo].dev}
        </p>
      )}
    </Stack>
  );
}
