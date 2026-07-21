/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 04 — Clock and request dispatch as concrete substitution points. Workload
//! orchestration feeds two trait seams (Clock, RequestSink) that resolve to one concrete
//! sink per transport, then a five-callback observer row. Ported from `SeamTheater`.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  panelNode,
  cardNode,
  chipNode,
  flowEdge,
  plainEdge,
  rank,
  type Detail,
} from "./parts.js";

type Transport = "http" | "grpc" | "dynosim";

const SELECTED: Record<
  Transport,
  { clock: string; clockSub: string; sink: string; target: string; dispatchSeam: string; dispatchSub: string }
> = {
  http: {
    clock: "Clock → RealClock",
    clockSub: "hot-path scheduling + native metrics",
    sink: "TransportSink",
    target: "Hyper + SSE",
    dispatchSeam: "RequestSink<Request>",
    dispatchSub: "dispatch request to terminal",
  },
  grpc: {
    clock: "Clock → RealClock",
    clockSub: "hot-path scheduling + native metrics",
    sink: "GrpcTransportSink",
    target: "Tonic stream",
    dispatchSeam: "RequestSink<GrpcRequest>",
    dispatchSub: "dispatch request to terminal",
  },
  dynosim: {
    clock: "Clock by transport ID",
    clockSub: "offline SimClock · online RealClock",
    sink: "DynosimExecutor",
    target: "crate::dynosim backend",
    dispatchSeam: "PreparedRunnerOperation",
    dispatchSub: "scheduled | graph dynosim operation",
  },
};

const OBSERVER_STAGES = ["arrival", "admit", "token", "usage", "terminal"] as const;

function buildNodes(transport: Transport, detail: Detail): Node[] {
  const sel = SELECTED[transport];
  const nodes: Node[] = [
    panelNode("workload", 300, 0, "Workload orchestration", "phase · pace · graph · admission", "primary"),
    cardNode("seam-clock", 60, 130, "TRAIT SEAM", sel.clock, sel.clockSub, "primary"),
    cardNode("seam-dispatch", 560, 130, "TRAIT SEAM", sel.dispatchSeam, sel.dispatchSub, "primary"),
    cardNode(
      "sink",
      300,
      270,
      sel.sink,
      sel.target,
      rank(detail) > 0 ? "worker-local concrete implementation" : undefined,
      "primary",
    ),
  ];
  OBSERVER_STAGES.forEach((label, index) => {
    const displayed = rank(detail) > 0 ? (label === "token" ? "token callbacks" : `on_${label}`) : label;
    nodes.push(chipNode(`obs-${label}`, 40 + index * 150, 400, displayed));
  });
  return nodes;
}

function buildEdges(): Edge[] {
  const edges: Edge[] = [
    plainEdge("e-workload-clock", "workload", "seam-clock"),
    plainEdge("e-workload-dispatch", "workload", "seam-dispatch"),
    flowEdge("e-clock-sink", "seam-clock", "sink"),
    flowEdge("e-dispatch-sink", "seam-dispatch", "sink", { speed: "fast" }),
  ];
  for (let i = 0; i < OBSERVER_STAGES.length - 1; i += 1) {
    edges.push(flowEdge(`e-obs-${i}`, `obs-${OBSERVER_STAGES[i]}`, `obs-${OBSERVER_STAGES[i + 1]}`));
  }
  edges.push(flowEdge("e-sink-obs", "sink", "obs-token"));
  return edges;
}

/** Section 04 diagram: transport-selected clock and dispatch seams into one observer vocabulary. */
export function SeamTheaterSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [transport, setTransport] = useState<Transport>("http");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="04"
          title="Clock and request dispatch are concrete substitution points"
          subtitle="Transport preparation resolves a clock implementation and an execution path for HTTP, gRPC, or DynoSim."
        />
        <Segmented
          ariaLabel="Transport"
          value={transport}
          onChange={setTransport}
          options={[
            { id: "http", label: "HTTP" },
            { id: "grpc", label: "gRPC" },
            { id: "dynosim", label: "DynoSim" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(transport, detail)} edges={buildEdges()} height={480} />
      <p className={`text-center text-xs ${inkClassName("tertiary")}`}>
        one observer vocabulary · transport-neutral metrics
      </p>

      <Grid columns={2} gap={14}>
        <Callout tone="info" title="TTFT derivation">
          Time to first token is derived from the first token callback.
        </Callout>
        <Callout tone="info" title="Worker-local observer graph">
          <Code inline>ObserverTee</Code> stores <Code inline>Rc&lt;dyn RequestObserver&gt;</Code>, and{" "}
          <Code inline>NativeMetricsObserver</Code> stores request state in <Code inline>RefCell</Code>.
        </Callout>
      </Grid>
      <SourcesRow
        detail={detail}
        paths={[
          { label: "Clock trait", path: "rust/runtime/src/clock/clock.rs" },
          { label: "RequestSink", path: "rust/loadgen-core/src/sink.rs" },
          { label: "metrics adapter", path: "rust/runtime/src/metrics.rs" },
        ]}
      />
    </SectionShell>
  );
}
