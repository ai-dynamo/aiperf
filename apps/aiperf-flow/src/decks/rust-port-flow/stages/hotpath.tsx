/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 7 — the request hot-path. This is the spine the play-layer `RequestParticle` animates:
//! `RequestRateWorkload` (a `Workload` draining through `Rc<ScheduledRuntime>`) issues a turn →
//! `SlotPool` + `StopChecker` admission gate → `Rc<dyn Dispatcher>::dispatch_collect` → the chosen
//! sink (`TransportSink` / `GrpcTransportSink`) → shared `reduce_parsed_response` fold → shared
//! `measure_dispatch` record. TTFT is the first-token observation (`reduce.rs`'s once-only latch
//! fires `on_first_token(at_ns - start_ns)`).
//!
//! All source anchors below were verified against the real `rust/runtime` tree (line numbers
//! confirmed by grep, not copied from the spec). The module fills in the already-exported
//! `hotPathStage` `StageDef` (level-1 `subgraph`, two level-2 `leaves`, `evidence`) and also
//! exports the additive `hotPathSteps` play fragment + a `HotPathCards` explainer grid the deck
//! composition can render alongside the canvas.

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { NodeRole } from "../stage.js";
import type { FlowStep } from "../../../interactive/index.js";
import { Grid } from "../../../layout/Grid.js";
import { Callout } from "../../../prose/Callout.js";
import type { StageDef } from "../stage.js";

/** A card node colored by semantic role, laid out at a fixed spine position. */
function card(
  id: string,
  title: string,
  subtitle: string,
  detail: string,
  role: NodeRole,
  x: number,
  y = 0,
): Node {
  return {
    id,
    type: "card",
    position: { x, y },
    data: { title, subtitle, detail, className: roleClassName(role) },
  };
}

function flowEdge(source: string, target: string, label?: string): Edge {
  return {
    id: `e-${source}-${target}`,
    source,
    target,
    type: "flow",
    ...(label !== undefined ? { label } : {}),
  };
}

const COL = 250;

// ── Level-1 subgraph: the six-step hot-path spine ─────────────────────────────
// Node ids `hotpath.admission` and `hotpath.dispatch` double as `leaves` keys so a click on either
// drills one level deeper (ZoomStage.drill matches the clicked node id against `subgraph.children`).

const HP_WORKLOAD = "hp-workload";
const HP_ADMISSION = "hotpath.admission";
const HP_DISPATCH = "hotpath.dispatch";
const HP_SINK = "hp-sink";
const HP_REDUCE = "hp-reduce";
const HP_MEASURE = "hp-measure";

const subgraphNodes: Node[] = [
  card(
    HP_WORKLOAD,
    "RequestRateWorkload",
    "impl Workload",
    "execute() drains scheduled work via Rc<ScheduledRuntime>.",
    "control",
    0,
  ),
  card(
    HP_ADMISSION,
    "Admission gate",
    "SlotPool · StopChecker",
    "Concurrency credit + run bounds — click to open.",
    "control",
    COL,
  ),
  card(
    HP_DISPATCH,
    "Rc<dyn Dispatcher>",
    "dispatch_collect",
    "One PreparedTurn to the chosen sink — click to open.",
    "transport",
    COL * 2,
  ),
  card(
    HP_SINK,
    "Chosen sink",
    "TransportSink / GrpcTransportSink",
    "The only transport-specific step; rest is shared.",
    "transport",
    COL * 3,
  ),
  card(
    HP_REDUCE,
    "reduce_parsed_response",
    "transport::reduce",
    "Folds each ParsedResponse; latches TTFT on first token.",
    "compute",
    COL * 4,
  ),
  card(
    HP_MEASURE,
    "measure_dispatch",
    "transport::measure",
    "Records one terminal into NativeMetricsObserver.",
    "compute",
    COL * 5,
  ),
];

const subgraphEdges: Edge[] = [
  flowEdge(HP_WORKLOAD, HP_ADMISSION),
  flowEdge(HP_ADMISSION, HP_DISPATCH, "admit"),
  flowEdge(HP_DISPATCH, HP_SINK),
  flowEdge(HP_SINK, HP_REDUCE, "TTFT: first token"),
  flowEdge(HP_REDUCE, HP_MEASURE),
];

// ── Level-2 leaf: the admission gate internals ────────────────────────────────

const admissionLeafNodes: Node[] = [
  card(
    "hp-slotpool",
    "SlotPool",
    "runtime::timing::slots",
    "Hands out concurrency credits (admit_ns); waits for a slot.",
    "control",
    0,
  ),
  card(
    "hp-stopchecker",
    "StopChecker",
    "runtime::timing::stop",
    "Enforces request-count / duration stop bounds.",
    "control",
    COL,
  ),
];

const admissionLeafEdges: Edge[] = [flowEdge("hp-slotpool", "hp-stopchecker", "then")];

// ── Level-2 leaf: dispatch → sink → shared reduce/measure ─────────────────────

const dispatchLeafNodes: Node[] = [
  card(
    "hp-dispatch-collect",
    "dispatch_collect",
    "trait Dispatcher",
    "Runs one owned PreparedTurn, retaining terminal facts.",
    "transport",
    0,
  ),
  card(
    "hp-on-first-token",
    "on_first_token(TTFT)",
    "&dyn Fn(i64)",
    "Invoked once with TTFT in ns — the first-token observation.",
    "compute",
    COL,
  ),
  card(
    "hp-first-token-latch",
    "first_token_released",
    "Cell<bool> latch",
    "Fires on_first_token(at_ns - start_ns) on first content.",
    "compute",
    COL * 2,
  ),
];

const dispatchLeafEdges: Edge[] = [
  flowEdge("hp-dispatch-collect", "hp-on-first-token"),
  flowEdge("hp-on-first-token", "hp-first-token-latch"),
];

/**
 * Stage 7 — the request hot-path. Fleshes out the foundation's exported stub with the real level-1
 * spine subgraph, two drill-in leaves (admission gate; dispatch → TTFT latch), and verified source
 * anchors.
 */
export const hotPathStage: StageDef = {
  id: "hotpath",
  order: 7,
  label: "Request hot-path",
  caption:
    "ScheduledRuntime/Workload (RequestRateWorkload etc.) → SlotPool + StopChecker admission → Rc<dyn Dispatcher> → the chosen sink → shared reduce_parsed_response → shared measure. TTFT = first token observation.",
  tone: "red",
  // v2 timeline: the request hits the Server lane — send, first token (TTFT, a wide wall-ms gap), reduce.
  lane: "server",
  events: [
    { id: "hp-send", label: "send", laneId: "server", atOrder: 9, realOffsetMs: 64 },
    { id: "hp-ttft", label: "TTFT", laneId: "server", atOrder: 10, realOffsetMs: 121 },
    { id: "hp-reduce", label: "reduce", laneId: "server", atOrder: 11, realOffsetMs: 205 },
  ],
  subgraph: {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    children: [HP_ADMISSION, HP_DISPATCH],
  },
  leaves: {
    [HP_ADMISSION]: {
      label: "Admission gate",
      nodes: admissionLeafNodes,
      edges: admissionLeafEdges,
    },
    [HP_DISPATCH]: {
      label: "Dispatch → TTFT",
      nodes: dispatchLeafNodes,
      edges: dispatchLeafEdges,
    },
  },
  evidence: [
    { label: "struct RequestRateWorkload", path: "runtime/src/request_rate.rs:140" },
    { label: "trait Workload::execute", path: "runtime/src/scheduled.rs:1115" },
    { label: "struct SlotPool", path: "runtime/src/timing/slots.rs:105" },
    { label: "struct StopChecker", path: "runtime/src/timing/stop.rs:164" },
    { label: "trait Dispatcher::dispatch_collect", path: "runtime/src/transport/core/dispatch.rs:332" },
    { label: "fn reduce_parsed_response", path: "runtime/src/transport/reduce.rs:55" },
    { label: "async fn measure_dispatch", path: "runtime/src/transport/measure.rs:92" },
    { label: "TTFT first-token latch", path: "runtime/src/transport/reduce.rs:72" },
  ],
};

/**
 * The play-layer fragment for this stage: one `FlowStep` per spine node, in traversal order, so the
 * shared `RequestParticle`/`useFlowPlayer` can drive a request through the hot-path with real,
 * type-named captions. Additive export — the overview shell plays one step per stage; a per-stage
 * play head consumes this fragment.
 */
export const hotPathSteps: readonly FlowStep[] = [
  {
    nodeId: HP_WORKLOAD,
    caption:
      "RequestRateWorkload::execute issues the next scheduled turn through Rc<ScheduledRuntime>.",
    variant: "issue",
  },
  {
    nodeId: HP_ADMISSION,
    caption:
      "SlotPool grants a concurrency credit and StopChecker confirms the run bounds still allow issuance.",
    variant: "admit",
  },
  {
    nodeId: HP_DISPATCH,
    caption:
      "Rc<dyn Dispatcher>::dispatch_collect runs the PreparedTurn and will call on_first_token once with TTFT.",
    variant: "dispatch",
  },
  {
    nodeId: HP_SINK,
    caption:
      "The chosen sink — TransportSink (hyper, streaming) or GrpcTransportSink (Tonic) — drives the request upstream.",
    variant: "sink",
  },
  {
    nodeId: HP_REDUCE,
    caption:
      "reduce_parsed_response folds each ParsedResponse into the accumulators; the first token latches TTFT.",
    variant: "reduce",
  },
  {
    nodeId: HP_MEASURE,
    caption:
      "measure_dispatch records one terminal DispatchResult into the worker-local NativeMetricsObserver.",
    variant: "measure",
  },
];

interface HotPathCard {
  tone: Parameters<typeof Callout>[0]["tone"];
  title: string;
  body: React.ReactNode;
}

const HOT_PATH_CARDS: readonly HotPathCard[] = [
  {
    tone: "danger",
    title: "Workload issues the schedule",
    body: (
      <>
        <code>RequestRateWorkload</code> implements <code>Workload</code>; its{" "}
        <code>execute()</code> drains all scheduled work through an <code>Rc&lt;ScheduledRuntime&gt;</code>,
        issuing one turn at a time.
      </>
    ),
  },
  {
    tone: "warning",
    title: "Admission gate",
    body: (
      <>
        <code>SlotPool</code> hands out a concurrency credit (the <code>admit_ns</code> stamp) and{" "}
        <code>StopChecker</code> enforces the request-count / duration bounds — both must pass before a
        turn dispatches.
      </>
    ),
  },
  {
    tone: "info",
    title: "Dispatcher indirection",
    body: (
      <>
        An <code>Rc&lt;dyn Dispatcher&gt;</code> hides which transport is wired in;{" "}
        <code>dispatch_collect</code> runs one <code>PreparedTurn</code> and invokes{" "}
        <code>on_first_token</code> exactly once with TTFT in nanoseconds.
      </>
    ),
  },
  {
    tone: "info",
    title: "The chosen sink is the only seam",
    body: (
      <>
        <code>TransportSink</code> (hyper, streaming) or <code>GrpcTransportSink</code> (Tonic,
        non-streaming) is the sole transport-specific step; everything downstream is shared.
      </>
    ),
  },
  {
    tone: "success",
    title: "Shared reduce + measure",
    body: (
      <>
        <code>reduce_parsed_response</code> folds each <code>ParsedResponse</code> into the
        accumulators, and <code>measure_dispatch</code> wraps the whole dispatch to record one
        terminal into the worker-local <code>NativeMetricsObserver</code>.
      </>
    ),
  },
  {
    tone: "neutral",
    title: "TTFT = first token observation",
    body: (
      <>
        The once-only <code>first_token_released</code> latch fires{" "}
        <code>on_first_token(at_ns - start_ns)</code> the first time content arrives — that delta is
        the time-to-first-token.
      </>
    ),
  },
];

/**
 * The explainer grid for the hot-path stage: a `Grid` of `Callout` cards naming the real types on
 * the spine. Additive export the deck composition renders beside the level-1 canvas.
 */
export function HotPathCards(): React.JSX.Element {
  return (
    <Grid columns={2} gap={14}>
      {HOT_PATH_CARDS.map((c) => (
        <Callout key={c.title} tone={c.tone} title={c.title}>
          {c.body}
        </Callout>
      ))}
    </Grid>
  );
}
