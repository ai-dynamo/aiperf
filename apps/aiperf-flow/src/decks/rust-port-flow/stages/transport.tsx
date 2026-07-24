/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 6 — The Transport seam. A transport is defined by exactly TWO traits — `WorkerSink`
//! (the worker-facing dispatch contract) and `ExecutionSinkBuilder` (constructs the `!Send`
//! worker-local sink on each reactor); everything upstream (workload, admission, `Rc<dyn
//! Dispatcher>`, the shared `reduce`/`measure` path) is target-agnostic. Level-1 shows the
//! Dispatcher → two-trait seam → four targets; each target drills (level-2) into its concrete
//! builder+sink impls. All anchors verified against `rust/runtime`:
//!   - `trait WorkerSink`               runtime/src/engine/turn_execution.rs:74
//!   - `trait ExecutionSinkBuilder`     runtime/src/engine/turn_execution.rs:136
//!   - `trait Dispatcher`               runtime/src/transport/core/dispatch.rs:332
//!   - HTTP  `struct TransportSink`     runtime/src/transport/http/sink.rs:164
//!          `HttpSinkBuilder`/impls     runtime/src/engine/turn_execution.rs:105, :166
//!   - gRPC  `struct GrpcTransportSink` runtime/src/transport/grpc/sink.rs:102
//!          `GrpcSinkBuilder`/impls     runtime/src/engine/grpc_turn_execution.rs:81, :101, :122
//!   - dry-run `DryRunTransportFactoryV2`/`DryRunNativeExecution`
//!                                       runtime/src/engine/dry_run.rs:315, :372
//!   - dynosim `SteppableEngine::new`   runtime/src/dynosim.rs:594
//!            `trait OfflineEngineFactory`/`NativeDynamoEngineFactory`
//!                                       runtime/src/dynosim.rs:122, :129

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { NodeRole } from "../stage.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";
import { Diagram, NodeChip, DbNode, DiamondNode, MiniArrow, BiArrow, MiniBars } from "../../../chalk/index.js";

// Leaf (level-2) ids. Each is BOTH a level-1 target-node id (so a click drills) AND a key of
// `leaves` (so `buildZoomTree` registers its subgraph). Namespaced to avoid any cross-stage
// collision in the shared `ZoomTree`.
const HTTP = "transport-http";
const GRPC = "transport-grpc";
const DRY_RUN = "transport-dry-run";
const DYNOSIM = "transport-dynosim";

/** Level-1: shared upstream `Dispatcher` → the two-trait seam boundary → the four sink targets. */
const seamNodes: Node[] = [
  {
    id: "transport__dispatcher",
    type: "card",
    position: { x: 0, y: 150 },
    data: {
      title: "Rc<dyn Dispatcher>",
      subtitle: "shared upstream",
      detail: "Shared workload + SlotPool admission; just holds a sink.",
      diagram: (
        <Diagram>
          <NodeChip>workload</NodeChip>
          <MiniArrow />
          <DiamondNode>admit</DiamondNode>
          <MiniArrow />
          <NodeChip accent>Sink</NodeChip>
        </Diagram>
      ),
      className: roleClassName("transport"),
    },
  },
  {
    id: "transport__seam",
    type: "card",
    position: { x: 260, y: 150 },
    data: {
      title: "Two-trait seam",
      subtitle: "WorkerSink + ExecutionSinkBuilder",
      detail: "Implement these two traits; everything else is shared.",
      diagram: (
        <Diagram>
          <NodeChip>Builder</NodeChip>
          <MiniArrow />
          <NodeChip accent>!Send sink</NodeChip>
          <BiArrow />
          <NodeChip>turn</NodeChip>
        </Diagram>
      ),
      className: roleClassName("transport"),
    },
  },
  {
    id: HTTP,
    type: "card",
    position: { x: 540, y: 0 },
    data: {
      title: "TransportSink",
      subtitle: "HTTP · hyper · streaming",
      detail: "supports_response_streaming() = true — live SSE.",
      diagram: (
        <Diagram>
          <NodeChip accent>hyper</NodeChip>
          <MiniArrow />
          <DiamondNode>1st?</DiamondNode>
          <MiniArrow />
          <NodeChip>t₁·t₂·t₃</NodeChip>
        </Diagram>
      ),
      className: roleClassName("transport"),
    },
  },
  {
    id: GRPC,
    type: "card",
    position: { x: 540, y: 110 },
    data: {
      title: "GrpcTransportSink",
      subtitle: "gRPC · Tonic · unary + streaming + bidi",
      detail: "Unary, server-streaming (KServe), or bidi (Riva) per endpoint.",
      diagram: (
        <Diagram>
          <NodeChip accent>Tonic</NodeChip>
          <BiArrow />
          <NodeChip>stream</NodeChip>
        </Diagram>
      ),
      className: roleClassName("transport"),
    },
  },
  {
    id: DRY_RUN,
    type: "card",
    position: { x: 540, y: 220 },
    data: {
      title: "DryRunTransportFactoryV2",
      subtitle: "dry-run · no I/O",
      detail: "Strict decoder; synthesizes timings offline.",
      diagram: (
        <Diagram>
          <NodeChip>decode</NodeChip>
          <MiniArrow />
          <DiamondNode accent>no I/O</DiamondNode>
          <MiniArrow />
          <MiniBars heights={[40, 72, 100, 84]} />
        </Diagram>
      ),
      className: roleClassName("transport"),
    },
  },
  {
    id: DYNOSIM,
    type: "card",
    position: { x: 540, y: 330 },
    data: {
      title: "SteppableEngine",
      subtitle: "dynosim · offline co-sim",
      detail: "In-process Dynamo mocker via SteppableReplay.",
      diagram: (
        <Diagram>
          <NodeChip>req</NodeChip>
          <MiniArrow />
          <NodeChip accent>engine</NodeChip>
          <MiniArrow />
          <DbNode>replay</DbNode>
        </Diagram>
      ),
      className: roleClassName("server"),
    },
  },
];

const seamEdges: Edge[] = [
  { id: "e-transport-dispatch-seam", source: "transport__dispatcher", target: "transport__seam", type: "flow" },
  { id: "e-transport-seam-http", source: "transport__seam", target: HTTP, type: "flow" },
  { id: "e-transport-seam-grpc", source: "transport__seam", target: GRPC, type: "flow" },
  { id: "e-transport-seam-dryrun", source: "transport__seam", target: DRY_RUN, type: "flow" },
  { id: "e-transport-seam-dynosim", source: "transport__seam", target: DYNOSIM, type: "flow" },
];

/** A level-2 target subgraph: its `ExecutionSinkBuilder`, its `WorkerSink`, and its response shape. */
function leafChain(
  prefix: string,
  builder: { title: string; detail: string; role: NodeRole },
  sink: { title: string; detail: string; role: NodeRole },
  tail: { title: string; detail: string; role: NodeRole },
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [
    {
      id: `${prefix}__builder`,
      type: "card",
      position: { x: 0, y: 80 },
      data: {
        title: builder.title,
        subtitle: "ExecutionSinkBuilder",
        detail: builder.detail,
        className: roleClassName(builder.role),
      },
    },
    {
      id: `${prefix}__sink`,
      type: "card",
      position: { x: 280, y: 80 },
      data: {
        title: sink.title,
        subtitle: "WorkerSink",
        detail: sink.detail,
        className: roleClassName(sink.role),
      },
    },
    {
      id: `${prefix}__tail`,
      type: "card",
      position: { x: 560, y: 80 },
      data: { title: tail.title, detail: tail.detail, className: roleClassName(tail.role) },
    },
  ];
  const edges: Edge[] = [
    { id: `e-${prefix}-builder-sink`, source: `${prefix}__builder`, target: `${prefix}__sink`, type: "flow" },
    { id: `e-${prefix}-sink-tail`, source: `${prefix}__sink`, target: `${prefix}__tail`, type: "flow" },
  ];
  return { nodes, edges };
}

/**
 * Stage 6 — the Transport seam. Level-1 = Dispatcher → two-trait seam → four targets; level-2 =
 * each target's concrete `ExecutionSinkBuilder` + `WorkerSink` impls. Content and anchors are
 * grounded in real `rust/runtime` code (see the module header for verified `file:line`s).
 */
export const transportStage: StageDef = {
  id: "transport",
  order: 6,
  label: "Transport seam",
  caption:
    "A transport implements exactly two traits (WorkerSink + ExecutionSinkBuilder); everything else is shared. Four targets: HTTP (TransportSink, hyper, streaming), gRPC (GrpcTransportSink, Tonic — unary, server-streaming for KServe, bidi for Riva), dry-run, dynosim (offline co-sim).",
  tone: "yellow",
  // v2 timeline: the Dispatcher→sink hop in the Transport lane (inside the Transport seam frame).
  lane: "transport",
  events: [{ id: "tp-dispatch", label: "dispatch", laneId: "transport", atOrder: 8, realOffsetMs: 62 }],
  subgraph: {
    nodes: seamNodes,
    edges: seamEdges,
    children: [HTTP, GRPC, DRY_RUN, DYNOSIM],
  },
  leaves: {
    [HTTP]: {
      label: "HTTP transport (hyper, streaming)",
      ...leafChain(
        "http",
        {
          title: "HttpSinkBuilder",
          detail: 'label()="http"; build_sink → TransportSink.',
          role: "transport",
        },
        {
          title: "TransportSink",
          detail: "hyper client; dispatch_measured streams responses.",
          role: "transport",
        },
        {
          title: "SSE token stream",
          detail: "First token → on_first_token(ts)=TTFT; feeds reduce.",
          role: "transport",
        },
      ),
    },
    [GRPC]: {
      label: "gRPC transport (Tonic, unary + streaming + bidi)",
      ...leafChain(
        "grpc",
        {
          title: "GrpcSinkBuilder",
          detail: 'label()="grpc"; build_sink → GrpcTransportSink.',
          role: "transport",
        },
        {
          title: "GrpcTransportSink",
          detail: "Tonic client; streaming(streaming) per endpoint binding.",
          role: "transport",
        },
        {
          title: "Unary / stream / bidi",
          detail: "KServe ModelStreamInfer + Riva bidi; all share measure.",
          role: "transport",
        },
      ),
    },
    [DRY_RUN]: {
      label: "dry-run transport (no network I/O)",
      ...leafChain(
        "dryrun",
        {
          title: "DryRunTransportFactoryV2",
          detail: "TransportFactory for the dry_run strict decoder.",
          role: "transport",
        },
        {
          title: "DryRunNativeExecution",
          detail: "Synthesizes token timings offline (lognormal).",
          role: "compute",
        },
        {
          title: "Measured record",
          detail: "Same DispatchResult shape, no real request.",
          role: "storage",
        },
      ),
    },
    [DYNOSIM]: {
      label: "dynosim transport (offline co-sim)",
      ...leafChain(
        "dynosim",
        {
          title: "NativeDynamoEngineFactory",
          detail: "OfflineEngineFactory: build() → SteppableReplay.",
          role: "transport",
        },
        {
          title: "SteppableEngine",
          detail: "In-process Dynamo mocker; execute_pass offline.",
          role: "server",
        },
        {
          title: "SteppableReplay contract",
          detail: "Workload/clock/observers depend only on this.",
          role: "server",
        },
      ),
    },
  },
  evidence: [
    { label: "trait WorkerSink", path: "runtime/src/engine/turn_execution.rs:74" },
    { label: "trait ExecutionSinkBuilder", path: "runtime/src/engine/turn_execution.rs:136" },
    { label: "trait Dispatcher", path: "runtime/src/transport/core/dispatch.rs:332" },
    { label: "struct TransportSink (HTTP)", path: "runtime/src/transport/http/sink.rs:164" },
    { label: "struct GrpcTransportSink", path: "runtime/src/transport/grpc/sink.rs:102" },
    { label: "struct DryRunTransportFactoryV2", path: "runtime/src/engine/dry_run.rs:315" },
    { label: "SteppableEngine (dynosim)", path: "runtime/src/dynosim.rs:594" },
  ],
};

/**
 * The FlowStep fragment for a request traversing the Transport seam: shared Dispatcher → the
 * two-trait seam boundary → the chosen `WorkerSink` → its response shape. Active node ids match
 * the level-1 `subgraph`/leaf node ids so the shared `RequestParticle`/`useFlowPlayer` can drive
 * the particle through this stage. Captions name the real Rust types.
 */
export const transportFlowSteps: readonly FlowStep[] = [
  {
    nodeId: "transport__dispatcher",
    caption:
      "Rc<dyn Dispatcher> holds the run's chosen sink — upstream workload + SlotPool admission are identical for every target.",
    variant: "seam",
  },
  {
    nodeId: "transport__seam",
    caption:
      "The seam is exactly two traits: ExecutionSinkBuilder builds the !Send worker-local sink; WorkerSink.dispatch_measured drives one turn to terminal.",
    variant: "seam",
  },
  {
    nodeId: HTTP,
    caption:
      "HTTP: TransportSink (hyper) streams SSE tokens; first token → on_first_token(ts) marks TTFT.",
    variant: "http",
  },
  {
    nodeId: GRPC,
    caption:
      "gRPC: GrpcTransportSink (Tonic) — unary, server-streaming (KServe ModelStreamInfer), or bidirectional (Riva) depending on the endpoint binding.",
    variant: "grpc",
  },
  {
    nodeId: DRY_RUN,
    caption:
      "dry-run: DryRunTransportFactoryV2 / DryRunNativeExecution synthesize measured timings with no network I/O.",
    variant: "dry-run",
  },
  {
    nodeId: DYNOSIM,
    caption:
      "dynosim: NativeDynamoEngineFactory builds a SteppableEngine (Dynamo mocker) for offline co-simulation.",
    variant: "dynosim",
  },
];
