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
import { categoryBgTintClassName } from "../../../theme/tokens.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";

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
      detail: "Same workload + SlotPool admission for every target; it just holds a sink.",
    },
  },
  {
    id: "transport__seam",
    type: "card",
    position: { x: 260, y: 150 },
    data: {
      title: "Two-trait seam",
      subtitle: "WorkerSink + ExecutionSinkBuilder",
      detail: "Implement exactly these two traits — everything else is shared.",
      className: categoryBgTintClassName("yellow"),
    },
  },
  {
    id: HTTP,
    type: "card",
    position: { x: 540, y: 0 },
    data: {
      title: "TransportSink",
      subtitle: "HTTP · hyper · streaming",
      detail: "supports_response_streaming() = true — live SSE tokens.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: GRPC,
    type: "card",
    position: { x: 540, y: 110 },
    data: {
      title: "GrpcTransportSink",
      subtitle: "gRPC · Tonic · non-streaming",
      detail: "supports_response_streaming() = false — one unary response.",
      className: categoryBgTintClassName("cyan"),
    },
  },
  {
    id: DRY_RUN,
    type: "card",
    position: { x: 540, y: 220 },
    data: {
      title: "DryRunTransportFactoryV2",
      subtitle: "dry-run · no I/O",
      detail: "Always-built strict decoder; synthesizes timings offline.",
      className: categoryBgTintClassName("gray"),
    },
  },
  {
    id: DYNOSIM,
    type: "card",
    position: { x: 540, y: 330 },
    data: {
      title: "SteppableEngine",
      subtitle: "dynosim · offline co-sim",
      detail: "In-process Dynamo mocker driven through SteppableReplay.",
      className: categoryBgTintClassName("orange"),
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
  builder: { title: string; detail: string },
  sink: { title: string; detail: string },
  tail: { title: string; detail: string },
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [
    {
      id: `${prefix}__builder`,
      type: "card",
      position: { x: 0, y: 80 },
      data: { title: builder.title, subtitle: "ExecutionSinkBuilder", detail: builder.detail },
    },
    {
      id: `${prefix}__sink`,
      type: "card",
      position: { x: 280, y: 80 },
      data: { title: sink.title, subtitle: "WorkerSink", detail: sink.detail },
    },
    {
      id: `${prefix}__tail`,
      type: "card",
      position: { x: 560, y: 80 },
      data: { title: tail.title, detail: tail.detail },
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
    "A transport implements exactly two traits (WorkerSink + ExecutionSinkBuilder); everything else is shared. Four targets: HTTP (TransportSink, hyper, streaming), gRPC (GrpcTransportSink, Tonic, non-streaming), dry-run, dynosim (offline co-sim).",
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
          detail: 'label() = "http"; build_sink(clock, worker_id) → TransportSink.',
        },
        {
          title: "TransportSink",
          detail: "hyper client; dispatch_measured streams intermediate parsed responses.",
        },
        {
          title: "SSE token stream",
          detail: "First token fires on_first_token(ts) → TTFT; each token feeds shared reduce.",
        },
      ),
    },
    [GRPC]: {
      label: "gRPC transport (Tonic, non-streaming)",
      ...leafChain(
        "grpc",
        {
          title: "GrpcSinkBuilder",
          detail: 'label() = "grpc"; build_sink(clock, worker_id) → GrpcTransportSink.',
        },
        {
          title: "GrpcTransportSink",
          detail: "Tonic client; supports_response_streaming() = false.",
        },
        {
          title: "Unary response",
          detail: "One terminal response — no live token stream; still shares measure.",
        },
      ),
    },
    [DRY_RUN]: {
      label: "dry-run transport (no network I/O)",
      ...leafChain(
        "dryrun",
        {
          title: "DryRunTransportFactoryV2",
          detail: "TransportFactory for the always-built dry_run strict decoder.",
        },
        {
          title: "DryRunNativeExecution",
          detail: "Synthesizes measured token timings offline (lognormal) — no server.",
        },
        {
          title: "Measured record",
          detail: "Emits the same DispatchResult shape without any real request.",
        },
      ),
    },
    [DYNOSIM]: {
      label: "dynosim transport (offline co-sim)",
      ...leafChain(
        "dynosim",
        {
          title: "NativeDynamoEngineFactory",
          detail: "OfflineEngineFactory: build() → Box<dyn SteppableReplay>.",
        },
        {
          title: "SteppableEngine",
          detail: "In-process Dynamo mocker; execute_pass single-runtime offline path.",
        },
        {
          title: "SteppableReplay contract",
          detail: "Workload, clock pump, observers depend only on this — swap the simulator freely.",
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
      "gRPC: GrpcTransportSink (Tonic) returns one unary response — supports_response_streaming() = false.",
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
