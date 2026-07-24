/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 4 — Workers synchronize & connect.
//!
//! The coordinator's `run_sharded_scheduled` spawns `W` thread-per-core sub-cell OS threads, each
//! running its own `current_thread` tokio runtime + `LocalSet`. Before the spawn it builds two
//! `Send`/`Copy` shared inputs on the main thread: a single `RealClockAnchor` monotonic origin
//! (every worker rebuilds a reactor-local `RealClock` from it, keeping one nanosecond timeline)
//! and a per-cell `GlobalAdmission` gate (shared `Arc<GlobalSlotPool>`/`GlobalRateGate` so the `W`
//! threads jointly enforce one cell-level cap/rate). Each worker runs `execute_scheduled_shard`,
//! stamping globally-unique two-level ordinals; `merge_shards` then concatenates and *sorts* by
//! `request_index` (it never renumbers). `workers == 1` collapses to one co-located transport sink
//! on the caller's reactor — the byte-unchanged single-worker path.
//!
//! Level-1 = the coordinator → shared-inputs → sub-cell → merge graph. Level-2 (`workers-thread`)
//! drills the per-thread `!Send` stack. Evidence anchors verified against `rust/` (trust code over
//! spec): file:line pinned by reading each cited function.

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { StageDef } from "../stage.js";

/** The drillable level-1 sub-cell node id, which is ALSO the `leaves` key it zooms into. */
const WORKER_LEAF_ID = "workers-thread";

/** Level-1: coordinator spawns W sub-cells sharing one clock origin + one admission gate, then merge. */
const level1Nodes: Node[] = [
  {
    id: "workers__coordinator",
    type: "card",
    position: { x: 0, y: 150 },
    data: {
      title: "run_sharded_scheduled",
      subtitle: "coordinator · main thread",
      detail: "Spawns W sub-cells over an mpsc; merge stays on main.",
      className: roleClassName("control"),
    },
  },
  {
    id: "workers__anchor",
    type: "panel",
    position: { x: 340, y: 0 },
    data: {
      title: "RealClockAnchor",
      detail: "One Copy monotonic origin; workers rebuild a local RealClock.",
      className: roleClassName("control"),
    },
  },
  {
    id: "workers__admission",
    type: "panel",
    position: { x: 340, y: 300 },
    data: {
      title: "GlobalAdmission",
      detail: "Shared Arc<GlobalSlotPool>/GlobalRateGate — one cell-level cap.",
      className: roleClassName("control"),
    },
  },
  {
    id: WORKER_LEAF_ID,
    type: "card",
    position: { x: 700, y: 150 },
    data: {
      title: "sub-cell worker × W",
      subtitle: "thread-per-core",
      detail: "Own current_thread runtime + LocalSet + sink. Click to open.",
      className: roleClassName("compute"),
    },
  },
  {
    id: "workers__merge",
    type: "card",
    position: { x: 1050, y: 150 },
    data: {
      title: "merge_shards",
      subtitle: "sort, don't renumber",
      detail: "Concatenates shards, sorts by request_index — never renumbers.",
      className: roleClassName("compute"),
    },
  },
];

const level1Edges: Edge[] = [
  { id: "e-coord-anchor", source: "workers__coordinator", target: "workers__anchor", type: "flow" },
  {
    id: "e-coord-admission",
    source: "workers__coordinator",
    target: "workers__admission",
    type: "flow",
  },
  {
    id: "e-coord-worker",
    source: "workers__coordinator",
    target: WORKER_LEAF_ID,
    type: "flow",
    label: "spawn × W",
  },
  { id: "e-anchor-worker", source: "workers__anchor", target: WORKER_LEAF_ID, label: "shared origin" },
  {
    id: "e-admission-worker",
    source: "workers__admission",
    target: WORKER_LEAF_ID,
    label: "shared gate",
  },
  { id: "e-worker-merge", source: WORKER_LEAF_ID, target: "workers__merge", type: "flow", label: "shard" },
];

/** Level-2: one worker's !Send per-thread stack, built inside the spawned thread from the shared inputs. */
const workerThreadNodes: Node[] = [
  {
    id: "wt__runtime",
    type: "panel",
    position: { x: 0, y: 0 },
    data: {
      title: "current_thread runtime",
      detail: "Builder::new_current_thread().enable_all().build()",
      className: roleClassName("compute"),
    },
  },
  {
    id: "wt__localset",
    type: "panel",
    position: { x: 0, y: 160 },
    data: {
      title: "LocalSet",
      detail: "block_on(local.run_until(execute_scheduled_shard(…)))",
      className: roleClassName("compute"),
    },
  },
  {
    id: "wt__clock",
    type: "panel",
    position: { x: 340, y: 0 },
    data: {
      title: "reactor-local RealClock",
      detail: "!Send Rc<RealClock> from the anchor; Sim clock forces workers==1.",
      className: roleClassName("control"),
    },
  },
  {
    id: "wt__sink",
    type: "panel",
    position: { x: 340, y: 160 },
    data: {
      title: "co-located transport sink",
      detail: "workers==1 keeps the sink on the caller's reactor — byte-unchanged.",
      className: roleClassName("transport"),
    },
  },
  {
    id: "wt__shard",
    type: "card",
    position: { x: 700, y: 80 },
    data: {
      title: "execute_scheduled_shard",
      subtitle: "worker_id",
      detail: "Runs the shard, stamping globally-unique two-level ordinals.",
      className: roleClassName("compute"),
    },
  },
];

const workerThreadEdges: Edge[] = [
  { id: "e-wt-runtime-localset", source: "wt__runtime", target: "wt__localset", type: "flow" },
  { id: "e-wt-localset-shard", source: "wt__localset", target: "wt__shard", type: "flow" },
  { id: "e-wt-clock-shard", source: "wt__clock", target: "wt__shard" },
  { id: "e-wt-sink-shard", source: "wt__sink", target: "wt__shard" },
];

export const workersStage: StageDef = {
  id: "workers",
  order: 4,
  label: "Workers sync & connect",
  caption:
    "coordinator → run_sharded_scheduled spawns W thread-per-core sub-cells (each its own current_thread runtime + LocalSet), a shared RealClockAnchor origin, a per-cell GlobalAdmission gate, merge_shards finalize (sort, don't renumber); workers == 1 is the byte-unchanged co-located path.",
  tone: "purple",
  // v2 timeline: worker fan-out + admission in the Scheduler lane (the Workload seam frames admit).
  lane: "scheduler",
  events: [
    { id: "wk-spawn", label: "spawn W", laneId: "scheduler", atOrder: 5, realOffsetMs: 52 },
    { id: "wk-admit", label: "admit", laneId: "scheduler", atOrder: 6, realOffsetMs: 60 },
  ],
  subgraph: {
    nodes: level1Nodes,
    edges: level1Edges,
    children: [WORKER_LEAF_ID],
  },
  leaves: {
    [WORKER_LEAF_ID]: {
      label: "Per-thread stack",
      nodes: workerThreadNodes,
      edges: workerThreadEdges,
    },
  },
  evidence: [
    { label: "run_sharded_scheduled", path: "runtime/src/engine/sharded_scheduled.rs:245" },
    { label: "run_worker_thread (current_thread + LocalSet)", path: "runtime/src/engine/sharded_scheduled.rs:342" },
    { label: "merge_shards (sort, not renumber)", path: "runtime/src/engine/sharded_scheduled.rs:358" },
    { label: "GlobalAdmission", path: "runtime/src/engine/execute/sharding.rs:25" },
    { label: "RealClockAnchor", path: "runtime/src/clock/real_clock.rs:27" },
    { label: "workers == 1 co-located (byte-unchanged)", path: "runtime/src/engine/turn_execution.rs:214" },
  ],
};
