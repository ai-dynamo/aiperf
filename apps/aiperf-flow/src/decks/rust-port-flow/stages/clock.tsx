/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 5 — the Clock seam (virtual vs. real time).
//!
//! `runtime` never reads wall time directly: every timestamp, sleep, and timeout routes through the
//! `Clock` trait (`clock/clock.rs`). `Clock::is_virtual()` is the seam selector — `false` picks the
//! live [`RealClock`] (monotonic `Instant`, `timerfd` ns-precision sleeps, the real tokio reactor);
//! `true` picks the [`SimClock`] (integer-nanosecond virtual time advanced by a discrete-event pump,
//! same-time wakes ordered deterministically by `(at_ns, seq_no)`). The whole transport stack —
//! `now_ns`, `sleep`, `with_timeout` — is built against `Clock`, never `Instant::now` /
//! `SystemTime::now` / `tokio::time`, so a run can swap wall time for reproducible virtual time
//! without touching a line of transport code.
//!
//! Level 1 draws the seam: `trait Clock` → `is_virtual()` selector → { RealClock, SimClock } → the
//! transport-timing consumer. The two backend nodes drill (level 2) into their internals.

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";
import { Diagram, NodeChip, RoundNode, DbNode, DiamondNode, MiniArrow } from "../../../chalk/index.js";

/** Level-1 seam nodes: the trait, the `is_virtual()` selector, both backends, and the consumer. */
const subgraphNodes: Node[] = [
  {
    id: "clock__trait",
    type: "card",
    position: { x: 0, y: 150 },
    data: {
      title: "trait Clock",
      subtitle: "clock/clock.rs",
      detail: "now_ns · sleep · is_virtual · drive — one time source",
      diagram: (
        <Diagram>
          <NodeChip accent>Clock</NodeChip>
          <MiniArrow />
          <NodeChip>now_ns</NodeChip>
          <NodeChip>sleep</NodeChip>
          <DiamondNode>is_virtual</DiamondNode>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "clock__selector",
    type: "chip",
    position: { x: 268, y: 168 },
    data: {
      label: "is_virtual()?",
      className: roleClassName("neutral"),
    },
  },
  {
    id: "clockReal",
    type: "card",
    position: { x: 470, y: 0 },
    data: {
      title: "RealClock",
      subtitle: "false → real reactor",
      detail: "monotonic Instant + timerfd sleeps on current-thread tokio",
      diagram: (
        <Diagram>
          <NodeChip accent>Instant</NodeChip>
          <MiniArrow />
          <RoundNode>now_ns</RoundNode>
          <MiniArrow />
          <NodeChip>timerfd</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "clockSim",
    type: "card",
    position: { x: 470, y: 285 },
    data: {
      title: "SimClock",
      subtitle: "true → simulation driver",
      detail: "integer-ns virtual time, deterministic (at_ns, seq_no) order",
      diagram: (
        <Diagram>
          <NodeChip>(at_ns, seq)</NodeChip>
          <MiniArrow />
          <DbNode accent>BinaryHeap</DbNode>
          <MiniArrow />
          <NodeChip>wake</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "clock__transport",
    type: "card",
    position: { x: 800, y: 150 },
    data: {
      title: "Transport timing",
      subtitle: "one nanosecond timeline",
      detail: "now_ns · sleep · with_timeout — never Instant/SystemTime::now",
      diagram: (
        <Diagram>
          <NodeChip>now_ns</NodeChip>
          <NodeChip>with_timeout</NodeChip>
          <MiniArrow />
          <RoundNode accent>Clock</RoundNode>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
];

/** Trait → selector → each backend → the transport consumer that reads time through the seam. */
const subgraphEdges: Edge[] = [
  { id: "e-clock-trait-selector", source: "clock__trait", target: "clock__selector", type: "flow" },
  { id: "e-clock-selector-real", source: "clock__selector", target: "clockReal", type: "flow" },
  { id: "e-clock-selector-sim", source: "clock__selector", target: "clockSim", type: "flow" },
  { id: "e-clock-real-transport", source: "clockReal", target: "clock__transport", type: "flow" },
  { id: "e-clock-sim-transport", source: "clockSim", target: "clock__transport", type: "flow" },
];

/** Level-2 drill into RealClock's live wall-clock machinery. */
const realLeafNodes: Node[] = [
  {
    id: "real__anchor",
    type: "panel",
    position: { x: 0, y: 0 },
    data: {
      title: "RealClockAnchor",
      detail: "Copy monotonic origin (Instant); one shared timeline",
      diagram: (
        <Diagram>
          <NodeChip accent>t₀ Instant</NodeChip>
          <MiniArrow />
          <RoundNode>elapsed</RoundNode>
          <MiniArrow />
          <NodeChip>1 ns line</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "real__now",
    type: "panel",
    position: { x: 0, y: 160 },
    data: {
      title: "now_ns()",
      detail: "start.elapsed().as_nanos() as i64 — monotonic, not wall",
      diagram: (
        <Diagram>
          <NodeChip>start</NodeChip>
          <MiniArrow />
          <RoundNode>elapsed</RoundNode>
          <MiniArrow />
          <NodeChip accent>i64 ns</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "real__timerfd",
    type: "panel",
    position: { x: 330, y: 0 },
    data: {
      title: "timerfd_sleep_ns",
      detail: "one-shot CLOCK_MONOTONIC timerfd via AsyncFd (Linux)",
      diagram: (
        <Diagram>
          <NodeChip accent>timerfd</NodeChip>
          <MiniArrow />
          <RoundNode>AsyncFd</RoundNode>
          <MiniArrow />
          <NodeChip>wake</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "real__fallback",
    type: "panel",
    position: { x: 330, y: 160 },
    data: {
      title: "tokio::time fallback",
      detail: "non-Linux / timerfd failure → coarser 1 ms wheel",
      diagram: (
        <Diagram>
          <DiamondNode>no timerfd</DiamondNode>
          <MiniArrow />
          <NodeChip accent>tokio::time</NodeChip>
          <MiniArrow />
          <NodeChip>1 ms wheel</NodeChip>
        </Diagram>
      ),
      className: roleClassName("neutral"),
    },
  },
];

const realLeafEdges: Edge[] = [
  { id: "e-real-anchor-now", source: "real__anchor", target: "real__now" },
  { id: "e-real-now-timerfd", source: "real__now", target: "real__timerfd", type: "flow" },
  { id: "e-real-timerfd-fallback", source: "real__timerfd", target: "real__fallback", type: "flow" },
];

/** Level-2 drill into SimClock's deterministic discrete-event core. */
const simLeafNodes: Node[] = [
  {
    id: "sim__state",
    type: "panel",
    position: { x: 0, y: 0 },
    data: {
      title: "now_ns · seq · heap",
      detail: "Cell now_ns, Cell seq, BinaryHeap<Sleeper> of wakers",
      diagram: (
        <Diagram>
          <NodeChip>now_ns</NodeChip>
          <NodeChip>seq</NodeChip>
          <MiniArrow />
          <DbNode accent>Sleeper heap</DbNode>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "sim__schedule",
    type: "panel",
    position: { x: 0, y: 160 },
    data: {
      title: "schedule(at_ns, waker)",
      detail: "park Sleeper { at_ns, seq_no, waker }; seq_no = reg order",
      diagram: (
        <Diagram>
          <NodeChip>at_ns</NodeChip>
          <MiniArrow />
          <NodeChip accent>Sleeper</NodeChip>
          <MiniArrow />
          <DbNode>heap</DbNode>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "sim__order",
    type: "panel",
    position: { x: 330, y: 160 },
    data: {
      title: "(at_ns, seq_no) order",
      detail: "Ord: earliest deadline first, ties by seq — deterministic",
      diagram: (
        <Diagram>
          <NodeChip>at_ns</NodeChip>
          <MiniArrow />
          <DiamondNode accent>tie?</DiamondNode>
          <MiniArrow />
          <NodeChip>seq</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
  {
    id: "sim__advance",
    type: "panel",
    position: { x: 330, y: 0 },
    data: {
      title: "advance_to(ns)",
      detail: "fast-forward to next event; drain_due wakes sleepers",
      diagram: (
        <Diagram>
          <DbNode>heap</DbNode>
          <MiniArrow />
          <NodeChip accent>advance</NodeChip>
          <MiniArrow />
          <NodeChip>wake due</NodeChip>
        </Diagram>
      ),
      className: roleClassName("control"),
    },
  },
];

const simLeafEdges: Edge[] = [
  { id: "e-sim-schedule-state", source: "sim__schedule", target: "sim__state" },
  { id: "e-sim-schedule-order", source: "sim__schedule", target: "sim__order", type: "flow" },
  { id: "e-sim-order-advance", source: "sim__order", target: "sim__advance", type: "flow" },
  { id: "e-sim-advance-state", source: "sim__advance", target: "sim__state", type: "flow" },
];

/**
 * Level-1 play-layer fragment for the Clock seam: the request particle walks the trait, forks on
 * `is_virtual()`, and lands on both backends before the shared transport-timing consumer. Node ids
 * match `subgraphNodes` so the shared `RequestParticle` can highlight each in turn.
 */
export const clockFlowSteps: FlowStep[] = [
  {
    nodeId: "clock__trait",
    caption: "runtime reads time only through trait Clock — now_ns(), sleep(), is_virtual(), drive().",
  },
  {
    nodeId: "clock__selector",
    caption: "Clock::is_virtual() is the seam selector: false → real reactor, true → simulation driver.",
  },
  {
    nodeId: "clockReal",
    caption: "RealClock paces against monotonic wall time with timerfd ns-precision sleeps.",
    variant: "real",
  },
  {
    nodeId: "clockSim",
    caption: "SimClock advances virtual time in integer-ns hops, waking sleepers in (at_ns, seq_no) order.",
    variant: "sim",
  },
  {
    nodeId: "clock__transport",
    caption: "The whole transport stack times through the selected Clock — never Instant::now or tokio::time.",
  },
];

/** Stage 5 — the Clock seam. Fleshed out from the deck's stub with the RealClock/SimClock zoom. */
export const clockStage: StageDef = {
  id: "clock",
  order: 5,
  label: "Clock seam",
  caption:
    "The Clock trait: RealClock (wall time) vs SimClock (integer-nanosecond virtual time, deterministic (at_ns, seq_no) ordering); Clock::is_virtual() selects real-reactor vs simulation driver.",
  tone: "orange",
  // v2 timeline: the Clock seam times the transport hop — its region sits in the Transport lane, just
  // before dispatch (the whole-axis Clock seam frame is the top-level grouping).
  lane: "transport",
  events: [{ id: "ck-select", label: "Clock", laneId: "transport", atOrder: 7, realOffsetMs: 61 }],
  subgraph: {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    children: ["clockReal", "clockSim"],
  },
  leaves: {
    clockReal: {
      label: "RealClock — live wall-clock backend",
      nodes: realLeafNodes,
      edges: realLeafEdges,
    },
    clockSim: {
      label: "SimClock — deterministic virtual time",
      nodes: simLeafNodes,
      edges: simLeafEdges,
    },
  },
  evidence: [
    { label: "trait Clock", path: "runtime/src/clock/clock.rs:12" },
    { label: "Clock::is_virtual()", path: "runtime/src/clock/clock.rs:24" },
    { label: "Clock::drive (real reactor)", path: "runtime/src/clock/clock.rs:34" },
    { label: "RealClock", path: "runtime/src/clock/real_clock.rs:52" },
    { label: "timerfd sleep_ns", path: "runtime/src/clock/real_clock.rs:88" },
    { label: "SimClock", path: "runtime/src/clock/sim_clock.rs:48" },
    { label: "Sleeper (at_ns, seq_no) Ord", path: "runtime/src/clock/sim_clock.rs:31" },
    { label: "SimClock::advance_to", path: "runtime/src/clock/sim_clock.rs:106" },
    { label: "is_virtual() selects driver", path: "runtime/src/phase_runtime.rs:721" },
    { label: "transport times via Clock", path: "runtime/src/transport/http/client/connection.rs:359" },
  ],
};
