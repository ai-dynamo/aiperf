/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 1 — Runtime & self-exec. The composition root: `aiperf-cli` resolves Config v2, projects it
//! into a protocol-v2 `EnvelopeV2`, and re-execs the SAME binary as `aiperf --execute` over stdio;
//! the child's `Coordinator::handle(EnvelopeV2)` is where the three orthogonal seams (Time /
//! Transport / Workload) are wired before it emits a one-line `RunTerminalV2`.
//!
//! Level-1 subgraph = the self-exec flow. Two level-2 leaves: the stdio handshake and the three
//! seams. `evidence` cites real `file:line` anchors verified against `rust/` (not the spec markdown).

import type { Edge, Node } from "@xyflow/react";
import { categoryBgTintClassName } from "../../../theme/tokens.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";

// Node ids that double as level-2 leaf keys (a click on one of these drills a level deeper). Kept
// stage-prefixed so they stay unique across every stage's leaves in the assembled ZoomTree.
const SELF_EXEC_LEAF = "runtime_selfexec";
const SEAMS_LEAF = "runtime_seams";

/** Level-1 subgraph: the front-door → self-exec → composition-root → terminal spine. */
const subgraphNodes: Node[] = [
  {
    id: "cli",
    type: "card",
    position: { x: 0, y: 40 },
    data: {
      title: "aiperf-cli",
      subtitle: "front door · cli/main.rs",
      detail: "Parses argv. On the non-execute path it resolves Config v2 and drives one execution child.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "envelope",
    type: "card",
    position: { x: 250, y: 40 },
    data: {
      title: "EnvelopeV2",
      subtitle: "protocol-v2 · stdio",
      detail: "protocol_version + OperationV2::Execute + a BenchmarkRunWireV2 (the exact Config v2 run).",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: SELF_EXEC_LEAF,
    type: "card",
    position: { x: 500, y: 40 },
    data: {
      title: "aiperf --execute",
      subtitle: "re-exec of current_exe()",
      detail: "exec_bin::resolve() picks this same binary; run_once spawns the child and feeds the envelope over stdin.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "coordinator",
    type: "card",
    position: { x: 750, y: 40 },
    data: {
      title: "Coordinator",
      subtitle: "composition root",
      detail: "handle(EnvelopeV2): concrete registries meet exactly once, then the frozen RunContext is handed to the transport/workload pair.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "terminal",
    type: "card",
    position: { x: 1000, y: 40 },
    data: {
      title: "RunTerminalV2",
      subtitle: "one-line stdout response",
      detail: "success + report_path on success, or a typed DiagnosticV2 failure envelope.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: SEAMS_LEAF,
    type: "card",
    position: { x: 750, y: 240 },
    data: {
      title: "Three orthogonal seams",
      subtitle: "Time · Transport · Workload",
      detail: "Wired here at the composition root, each varies independently. Click to open the seam axes.",
      className: categoryBgTintClassName("purple"),
    },
  },
];

const subgraphEdges: Edge[] = [
  { id: "e-cli-envelope", source: "cli", target: "envelope", type: "flow" },
  { id: "e-envelope-selfexec", source: "envelope", target: SELF_EXEC_LEAF, type: "flow" },
  { id: "e-selfexec-coordinator", source: SELF_EXEC_LEAF, target: "coordinator", type: "flow" },
  { id: "e-coordinator-terminal", source: "coordinator", target: "terminal", type: "flow" },
  { id: "e-coordinator-seams", source: "coordinator", target: SEAMS_LEAF, type: "flow", label: "opens 3 seams" },
];

/** Level-2 leaf: the parent ⇄ `--execute` child stdio protocol (the self-exec handshake). */
const selfExecLeafNodes: Node[] = [
  {
    id: "sx_parent",
    type: "card",
    position: { x: 0, y: 40 },
    data: {
      title: "execute::run_once",
      subtitle: "parent · cli/execute.rs",
      detail: "Spawns the child with the --execute flag and writes the EnvelopeV2 JSON to its stdin.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "sx_child",
    type: "card",
    position: { x: 300, y: 40 },
    data: {
      title: "aiperf --execute child",
      subtitle: "execute_mode::dispatch",
      detail: "Reads one bare protocol-v2 BenchmarkRun from stdin; the operation is chosen by argv MODE, not a wire field.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "sx_coord",
    type: "card",
    position: { x: 600, y: 40 },
    data: {
      title: "Coordinator::handle(EnvelopeV2)",
      subtitle: "composition root",
      detail: "Concrete registries meet once; the frozen RunContext goes to the transport/workload pair adapters.",
      className: categoryBgTintClassName("blue"),
    },
  },
  {
    id: "sx_reply",
    type: "card",
    position: { x: 600, y: 210 },
    data: {
      title: "RunTerminalV2 → stdout",
      detail: "Exactly one line back to the parent: success + report_path, or a typed failure envelope.",
      className: categoryBgTintClassName("green"),
    },
  },
];

const selfExecLeafEdges: Edge[] = [
  { id: "e-sx-parent-child", source: "sx_parent", target: "sx_child", type: "flow", label: "EnvelopeV2 / stdin" },
  { id: "e-sx-child-coord", source: "sx_child", target: "sx_coord", type: "flow" },
  { id: "e-sx-coord-reply", source: "sx_coord", target: "sx_reply", type: "flow" },
  { id: "e-sx-reply-parent", source: "sx_reply", target: "sx_parent", type: "flow", label: "stdout" },
];

/** Level-2 leaf: the three orthogonal seams as independent axes off the frozen RunContext. */
const seamsLeafNodes: Node[] = [
  {
    id: "sm_ctx",
    type: "card",
    position: { x: 0, y: 120 },
    data: {
      title: "RunContext (frozen)",
      subtitle: "handed to the pair adapters",
      detail: "The composition root freezes it once; the three seams below then vary completely independently.",
      className: categoryBgTintClassName("purple"),
    },
  },
  {
    id: "sm_time",
    type: "panel",
    position: { x: 340, y: 0 },
    data: {
      title: "Time · trait Clock",
      detail: "RealClock (wall) vs SimClock (virtual ns). is_virtual() selects the real reactor vs the simulation driver.",
      className: categoryBgTintClassName("green"),
    },
  },
  {
    id: "sm_transport",
    type: "panel",
    position: { x: 340, y: 120 },
    data: {
      title: "Transport · trait WorkerSink",
      detail: "Plus ExecutionSinkBuilder. One of HTTP / gRPC / dry-run / dynosim; everything else is shared.",
      className: categoryBgTintClassName("yellow"),
    },
  },
  {
    id: "sm_workload",
    type: "panel",
    position: { x: 340, y: 240 },
    data: {
      title: "Workload · trait Workload",
      detail: "RequestRateWorkload and friends generate the schedule the runtime drives.",
      className: categoryBgTintClassName("orange"),
    },
  },
];

const seamsLeafEdges: Edge[] = [
  { id: "e-sm-ctx-time", source: "sm_ctx", target: "sm_time", label: "Time" },
  { id: "e-sm-ctx-transport", source: "sm_ctx", target: "sm_transport", label: "Transport" },
  { id: "e-sm-ctx-workload", source: "sm_ctx", target: "sm_workload", label: "Workload" },
];

/**
 * Per-stage play fragment: an animated request particle traverses these level-1 node ids in order,
 * each caption naming the real types. Exported for a per-stage play traversal (the deck's overview
 * play layer builds its own one-step-per-stage sequence; this is the finer-grained runtime path).
 */
export const runtimeSteps: readonly FlowStep[] = [
  { nodeId: "cli", caption: "aiperf-cli parses argv and resolves Config v2 for the run." },
  {
    nodeId: "envelope",
    caption: "Config v2 is projected into a protocol-v2 EnvelopeV2 (OperationV2::Execute + BenchmarkRunWireV2).",
  },
  {
    nodeId: SELF_EXEC_LEAF,
    caption: "The binary re-execs itself as aiperf --execute (exec_bin::resolve = current_exe); the envelope goes over stdin.",
  },
  {
    nodeId: "coordinator",
    caption: "Coordinator::handle(EnvelopeV2) — the composition root; registries meet once and RunContext is frozen.",
  },
  {
    nodeId: SEAMS_LEAF,
    caption: "Three orthogonal seams open: Time (trait Clock), Transport (trait WorkerSink), Workload (trait Workload).",
  },
  { nodeId: "terminal", caption: "The child emits RunTerminalV2 on stdout: success + report_path." },
];

export const runtimeStage: StageDef = {
  id: "runtime",
  order: 1,
  label: "Runtime & self-exec",
  caption:
    "aiperf-cli → Config v2 → protocol-v2 EnvelopeV2 stdio → re-exec of the same binary in --execute mode (the composition root). Three orthogonal seams — Time / Transport / Workload — are wired here.",
  tone: "blue",
  subgraph: {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    children: [SELF_EXEC_LEAF, SEAMS_LEAF],
  },
  leaves: {
    [SELF_EXEC_LEAF]: {
      label: "Self-exec stdio handshake",
      nodes: selfExecLeafNodes,
      edges: selfExecLeafEdges,
    },
    [SEAMS_LEAF]: {
      label: "Three orthogonal seams",
      nodes: seamsLeafNodes,
      edges: seamsLeafEdges,
    },
  },
  evidence: [
    { label: "struct EnvelopeV2", path: "runtime/src/engine/protocol_v2.rs:115" },
    { label: "struct BenchmarkRunWireV2 (Config v2 run)", path: "runtime/src/engine/protocol_v2.rs:163" },
    { label: 'EXECUTE_FLAG "--execute"', path: "cli/src/execute_mode.rs:49" },
    { label: "is_execution_mode dispatch", path: "cli/src/main.rs:43" },
    { label: "exec_bin::resolve → current_exe()", path: "cli/src/exec_bin.rs:16" },
    { label: "execute::run_once (spawn child, stdio)", path: "cli/src/execute.rs:60" },
    { label: "Coordinator::handle (composition root)", path: "runtime/src/engine/coordinator.rs:114" },
    { label: "struct RunTerminalV2", path: "runtime/src/engine/protocol_v2.rs:1058" },
    { label: "trait Clock — Time seam", path: "runtime/src/clock/clock.rs:12" },
    { label: "trait WorkerSink — Transport seam", path: "runtime/src/engine/turn_execution.rs:74" },
    { label: "trait Workload — Workload seam", path: "runtime/src/scheduled.rs:1115" },
  ],
};
