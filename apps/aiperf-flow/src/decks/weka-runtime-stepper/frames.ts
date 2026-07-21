/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Frame data for the weka-runtime-stepper deck, ported verbatim from
//! `docs/canvases/weka-runtime-stepper.canvas.tsx`'s `FRAMES` array. One weka trie trace as the
//! TraceExecutor drives it: a root Step (A), a concurrent fan-out (B, C), and an AND-fan-in join
//! (D) gated on two channel arrivals.

export type NodeState = "pending" | "ready" | "firing" | "done";

export type ChannelWrite = { name: string; seq: number };

export type Frame = {
  desc: string;
  states: Record<string, NodeState>;
  channels: ChannelWrite[];
  gateHave: number;
};

export const GATE_NEED = 2;

export const FRAMES: Frame[] = [
  {
    desc: "Scheduler seeds the frontier: A is the entry Step (a START successor). No central queue — readiness is task creation + channel waiters.",
    states: { A: "ready" },
    channels: [],
    gateHave: 0,
  },
  {
    desc: "A fires — effect: Dispatch. It builds a DispatchRequest, parks a Future in the CreditDispatchAdapter, and awaits the credit return.",
    states: { A: "firing" },
    channels: [],
    gateHave: 0,
  },
  {
    desc: "A's return resolves the Future. A writes A_out (seq 1) and marks its producer done; successors B and C are scheduled.",
    states: { A: "done", B: "ready", C: "ready" },
    channels: [{ name: "A_out", seq: 1 }],
    gateHave: 0,
  },
  {
    desc: "B and C fire concurrently — there is no edge between them, so both run at once (concurrency = absence of an edge).",
    states: { A: "done", B: "firing", C: "firing" },
    channels: [{ name: "A_out", seq: 1 }],
    gateHave: 0,
  },
  {
    desc: "B writes B_out (seq 2). D's AND-fan-in gate now holds 1 of 2 inputs and keeps waiting.",
    states: { A: "done", B: "done", C: "firing" },
    channels: [
      { name: "A_out", seq: 1 },
      { name: "B_out", seq: 2 },
    ],
    gateHave: 1,
  },
  {
    desc: "C writes C_out (seq 3). D's gate is satisfied (2 of 2) — await_inputs wakes and D becomes ready.",
    states: { A: "done", B: "done", C: "done", D: "ready" },
    channels: [
      { name: "A_out", seq: 1 },
      { name: "B_out", seq: 2 },
      { name: "C_out", seq: 3 },
    ],
    gateHave: 2,
  },
  {
    desc: "D fires — the recorded AND-join. The binding cause is whichever of B,C ended last; the prompt still comes from the segment pool, not the channel values.",
    states: { A: "done", B: "done", C: "done", D: "firing" },
    channels: [
      { name: "A_out", seq: 1 },
      { name: "B_out", seq: 2 },
      { name: "C_out", seq: 3 },
    ],
    gateHave: 2,
  },
  {
    desc: "D writes D_out (seq 4) and marks done. The frontier is empty — the trace is complete.",
    states: { A: "done", B: "done", C: "done", D: "done" },
    channels: [
      { name: "A_out", seq: 1 },
      { name: "B_out", seq: 2 },
      { name: "C_out", seq: 3 },
      { name: "D_out", seq: 4 },
    ],
    gateHave: 2,
  },
];

export const GRAPH_NODE_IDS = ["START", "A", "B", "C", "D"] as const;

export const GRAPH_EDGES: Array<{ from: string; to: string }> = [
  { from: "START", to: "A" },
  { from: "A", to: "B" },
  { from: "A", to: "C" },
  { from: "B", to: "D" },
  { from: "C", to: "D" },
];

/** State of a node at a given frame; `START` is always resolved as `"done"` (the entry point has already fired). */
export function stateOf(frame: Frame, id: string): NodeState {
  if (id === "START") {
    return "done";
  }
  return frame.states[id] ?? "pending";
}

export function stateLabel(s: NodeState): string {
  switch (s) {
    case "firing":
      return "firing";
    case "ready":
      return "ready";
    case "done":
      return "done";
    default:
      return "pending";
  }
}
