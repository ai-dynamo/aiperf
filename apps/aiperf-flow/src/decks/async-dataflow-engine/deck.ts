/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the Graph-IR async dataflow engine, and the reference
//! example for `src/audio` autoplay.
//!
//! Every claim is grounded in `rust/runtime/src/graph/`; slide captions carry the
//! file and symbol so a viewer can go read the real thing.

import type { Edge, Node } from "@xyflow/react";
import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";

const COLUMN_X = 300;

/**
 * Left-to-right row of cards at a fixed vertical band.
 *
 * `startColumn` indents the row. Cards use left/right handles, so an edge into a row
 * that starts further left than its source has to hook back around; indenting the
 * lower band so the target sits under its source keeps that hop a short vertical curve.
 */
function row(
  y: number,
  cards: readonly { id: string; title: string; subtitle?: string }[],
  startColumn = 0,
): Node[] {
  return cards.map((card, index) => ({
    id: card.id,
    type: "card",
    position: { x: (startColumn + index) * COLUMN_X, y },
    data: { title: card.title, subtitle: card.subtitle },
  }));
}

function band(id: string, title: string, y: number, startColumn = 0): Node {
  return {
    id,
    type: "header",
    position: { x: startColumn * COLUMN_X, y },
    data: { title },
  };
}

/**
 * Edge tones. Colour carries meaning here rather than decoration: a viewer should be
 * able to tell a control-flow hop from a data hand-off without reading the labels.
 */
const TONE = {
  /** Control flow: scheduling, ordering, the default hop. */
  control: "var(--color-accent-primary)",
  /** Data movement: content and channel values. */
  data: "var(--color-category-purple)",
  /** Time: firing gates and delay anchors. */
  time: "var(--color-category-yellow)",
  /** Failure and abort paths. */
  failure: "var(--color-category-red)",
  /** Successful completion. */
  done: "var(--color-category-green)",
} as const;

type Tone = keyof typeof TONE;

/**
 * Animated flow connector. Ids are derived from the pair, which React Flow requires
 * to be unique — no slide connects the same pair twice.
 */
function link(
  source: string,
  target: string,
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    type: "flow",
    data: { color: TONE[tone], speed },
  };
}

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "shape",
    eyebrow: "01 · THE SHAPE",
    title: "One node kind, one output channel",
    lede: "A graph is state channels, LlmNodes, and static edges. Branch, fork, spawn, and join are lowered away before the graph exists.",
    narration:
      "Start with what surprises people. There is exactly one executable node kind. Branching, forking, spawning, and joining are not node types at runtime. They are lowered away into plain nodes and edges before a graph record ever exists. Every node writes exactly one output channel, and reads a list of input requirements. That is the whole vocabulary.",
    caption:
      "graph/model.rs:226 GraphRecord {state, nodes, edges}; LlmNode:153 with a single `output: String`. graph/flat.rs:43 states the lowering claim outright.",
    nodes: [
      band("b-ir", "GRAPH RECORD", 0),
      ...row(40, [
        { id: "state", title: "state", subtitle: "channel -> spec" },
        { id: "nodes", title: "nodes", subtitle: "id -> LlmNode" },
        { id: "edges", title: "edges", subtitle: "StaticEdge list" },
      ]),
      // Indented so `inputs` sits directly under its source, `nodes`.
      band("b-node", "ONE NODE", 190, 1),
      ...row(
        230,
        [
          { id: "inputs", title: "inputs", subtitle: "ChannelRequirement" },
          { id: "items", title: "items", subtitle: "prompt program" },
          { id: "output", title: "output", subtitle: "exactly one channel" },
        ],
        1,
      ),
    ],
    edges: [
      link("nodes", "inputs", "control", "slow"),
      link("inputs", "items", "data"),
      link("items", "output", "data"),
    ],
    revealOrder: ["b-ir", "state", "nodes", "edges", "b-node", "inputs", "items", "output"],
  },
  {
    id: "compile",
    eyebrow: "02 · BEFORE ANYTHING RUNS",
    title: "Validation proves the graph cannot deadlock",
    lede: "Four static checks. The fourth is a fireability fixpoint that subsumes self-dependency, unreachable producers, and impossible counts.",
    narration:
      "Compilation lowers authored workloads into graphs and traces, then validation runs four checks. Dangling edges. Undeclared channels. Unreachable nodes. And the interesting one: a fireability fixpoint. Repeatedly mark a node fireable once every input has enough producers that can themselves fire. Anything reachable that never becomes fireable is a deadlock, and the run refuses to start.",
    caption:
      "graph/validate.rs:28 validate(); the fixpoint at :79 subsumes self-dependency, unreachable producers, count > producers, and cyclic gates. graph/run.rs:40 calls it first.",
    nodes: [
      band("b-c", "COMPILE", 0),
      ...row(40, [
        { id: "authored", title: "authored workload", subtitle: "dag_jsonl" },
        { id: "lower", title: "lower_catalog", subtitle: "one graph per root" },
        { id: "plans", title: "GraphTracePlan", subtitle: "unit of placement" },
      ]),
      // Indented so the first check sits under `plans`, which feeds it.
      band("b-v", "VALIDATE", 190, 2),
      ...row(
        230,
        [
          { id: "v1", title: "edges resolve", subtitle: "check 1" },
          { id: "v2", title: "channels declared", subtitle: "check 2" },
          { id: "v3", title: "reachable", subtitle: "check 3" },
          { id: "v4", title: "fireability fixpoint", subtitle: "check 4 · deadlock-free" },
        ],
        2,
      ),
    ],
    edges: [
      link("authored", "lower", "data"),
      link("lower", "plans", "data"),
      link("plans", "v1", "control", "slow"),
      link("v1", "v2"),
      link("v2", "v3"),
      link("v3", "v4"),
    ],
    revealOrder: ["b-c", "authored", "lower", "plans", "b-v", "v1", "v2", "v3", "v4"],
  },
  {
    id: "readiness",
    eyebrow: "03 · THE CORE IDEA",
    title: "Scheduling and readiness are different things",
    lede: "The scheduler decides when a node is considered. The channel store decides when it may actually run.",
    narration:
      "This is the part worth slowing down for. There are two independent mechanisms and they are easy to confuse. The scheduler is pure topology: when a predecessor completes, its successors get spawned as tasks. But being spawned is not being ready. The spawned task immediately parks on the channel store, waiting for its input counts to be satisfied. Topology proposes. Channels decide.",
    caption:
      "graph/scheduler.rs:24 Scheduler is per-graph adjacency with no per-trace state. The real gate is executor.rs:465 prepare_node_inputs -> channel_store.rs:197 await_inputs.",
    nodes: [
      band("b-s", "SCHEDULER · TOPOLOGY", 0),
      ...row(40, [
        { id: "pred", title: "predecessor done", subtitle: "completion" },
        { id: "spawn", title: "spawn_local", subtitle: "task exists" },
      ]),
      band("b-r", "CHANNEL STORE · READINESS", 190),
      ...row(230, [
        { id: "await", title: "await_inputs", subtitle: "counts unmet -> park" },
        { id: "ready", title: "requirements met", subtitle: "now it may fire" },
      ]),
      {
        id: "note",
        type: "panel",
        position: { x: 640, y: 130 },
        data: {
          title: "Spawned is not ready",
          detail: "A node with AND fan-in is scheduled by whichever predecessor finishes first, then blocks until the rest arrive.",
        },
      },
    ],
    edges: [
      link("pred", "spawn"),
      link("spawn", "await", "control", "slow"),
      link("await", "ready", "done"),
    ],
    revealOrder: ["b-s", "pred", "spawn", "b-r", "await", "ready", "note"],
  },
  {
    id: "park",
    eyebrow: "04 · WHY IT IS RACE-FREE",
    title: "Check, then park — and nothing runs in between",
    lede: "Single-threaded per trace, so the synchronous check and the await are atomic with respect to other tasks. A notify cannot be lost.",
    narration:
      "Every wait in the engine follows the same shape. Synchronously check the condition. If it does not hold, clone the notifier and await it. Because a trace runs single-threaded, nothing else executes between the check and the park, so a wake can never slip through the gap. That is the entire reason this design needs no locks anywhere in per-trace state.",
    caption:
      "channel_store.rs:237 await_count is the canonical instance; same pattern at context.rs:76 for first-token and abort. All wakes use notify_waiters (wake-all, re-check), never notify_one.",
    nodes: [
      band("b-p", "EVERY WAIT SITE", 0),
      ...row(40, [
        { id: "check", title: "check count", subtitle: "synchronous" },
        { id: "clone", title: "clone Notify", subtitle: "no await yet" },
        { id: "park", title: "notified().await", subtitle: "park" },
        { id: "recheck", title: "wake · re-check", subtitle: "loop" },
      ]),
      {
        id: "why",
        type: "panel",
        position: { x: 0, y: 200 },
        data: {
          title: "Rc and RefCell, zero locks",
          detail: "Per-trace state is single-threaded by construction. Parallelism is across traces on separate threads, never inside one.",
        },
      },
      {
        id: "orphan",
        type: "panel",
        position: { x: 460, y: 200 },
        data: {
          title: "Unsatisfiable readers orphan themselves",
          detail: "If arrivals plus remaining producers cannot reach the target, this reader errors — the channel is not poisoned, since a lower-count reader may still be satisfiable.",
        },
      },
    ],
    edges: [
      link("check", "clone"),
      link("clone", "park", "time"),
      link("park", "recheck", "time", "slow"),
    ],
    revealOrder: ["b-p", "check", "clone", "park", "recheck", "why", "orphan"],
  },
  {
    id: "gates",
    eyebrow: "05 · TIME",
    title: "Readiness freezes the version, then time passes",
    lede: "Reads happen at a sequence number captured before the firing delay — never at 'latest'.",
    narration:
      "Once inputs are satisfied, the engine captures the current sequence number, and only then applies timing delays. That ordering matters. A write that lands while this node is sleeping through its firing delay is invisible to this firing. Reads are always at a version, never at whatever happens to be latest. It is what makes replays reproducible instead of subtly timing-dependent.",
    caption:
      "executor.rs:465 captures gate_seq immediately after await_inputs and before apply_firing_delay. Four edge delay kinds are max-combined in compute_firing_gate_us:508.",
    nodes: [
      band("b-g", "FIRING SEQUENCE", 0),
      ...row(40, [
        { id: "g1", title: "inputs satisfied", subtitle: "step 1" },
        { id: "g2", title: "freeze gate_seq", subtitle: "step 2" },
        { id: "g3", title: "sleep on Clock", subtitle: "step 3" },
        { id: "g4", title: "read at gate_seq", subtitle: "step 4" },
      ]),
      band("b-d", "FOUR DELAY ANCHORS · MAX-COMBINED", 190),
      ...row(230, [
        { id: "d1", title: "after first token", subtitle: "predecessor TTFT" },
        { id: "d2", title: "after completion", subtitle: "predecessor finish" },
        { id: "d3", title: "after dispatch", subtitle: "predecessor start" },
        { id: "d4", title: "min start delay", subtitle: "node-level" },
      ]),
    ],
    edges: [
      link("g1", "g2", "done"),
      link("g2", "g3", "time"),
      link("g3", "g4", "data"),
    ],
    revealOrder: ["b-g", "g1", "g2", "g3", "g4", "b-d", "d1", "d2", "d3", "d4"],
  },
  {
    id: "data",
    eyebrow: "06 · DATA MOVEMENT",
    title: "Content-addressed once, never reserialized",
    lede: "Segment identity is prefix-dependent BLAKE3. Channel values carry both a JSON value and the pre-serialized wire bytes.",
    narration:
      "Prompt content is interned once into a single content-addressed arena, hashed with BLAKE3. Identity folds in the parent, so the same message under a different conversation prefix gets a different identity — which is exactly what you want for prefix caching. Channel values then carry two representations in lockstep: the parsed value, and the original wire bytes. Splicing a reply into the next prompt clones a reference count. It never re-encodes.",
    caption:
      "dataset/segment.rs:554 payload_id folds hash_parent; graph/segment.rs:426 asserts prefix-dependence. reducers.rs:262 ChanVal::EncodedMessages carries value + wires together.",
    nodes: [
      band("b-seg", "ONE SHARED ARENA", 0),
      ...row(40, [
        { id: "pool", title: "SegmentPool", subtitle: "write side" },
        { id: "freeze", title: "freeze()", subtitle: "immutable" },
        { id: "handle", title: "Handle(u32)", subtitle: "dense index · serde-ready" },
      ]),
      band("b-flow", "REPLY -> NEXT PROMPT", 190),
      ...row(230, [
        { id: "reply", title: "GraphReply", subtitle: "text + wire bytes" },
        { id: "chan", title: "ChanVal", subtitle: "value + wires" },
        { id: "splice", title: "splice", subtitle: "clones Bytes refcount" },
      ]),
    ],
    edges: [
      link("pool", "freeze", "data"),
      link("freeze", "handle", "data"),
      link("reply", "chan", "data"),
      link("chan", "splice", "data", "fast"),
    ],
    revealOrder: ["b-seg", "pool", "freeze", "handle", "b-flow", "reply", "chan", "splice"],
  },
  {
    id: "dispatch",
    eyebrow: "07 · IN FLIGHT",
    title: "First token does double duty",
    lede: "One event releases the prefill slot and unblocks first-token-anchored successors.",
    narration:
      "When the first token arrives, two things happen on the same edge. The prefill admission slot is released — so prefill concurrency is bounded by outstanding time-to-first-token, not by request count. And successors anchored on first token are unblocked. Separately, successors on a start-anchored edge were already scheduled back at dispatch, which means one can be building its prompt while its predecessor is still streaming.",
    caption:
      "executor.rs:293 guards on a Cell<bool> and calls permit.on_first_token() plus ctx.set_first_token(). policy.rs:103 releases the slot on first token OR terminal. mark_dispatch_start:435 schedules start-anchored successors at dispatch.",
    nodes: [
      band("b-t", "TIMELINE OF ONE NODE", 0),
      ...row(40, [
        { id: "admit", title: "admit", subtitle: "prefill slot taken" },
        { id: "disp", title: "dispatch", subtitle: "start-anchored fire now" },
        { id: "ttft", title: "first token", subtitle: "slot released" },
        { id: "term", title: "terminal", subtitle: "reply complete" },
      ]),
      {
        id: "cancel",
        type: "panel",
        position: { x: 0, y: 200 },
        data: {
          title: "Cancellation is a dropped future",
          detail: "A biased select races abort against the dispatch. Dropping the dispatch future is what actually stops the request — and under a virtual clock, its remaining modeled latency too.",
        },
      },
    ],
    edges: [
      link("admit", "disp"),
      link("disp", "ttft", "time"),
      link("ttft", "term", "done", "fast"),
    ],
    revealOrder: ["b-t", "admit", "disp", "ttft", "term", "cancel"],
  },
  {
    id: "finish",
    eyebrow: "08 · COMPLETION AND FAILURE",
    title: "Mutate state, then notify — in that order",
    lede: "A successor woken before the state it is about to read has been written will make the wrong decision.",
    narration:
      "Completion has a strict ordering rule. Record the finish time and mark the node completed, and only then notify waiters. A successor gated on this node's first token has to see resolved state when it wakes, because it finished without ever producing one. Failure follows the same discipline: the first error wins, and it poisons every channel — because a fail-fast can prevent a downstream producer from ever being scheduled, and static accounting cannot decrement a node that never started.",
    caption:
      "executor.rs:444 finalize_node — the comment at :448 explains the ordering. context.rs:106 set_abort is first-write-wins and calls store.abort_all; channel_store.rs:432 explains why nothing narrower is sound.",
    nodes: [
      band("b-f", "FINALIZE ORDER", 0),
      ...row(40, [
        { id: "f1", title: "record finish", subtitle: "1" },
        { id: "f2", title: "mark completed", subtitle: "2" },
        { id: "f3", title: "notify waiters", subtitle: "3 · last" },
        { id: "f4", title: "mark producer done", subtitle: "4" },
      ]),
      band("b-e", "ON FAILURE", 190),
      ...row(230, [
        { id: "e1", title: "first error wins", subtitle: "idempotent" },
        { id: "e2", title: "poison all channels", subtitle: "abort_all" },
        { id: "e3", title: "or continue empty", subtitle: "resilient policy" },
      ]),
    ],
    edges: [
      link("f1", "f2"),
      link("f2", "f3", "done"),
      link("f3", "f4", "done"),
      link("e1", "e2", "failure"),
      link("e1", "e3", "failure", "slow"),
    ],
    revealOrder: ["b-f", "f1", "f2", "f3", "f4", "b-e", "e1", "e2", "e3"],
  },
  {
    id: "placement",
    eyebrow: "09 · SCALING OUT",
    title: "A trace is the atomic unit of placement",
    lede: "Fan-out, joins, firing gates, and reply splices are trace-local state owned by one executor. Individual node turns are never distributed.",
    narration:
      "Finally, scaling. A whole trace moves as one unit, never individual node turns, because fan-out, joins, firing gates, and dynamic reply splices are all trace-local state owned by a single executor. Thread-per-core placement round-robins traces across threads. But note the trap: thread-per-core is fundamentally incompatible with the simulated clock, because the idle pump can only advance sleepers on the one reactor it drives. Simulation runs inline.",
    caption:
      "model.rs:250 states the atomicity rationale. placement.rs:51 ThreadPerCoreTracePlacement; the SimClock incompatibility is documented at placement.rs:259, which is why LocalTracePlacement exists.",
    nodes: [
      band("b-pl", "PLACEMENT", 0),
      ...row(40, [
        { id: "plan", title: "GraphTracePlan", subtitle: "moves as one unit" },
        { id: "tpc", title: "thread-per-core", subtitle: "real clock" },
        { id: "local", title: "local inline", subtitle: "sim clock" },
      ]),
      {
        id: "trap",
        type: "panel",
        position: { x: 0, y: 200 },
        data: {
          title: "Why simulation cannot be thread-per-core",
          detail: "The idle pump advances virtual time only for the reactor it drives. Work on other threads never has its arrivals advanced, and the replay stalls after the first root node.",
        },
      },
      {
        id: "park2",
        type: "panel",
        position: { x: 460, y: 200 },
        data: {
          title: "Never yield_now under virtual time",
          detail: "A yield self-wakes, so the pump re-polls the same instant instead of advancing. Park on the Clock. ClockStarved exists to name this failure.",
        },
      },
    ],
    edges: [link("plan", "tpc", "control"), link("plan", "local", "time", "slow")],
    revealOrder: ["b-pl", "plan", "tpc", "local", "trap", "park2"],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/async-dataflow-engine`. */
export const ASYNC_DATAFLOW_ENGINE_DECK: DeckDefinition = {
  id: "async-dataflow-engine",
  title: "How the async dataflow engine works",
  slides: SLIDES,
};
