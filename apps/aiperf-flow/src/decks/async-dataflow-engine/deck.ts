/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the Graph-IR async dataflow engine, and the reference
//! example for `src/audio` autoplay.
//!
//! Every claim is grounded in `rust/runtime/src/graph/`; slide captions carry the
//! file and symbol so a viewer can go read the real thing. Diagrams draw the engine's
//! actual shape - AND-joins, successor fan-out, one arena feeding many citers - rather
//! than flattening each idea into a chain.

import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";
import { band, card, chip, fanIn, fanOut, link, note } from "./layout.js";

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "shape",
    eyebrow: "01 · THE SHAPE",
    title: "One node kind, one output channel",
    lede:
      "Many input requirements converge on a node; exactly one output channel leaves it. Branch, fork, spawn, and join are lowered away before the graph exists.",
    narration:
      "Start with what surprises people. There is exactly one executable node kind. Branching, forking, spawning, and joining are not node types at runtime. They are lowered away into plain nodes and edges before a graph record ever exists. Every node writes exactly one output channel, and reads a list of input requirements. That is the whole vocabulary.",
    caption:
      "graph/model.rs:226 GraphRecord {state, nodes, edges}; LlmNode:153 with a single `output: String`. graph/flat.rs:43 states the lowering claim outright.",
    nodes: [
      band("b-rec", "GRAPH RECORD", { col: 0, row: 0 }),
      card("record", "GraphRecord", "the whole program", "control", { col: 0, row: 1 }),
      card("state", "state", "channel -> spec", "data", { col: 1, row: 0 }),
      card("nodes", "nodes", "id -> LlmNode", "control", { col: 1, row: 1 }),
      card("edges", "edges", "StaticEdge list", "muted", { col: 1, row: 2 }),

      band("b-node", "ONE NODE, WHATEVER ITS FAN-IN", { col: 2.6, row: 0 }),
      card("in1", "input: prompt", "count 1", "data", { col: 2.6, row: 0 }),
      card("in2", "input: reply", "count 1", "data", { col: 2.6, row: 1 }),
      card("in3", "input: gate", "count all", "data", { col: 2.6, row: 2 }),
      card("llm", "LlmNode", "fires once, when all are met", "control", { col: 3.8, row: 1 }),
      card("out", "output", "exactly one channel", "done", { col: 4.9, row: 1 }),
    ],
    edges: [
      ...fanOut("record", ["state", "nodes", "edges"], "control", "slow"),
      ...fanIn(["in1", "in2", "in3"], "llm", "data"),
      link("llm", "out", "done"),
    ],
    revealOrder: ["b-rec", "record", "state", "nodes", "edges", "b-node", "in1", "in2", "in3", "llm", "out"],
  },
  {
    id: "compile",
    eyebrow: "02 · BEFORE ANYTHING RUNS",
    title: "Validation proves the graph cannot deadlock",
    lede:
      "Lowering, then four checks that fan out and must all pass. The fourth is a fireability fixpoint that subsumes self-dependency, unreachable producers, and impossible counts.",
    narration:
      "Compilation lowers authored workloads into graphs and traces, then validation runs four checks. Dangling edges. Undeclared channels. Unreachable nodes. And the interesting one: a fireability fixpoint. Repeatedly mark a node fireable once every input has enough producers that can themselves fire. Anything reachable that never becomes fireable is a deadlock, and the run refuses to start.",
    caption:
      "graph/validate.rs:28 validate(); the fixpoint at :79 subsumes self-dependency, unreachable producers, count > producers, and cyclic gates. graph/run.rs:40 calls it first.",
    nodes: [
      band("b-c", "LOWER", { col: 0, row: 1 }),
      card("authored", "authored workload", "dag_jsonl", "data", { col: 0, row: 1 }),
      card("lower", "lower_catalog", "one graph per root", "data", { col: 1, row: 1 }),
      card("plans", "GraphTracePlan", "unit of placement", "control", { col: 2, row: 1 }),

      band("b-v", "ALL FOUR MUST PASS", { col: 3.1, row: -0.5 }),
      card("v1", "edges resolve", "check 1", "muted", { col: 3.1, row: -0.5 }),
      card("v2", "channels declared", "check 2", "muted", { col: 3.1, row: 0.5 }),
      card("v3", "reachable", "check 3", "muted", { col: 3.1, row: 1.5 }),
      card("v4", "fireability fixpoint", "check 4 - deadlock-free", "time", { col: 3.1, row: 2.5 }),

      card("start", "run starts", "or refuses to", "done", { col: 4.3, row: 1 }),
      note(
        "why-fix",
        "Why a fixpoint, not a cycle check",
        "Marking a node fireable once its inputs have enough fireable producers catches self-dependency, unreachable producers, count > producers, and mutual gates in one pass.",
        { col: 2, row: 3.1 },
      ),
    ],
    edges: [
      link("authored", "lower", "data"),
      link("lower", "plans", "data"),
      ...fanOut("plans", ["v1", "v2", "v3", "v4"], "control"),
      ...fanIn(["v1", "v2", "v3", "v4"], "start", "done", "slow"),
    ],
    revealOrder: ["b-c", "authored", "lower", "plans", "b-v", "v1", "v2", "v3", "v4", "start", "why-fix"],
  },
  {
    id: "readiness",
    eyebrow: "03 · THE CORE IDEA",
    title: "Scheduling and readiness are different things",
    lede:
      "Topology spawns a task; the channel store decides when it may run. An AND-join waits for every producer, however many predecessors already finished.",
    narration:
      "This is the part worth slowing down for. There are two independent mechanisms and they are easy to confuse. The scheduler is pure topology: when a predecessor completes, its successors get spawned as tasks. But being spawned is not being ready. The spawned task immediately parks on the channel store, waiting for its input counts to be satisfied. Topology proposes. Channels decide.",
    caption:
      "graph/scheduler.rs:24 Scheduler is per-graph adjacency with no per-trace state. The real gate is executor.rs:465 prepare_node_inputs -> channel_store.rs:197 await_inputs.",
    nodes: [
      // The scheduler band sits directly above the gate so its hop drops straight in.
      // Running it in from the left made a long horizontal line that passed behind the
      // producer cards and read as though it were joining them.
      band("b-s", "TOPOLOGY PROPOSES", { col: 1.15, row: -1.7 }),
      card("pred", "predecessor done", "completion", "control", { col: 0, row: -1.7 }),
      card("spawn", "spawn_local", "task exists - not ready", "control", { col: 1.15, row: -1.7 }),

      band("b-p", "CHANNELS DECIDE", { col: 0, row: -0.3 }),
      card("pA", "producer A", "wrote - arrived", "done", { col: 0, row: -0.3 }),
      card("pB", "producer B", "wrote - arrived", "done", { col: 0, row: 0.7 }),
      card("pC", "producer C", "still running", "muted", { col: 0, row: 1.7 }),
      chip("gate", "count: all", { col: 1.15, row: 0.7 }),
      card("fires", "node fires", "only now", "done", { col: 1.9, row: 0.7 }),

      note(
        "note",
        "Spawned is not ready",
        "An AND fan-in node is scheduled by whichever predecessor finishes first, then parks until the rest arrive. Two of three producers is not enough.",
        { col: 2.9, row: 0.2 },
      ),
    ],
    edges: [
      link("pred", "spawn", "control"),
      link("spawn", "gate", "control", "slow"),
      ...fanIn(["pA", "pB"], "gate", "done"),
      link("pC", "gate", "muted", "slow"),
      link("gate", "fires", "done"),
    ],
    revealOrder: ["b-s", "pred", "spawn", "b-p", "pA", "pB", "pC", "gate", "fires", "note"],
  },
  {
    id: "park",
    eyebrow: "04 · WHY IT IS RACE-FREE",
    title: "Check, then park — and nothing runs in between",
    lede:
      "A closed loop: check, park, wake, re-check. Single-threaded per trace, so nothing runs between the check and the park and a notify cannot be lost.",
    narration:
      "Every wait in the engine follows the same shape. Synchronously check the condition. If it does not hold, clone the notifier and await it. Because a trace runs single-threaded, nothing else executes between the check and the park, so a wake can never slip through the gap. That is the entire reason this design needs no locks anywhere in per-trace state.",
    caption:
      "channel_store.rs:237 await_count is the canonical instance; same pattern at context.rs:76 for first-token and abort. All wakes use notify_waiters (wake-all, re-check), never notify_one.",
    nodes: [
      band("b-p", "EVERY WAIT SITE IS THIS LOOP", { col: 0.5, row: 0 }),
      card("check", "check count", "synchronous", "control", { col: 0.5, row: 0.6 }),
      chip("met", "met?", { col: 1.7, row: 0.6 }),
      card("park", "notified().await", "park", "time", { col: 2.4, row: 1.5 }),
      card("wake", "notify_waiters", "wake all", "control", { col: 1.2, row: 1.5 }),
      card("proceed", "proceed", "requirements met", "done", { col: 2.6, row: -0.3 }),

      note(
        "why",
        "No locks, by construction",
        "A trace is single-threaded, so nothing executes between the synchronous check and the await. A notify cannot slip through the gap.",
        { col: 0, row: 2.6 },
      ),
      note(
        "orphan",
        "Unsatisfiable readers orphan themselves",
        "If arrivals plus remaining producers cannot reach the target, this reader errors alone. The channel is not poisoned - a lower-count reader may still be satisfiable.",
        { col: 1.6, row: 2.6 },
      ),
    ],
    edges: [
      link("check", "met", "control"),
      link("met", "proceed", "done"),
      link("met", "park", "time"),
      link("park", "wake", "time", "slow"),
      link("wake", "check", "control", "slow"),
    ],
    revealOrder: ["b-p", "check", "met", "proceed", "park", "wake", "why", "orphan"],
  },
  {
    id: "gates",
    eyebrow: "05 · TIME",
    title: "Readiness freezes the version, then time passes",
    lede:
      "Four independent delay anchors converge on one firing instant via max, and the read happens at a sequence frozen before any of them elapse.",
    narration:
      "Once inputs are satisfied, the engine captures the current sequence number, and only then applies timing delays. That ordering matters. A write that lands while this node is sleeping through its firing delay is invisible to this firing. Reads are always at a version, never at whatever happens to be latest. It is what makes replays reproducible instead of subtly timing-dependent.",
    caption:
      "executor.rs:465 captures gate_seq immediately after await_inputs and before apply_firing_delay. Four edge delay kinds are max-combined in compute_firing_gate_us:508.",
    nodes: [
      band("b-d", "FOUR ANCHORS, MAX-COMBINED", { col: 0, row: -0.5 }),
      card("d1", "after first token", "predecessor TTFT", "time", { col: 0, row: -0.5 }),
      card("d2", "after completion", "predecessor finish", "time", { col: 0, row: 0.5 }),
      card("d3", "after dispatch", "predecessor start", "time", { col: 0, row: 1.5 }),
      card("d4", "min start delay", "node-level", "time", { col: 0, row: 2.5 }),
      chip("max", "max()", { col: 1.25, row: 1 }),
      card("gate", "firing instant", "sleep on Clock", "time", { col: 1.9, row: 1 }),

      band("b-g", "THE READ IS ALREADY FROZEN", { col: 3.1, row: 0.4 }),
      card("ready", "inputs satisfied", "step 1", "done", { col: 3.1, row: 0.4 }),
      card("seq", "freeze gate_seq", "step 2 - before the delay", "control", { col: 3.1, row: 1.4 }),
      card("read", "read at gate_seq", "step 4", "data", { col: 4.2, row: 1.4 }),

      note(
        "frozen",
        "A write during the sleep is invisible",
        "Reads happen at a version, never at latest. That is what makes a replay reproducible instead of subtly timing-dependent.",
        { col: 1.9, row: 2.5 },
      ),
    ],
    edges: [
      ...fanIn(["d1", "d2", "d3", "d4"], "max", "time"),
      link("max", "gate", "time"),
      link("ready", "seq", "done"),
      link("seq", "read", "data", "slow"),
      link("gate", "read", "time", "slow"),
    ],
    revealOrder: ["b-d", "d1", "d2", "d3", "d4", "max", "gate", "b-g", "ready", "seq", "read", "frozen"],
  },
  {
    id: "data",
    eyebrow: "06 · DATA MOVEMENT",
    title: "Content-addressed once, never reserialized",
    lede:
      "One content-addressed arena fans out to every node that cites it, and one reply fans out to every successor that splices it — with no reserialization on either path.",
    narration:
      "Prompt content is interned once into a single content-addressed arena, hashed with BLAKE3. Identity folds in the parent, so the same message under a different conversation prefix gets a different identity — which is exactly what you want for prefix caching. Channel values then carry two representations in lockstep: the parsed value, and the original wire bytes. Splicing a reply into the next prompt clones a reference count. It never re-encodes.",
    caption:
      "dataset/segment.rs:554 payload_id folds hash_parent; graph/segment.rs:426 asserts prefix-dependence. reducers.rs:262 ChanVal::EncodedMessages carries value + wires together.",
    nodes: [
      band("b-seg", "ONE ARENA, MANY CITERS", { col: 0, row: 0.5 }),
      card("pool", "SegmentPool", "intern once", "data", { col: 0, row: 0.5 }),
      card("freeze", "freeze()", "immutable store", "data", { col: 1, row: 0.5 }),
      card("h1", "node A items", "Handle(u32)", "muted", { col: 2.1, row: -0.4 }),
      card("h2", "node B items", "Handle(u32)", "muted", { col: 2.1, row: 0.5 }),
      card("h3", "node C items", "Handle(u32)", "muted", { col: 2.1, row: 1.4 }),

      band("b-flow", "ONE REPLY, MANY SPLICES", { col: 0, row: 2.6 }),
      card("reply", "GraphReply", "text + wire bytes", "done", { col: 0, row: 2.6 }),
      card("chan", "ChanVal", "value + wires, in lockstep", "data", { col: 1, row: 2.6 }),
      card("s1", "successor splice", "clones a refcount", "data", { col: 2.1, row: 2.1 }),
      card("s2", "successor splice", "clones a refcount", "data", { col: 2.1, row: 3.1 }),

      note(
        "hash",
        "Identity folds in the parent",
        "BLAKE3 over the payload and its parent digest, so the same message under a different conversation prefix is a different segment - exactly what prefix caching needs.",
        { col: 3.3, row: 0.5 },
      ),
    ],
    edges: [
      link("pool", "freeze", "data"),
      ...fanOut("freeze", ["h1", "h2", "h3"], "data", "slow"),
      link("reply", "chan", "done"),
      ...fanOut("chan", ["s1", "s2"], "data", "fast"),
    ],
    revealOrder: ["b-seg", "pool", "freeze", "h1", "h2", "h3", "b-flow", "reply", "chan", "s1", "s2", "hash"],
  },
  {
    id: "dispatch",
    eyebrow: "07 · IN FLIGHT",
    title: "First token does double duty",
    lede:
      "First token is a fan-out: it releases the prefill slot and unblocks anchored successors on the same edge, while the stream runs on to terminal.",
    narration:
      "When the first token arrives, two things happen on the same edge. The prefill admission slot is released — so prefill concurrency is bounded by outstanding time-to-first-token, not by request count. And successors anchored on first token are unblocked. Separately, successors on a start-anchored edge were already scheduled back at dispatch, which means one can be building its prompt while its predecessor is still streaming.",
    caption:
      "executor.rs:293 guards on a Cell<bool> and calls permit.on_first_token() plus ctx.set_first_token(). policy.rs:103 releases the slot on first token OR terminal. mark_dispatch_start:435 schedules start-anchored successors at dispatch.",
    nodes: [
      band("b-t", "ONE NODE IN FLIGHT", { col: 0, row: 1 }),
      card("admit", "admit", "prefill slot taken", "control", { col: 0, row: 1 }),
      card("disp", "dispatch", "request on the wire", "control", { col: 1, row: 1 }),
      card("ttft", "first token", "one event", "time", { col: 2, row: 1 }),

      card("slot", "release prefill slot", "bounded by outstanding TTFT", "done", { col: 3.1, row: 0 }),
      card("succ", "unblock anchored successors", "they may fire now", "done", { col: 3.1, row: 1 }),
      card("term", "stream to terminal", "reply completes", "data", { col: 3.1, row: 2 }),

      card("early", "start-anchored successor", "scheduled back at dispatch", "control", { col: 1, row: 2.6 }),
      note(
        "cancel",
        "Cancellation is a dropped future",
        "A biased select races abort against the dispatch. Dropping the future is what stops the request - and under a virtual clock, its remaining modeled latency too.",
        { col: 2.2, row: 3 },
      ),
    ],
    edges: [
      link("admit", "disp", "control"),
      link("disp", "ttft", "control"),
      ...fanOut("ttft", ["slot", "succ", "term"], "done"),
      link("disp", "early", "control", "slow"),
    ],
    revealOrder: ["b-t", "admit", "disp", "ttft", "slot", "succ", "term", "early", "cancel"],
  },
  {
    id: "finish",
    eyebrow: "08 · COMPLETION AND FAILURE",
    title: "Mutate state, then notify — in that order",
    lede:
      "Completion is strictly ordered. Failure is the opposite shape: one error fans out to poison every channel at once.",
    narration:
      "Completion has a strict ordering rule. Record the finish time and mark the node completed, and only then notify waiters. A successor gated on this node's first token has to see resolved state when it wakes, because it finished without ever producing one. Failure follows the same discipline: the first error wins, and it poisons every channel — because a fail-fast can prevent a downstream producer from ever being scheduled, and static accounting cannot decrement a node that never started.",
    caption:
      "executor.rs:444 finalize_node — the comment at :448 explains the ordering. context.rs:106 set_abort is first-write-wins and calls store.abort_all; channel_store.rs:432 explains why nothing narrower is sound.",
    nodes: [
      band("b-f", "ORDERED, AND THE ORDER MATTERS", { col: 0, row: 0.4 }),
      card("f1", "record finish", "1", "done", { col: 0, row: 0.4 }),
      card("f2", "mark completed", "2", "done", { col: 1, row: 0.4 }),
      card("f3", "notify waiters", "3 - only now", "control", { col: 2, row: 0.4 }),
      card("f4", "mark producer done", "4", "done", { col: 3, row: 0.4 }),

      band("b-e", "ONE ERROR, EVERY CHANNEL", { col: 0, row: 2 }),
      card("e1", "first error wins", "idempotent", "failure", { col: 0, row: 2 }),
      card("c1", "channel poisoned", "abort_all", "failure", { col: 1.4, row: 1.4 }),
      card("c2", "channel poisoned", "abort_all", "failure", { col: 1.4, row: 2.4 }),
      card("c3", "channel poisoned", "abort_all", "failure", { col: 1.4, row: 3.4 }),
      card("alt", "or continue with empty", "resilient policy", "muted", { col: 2.7, row: 2.4 }),

      note(
        "why-all",
        "Why nothing narrower is sound",
        "A fail-fast can stop a downstream producer from ever being scheduled, and static producer accounting cannot decrement a node that never started.",
        { col: 3.9, row: 2 },
      ),
    ],
    edges: [
      link("f1", "f2", "done"),
      link("f2", "f3", "done"),
      link("f3", "f4", "control"),
      ...fanOut("e1", ["c1", "c2", "c3"], "failure", "fast"),
      link("e1", "alt", "muted", "slow"),
    ],
    revealOrder: ["b-f", "f1", "f2", "f3", "f4", "b-e", "e1", "c1", "c2", "c3", "alt", "why-all"],
  },
  {
    id: "placement",
    eyebrow: "09 · SCALING OUT",
    title: "A trace is the atomic unit of placement",
    lede:
      "One controller fans traces out across workers, each trace moving whole. Simulation cannot take this path at all.",
    narration:
      "Finally, scaling. A whole trace moves as one unit, never individual node turns, because fan-out, joins, firing gates, and dynamic reply splices are all trace-local state owned by a single executor. Thread-per-core placement round-robins traces across threads. But note the trap: thread-per-core is fundamentally incompatible with the simulated clock, because the idle pump can only advance sleepers on the one reactor it drives. Simulation runs inline.",
    caption:
      "model.rs:250 states the atomicity rationale. placement.rs:51 ThreadPerCoreTracePlacement; the SimClock incompatibility is documented at placement.rs:259, which is why LocalTracePlacement exists.",
    nodes: [
      band("b-pl", "ONE CONTROLLER, N WORKERS", { col: 0, row: 1 }),
      card("plan", "GraphTracePlan", "moves as one unit", "control", { col: 0, row: 1 }),
      card("w0", "worker 0", "own runtime + LocalSet", "done", { col: 1.3, row: 0 }),
      card("w1", "worker 1", "own runtime + LocalSet", "done", { col: 1.3, row: 1 }),
      card("w2", "worker N", "own runtime + LocalSet", "done", { col: 1.3, row: 2 }),
      card("t0", "whole trace", "fan-out, joins, gates stay local", "data", { col: 2.5, row: 0 }),
      card("t1", "whole trace", "fan-out, joins, gates stay local", "data", { col: 2.5, row: 1 }),
      card("t2", "whole trace", "fan-out, joins, gates stay local", "data", { col: 2.5, row: 2 }),

      band("b-sim", "SIMULATION CANNOT", { col: 0, row: 3.2 }),
      card("sim", "SimClock", "inline placement only", "time", { col: 0, row: 3.2 }),
      note(
        "trap",
        "The idle pump drives one reactor",
        "It can only advance the sleepers of the reactor it drives. Work on other threads never has its virtual arrivals advanced, and the replay stalls after the first root node.",
        { col: 1.3, row: 3.2 },
      ),
    ],
    edges: [
      ...fanOut("plan", ["w0", "w1", "w2"], "control"),
      link("w0", "t0", "data", "slow"),
      link("w1", "t1", "data", "slow"),
      link("w2", "t2", "data", "slow"),
      link("sim", "trap", "time", "slow"),
    ],
    revealOrder: ["b-pl", "plan", "w0", "w1", "w2", "t0", "t1", "t2", "b-sim", "sim", "trap"],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/async-dataflow-engine`. */
export const ASYNC_DATAFLOW_ENGINE_DECK: DeckDefinition = {
  id: "async-dataflow-engine",
  title: "How the async dataflow engine works",
  slides: SLIDES,
};
