/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the Python Graph Workload plane: Dynamo trace ingest,
//! workload construction, and the asyncio dataflow executor.
//!
//! Grounded in `src/aiperf/dataset/graph/` and `src/aiperf/graph/` on the
//! `ajc/dynamo-graph-ir` worktree; captions carry file and symbol.

import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";
import { band, card, chip, fanIn, fanOut, link, note } from "../shared/diagram.js";

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "planes",
    eyebrow: "01 · THE SHAPE OF THE SYSTEM",
    title: "Two planes, and only one of them parses",
    lede:
      "The DatasetManager is the sole parser. It writes an mmap store and a content-free sidecar; the TimingManager never sees prompt text at all.",
    narration:
      "Start with the split that everything else depends on. Only one process parses a workload: the dataset manager. It builds a memory-mapped segment store and a sidecar with every prompt stripped out. The timing manager reads only that sidecar, so it never touches content. The worker reads envelopes back out of the store. What holds the three together is a single shared number — the node ordinal, derived from that one parse.",
    caption:
      "dataset/graph/workload_detect.py:16-25 states the two-plane split; store_build.py GraphStoreBuilder; graph_meta_sidecar.py:76 strip_replay_text empties prompts but keeps the `trie` key so routing still works.",
    nodes: [
      band("b-p", "PARSE ONCE", { col: 0, row: 1 }),
      card("ds", "DatasetManager", "the only parser", "control", { col: 0, row: 1 }),
      card("store", "mmap segment store", "content.blob + nodes.blob", "data", { col: 1.3, row: 0 }),
      card("side", "graph_meta sidecar", "prompts stripped", "muted", { col: 1.3, row: 2 }),
      card("tm", "TimingManager", "never parses", "time", { col: 2.6, row: 2 }),
      card("wk", "Worker", "reads envelopes", "done", { col: 2.6, row: 0 }),
      chip("ord", "node ordinal", { col: 1.3, row: 1 }),
      note(
        "inv",
        "One number ties the planes together",
        "Build-time ordinals and dispatch-time catalog ordinals both come from the same parse. That is what makes the worker read the right envelope.",
        { col: 3.7, row: 1 },
      ),
    ],
    edges: [
      ...fanOut("ds", ["store", "side"], "data"),
      link("store", "wk", "data"),
      link("side", "tm", "time"),
      link("ds", "ord", "control", "slow"),
    ],
    revealOrder: ["b-p", "ds", "store", "side", "wk", "tm", "ord", "inv"],
  },
  {
    id: "forest",
    eyebrow: "02 · DYNAMO TRACE",
    title: "A trace is a forest, not a list",
    lede:
      "Sessions link by parent_trajectory_id into trees. Each tree lowers independently, so cross-tree edges vanish by construction.",
    narration:
      "A Dynamo trace arrives as a flat stream of request-end and tool events. The reader folds them into sessions, and sessions link to their parents to form a forest. That forest is partitioned into trees, and here is the useful part: each tree is lowered completely independently, which means cross-tree edges cannot exist by construction rather than by a check. The trees are then merged into one multi-graph workload keyed by reference.",
    caption:
      "adapters/dynamo/trace_reader.py:667 iter_session_records folds records; trace.py:513 group_chains_into_trees and :544 root_of_sessions; :484 _finalize_parsed_graph then merge_parsed_graphs.",
    nodes: [
      band("b-r", "FLAT RECORD STREAM", { col: 0, row: 1.2 }),
      card("rec", "request_end / tool_*", "one JSONL line each", "muted", { col: 0, row: 1.2 }),
      chip("fold", "fold by session", { col: 1.1, row: 1.2 }),

      band("b-f", "SESSION FOREST", { col: 2, row: -0.4 }),
      card("r1", "root session A", "trajectory root", "control", { col: 2, row: -0.4 }),
      card("c1", "child A1", "parent_trajectory_id", "data", { col: 3.1, row: -0.9 }),
      card("c2", "child A2", "parent_trajectory_id", "data", { col: 3.1, row: 0.1 }),
      card("r2", "root session B", "separate tree", "control", { col: 2, row: 1.6 }),
      card("c3", "child B1", "parent_trajectory_id", "data", { col: 3.1, row: 1.6 }),

      card("merge", "multi-graph workload", "graphs[root_id]", "done", { col: 2.55, row: 2.9 }),
      note(
        "iso",
        "Isolation by construction",
        "Each tree lowers on its own, so a cross-tree edge is not rejected — it is unrepresentable.",
        { col: 4.3, row: 0.4 },
      ),
    ],
    edges: [
      link("rec", "fold", "muted"),
      link("fold", "r1", "control", "slow"),
      link("fold", "r2", "control", "slow"),
      ...fanOut("r1", ["c1", "c2"], "data"),
      link("r2", "c3", "data"),
      ...fanIn(["r1", "r2"], "merge", "done", "slow"),
    ],
    revealOrder: ["b-r", "rec", "fold", "b-f", "r1", "c1", "c2", "r2", "c3", "merge", "iso"],
  },
  {
    id: "trie",
    eyebrow: "03 · BUILDING THE WORKLOAD",
    title: "Four stages, with a shortcut through the middle",
    lede:
      "Content parents, then timing warp and edges, then block tags, then per-node emission — which splices the parent's prefix instead of re-decoding it.",
    narration:
      "The workload build runs in four stages. First, resolve which earlier request each one shares a prefix with. Second, warp idle time and derive the timing edges. Third, freeze the block tags. Fourth, emit each node. The fourth stage has a shortcut worth knowing: rather than re-decoding a shared prefix, it reuses the parent's segment-id chain and emits only the fragment that straddles the boundary plus the genuinely new region.",
    caption:
      "segment_ir/trie_content.py:10-38 documents the four stages; :1048 build_trie_ir; prefix splice via bisect_right over msg_end_blocks reusing _EmissionRecord:538; assert_covered_isl:996 gates the assembled token count.",
    nodes: [
      band("b-s", "FOUR STAGES", { col: 0, row: 0.4 }),
      card("s1", "content parents", "prefix automaton", "data", { col: 0, row: 0.4 }),
      card("s2", "warp + edges", "timing", "time", { col: 1.15, row: 0.4 }),
      card("s3", "block tags", "role, new-message", "muted", { col: 2.3, row: 0.4 }),
      card("s4", "emit nodes", "segments per node", "control", { col: 3.45, row: 0.4 }),

      card("splice", "prefix splice", "reuse parent sid chain", "done", { col: 2.3, row: 1.7 }),
      chip("gate", "ISL gate", { col: 3.6, row: 1.7 }),
      note(
        "why",
        "The shortcut is the optimisation",
        "A shared block is decoded once. The child re-emits only the straddling fragment and its new region, and the assembled token count is asserted against the recorded ISL.",
        { col: 0, row: 2.6 },
      ),
    ],
    edges: [
      link("s1", "s2", "data"),
      link("s2", "s3", "time"),
      link("s3", "s4", "muted"),
      link("s1", "splice", "done", "slow"),
      link("splice", "s4", "done"),
      link("splice", "gate", "control"),
    ],
    revealOrder: ["b-s", "s1", "s2", "s3", "s4", "splice", "gate", "why"],
  },
  {
    id: "parents",
    eyebrow: "04 · CONTENT vs TIMING",
    title: "The content parent is not the timing parent",
    lede:
      "Prefix sharing can reach arbitrarily far back. Using that same link for the firing delay would accumulate the whole distance — the aggregate-timestamp bug.",
    narration:
      "These two parent relationships look alike and must never be conflated. The content parent answers: whose prompt prefix do I share? That branch point can be arbitrarily far back in the trace. The timing parent answers: what do I fire after? If you route the firing delay through the content link, the delay becomes the cumulative warped distance back to that ancestor. The code calls this out by name — the aggregate-timestamp bug.",
    caption:
      "trie_content.py:365-373 TrieNode.content_parent — 'CONTENT/PROMPT ONLY'. Timing edges come from interval_order.py:49 build_interval_edges, a separate derivation.",
    nodes: [
      band("b-c", "CONTENT PARENT", { col: 0, row: 0.2 }),
      card("anc", "request 3", "shared prefix owner", "data", { col: 0, row: 0.2 }),
      card("mid1", "request 40", "unrelated", "muted", { col: 1.2, row: 0.2 }),
      card("me1", "request 91", "splices from 3", "data", { col: 2.4, row: 0.2 }),

      band("b-t", "TIMING PARENT", { col: 0, row: 1.9 }),
      card("prev", "request 90", "finished just before", "time", { col: 1.2, row: 1.9 }),
      card("me2", "request 91", "fires after 90", "time", { col: 2.4, row: 1.9 }),

      note(
        "bug",
        "The aggregate-timestamp bug",
        "Route the delay through the content link and it becomes the cumulative warped distance back to an arbitrarily old ancestor.",
        { col: 3.6, row: 1 },
      ),
    ],
    edges: [
      link("anc", "me1", "data", "slow"),
      link("prev", "me2", "time"),
      link("mid1", "me1", "muted", "slow"),
    ],
    revealOrder: ["b-c", "anc", "mid1", "me1", "b-t", "prev", "me2", "bug"],
  },
  {
    id: "frontier",
    eyebrow: "05 · DERIVING EDGES",
    title: "One binding predecessor carries the delay",
    lede:
      "Candidates reduce to a maximal frontier. The latest-ending member binds and carries the whole warped delay; every other frontier edge is an AND-join at zero.",
    narration:
      "Timing edges come from interval order. A request can follow anything that finished before it started, which is far too many edges, so the candidate set is transitively reduced to a frontier. Then one member binds: the one that ends latest. It carries the entire warped delay. Every other frontier edge is still emitted, as an and-join at zero delay, so the node genuinely waits for all of them. If the frontier is empty the node roots at start instead.",
    caption:
      "segment_ir/interval_order.py:49 build_interval_edges — frontier is the transitive reduction; binding predecessor is max(frontier, key=.end); empty frontier roots at START with min_start_delay_us.",
    nodes: [
      band("b-c", "CANDIDATES", { col: 0, row: 0.2 }),
      card("a", "req A", "ends earliest", "muted", { col: 0, row: -0.5 }),
      card("b", "req B", "covered by C", "muted", { col: 0, row: 0.5 }),
      card("c", "req C", "ends latest", "done", { col: 0, row: 1.5 }),

      chip("red", "transitive reduction", { col: 1.25, row: 0.5 }),

      band("b-f", "FRONTIER", { col: 2.4, row: -0.5 }),
      card("fa", "req A", "AND-join, delay 0", "control", { col: 2.4, row: -0.5 }),
      card("fc", "req C", "binding - carries delay", "done", { col: 2.4, row: 1.5 }),
      card("node", "the node", "waits on both", "data", { col: 3.6, row: 0.5 }),
      note(
        "root",
        "Empty frontier roots at START",
        "With nothing before it, the node hangs off START with min_start_delay_us set from its recorded start.",
        { col: 1.1, row: 2.6 },
      ),
    ],
    edges: [
      ...fanIn(["a", "b", "c"], "red", "muted"),
      link("red", "fa", "control"),
      link("red", "fc", "done"),
      link("fa", "node", "control"),
      link("fc", "node", "done"),
    ],
    revealOrder: ["b-c", "a", "b", "c", "red", "b-f", "fa", "fc", "node", "root"],
  },
  {
    id: "warp",
    eyebrow: "06 · IDLE WARP",
    title: "Cap the dead air, never the request",
    lede:
      "The warp collapses gaps between the running-max end and the next start. Capping start-to-start would eat into a long request's own service time.",
    narration:
      "Recorded traces contain long idle stretches, and replaying them verbatim would take as long as the original session. So gaps get capped. The subtlety is which gap. Capping start-to-start distance eats into a long request's own service time and manufactures overlaps that never happened. The warp instead caps only true dead air — from the running maximum end to the next start. The invariant that falls out is that a warped request keeps its exact service time.",
    caption:
      "trie_content.py:193-212 ActiveIdleWarp docstring; invariant warped_end == warped_start + api_time always holds. TrieNode.end:405 adds api_time raw to the warped start.",
    nodes: [
      band("b-n", "NAIVE · START TO START", { col: 0, row: 0.2 }),
      card("n1", "request X", "long service time", "muted", { col: 0, row: 0.2 }),
      card("n2", "capped gap", "eats into X", "failure", { col: 1.15, row: 0.2 }),
      card("n3", "false overlap", "never happened", "failure", { col: 2.3, row: 0.2 }),

      band("b-w", "WARP · ACTIVE INTERVAL", { col: 0, row: 1.8 }),
      card("w1", "request X", "service time intact", "done", { col: 0, row: 1.8 }),
      card("w2", "dead air only", "running-max end -> next start", "time", { col: 1.15, row: 1.8 }),
      card("w3", "next request", "ordering preserved", "done", { col: 2.3, row: 1.8 }),

      note(
        "inv",
        "The invariant",
        "warped_end equals warped_start plus api_time, always. Service time is never compressed.",
        { col: 3.5, row: 1 },
      ),
    ],
    edges: [
      link("n1", "n2", "failure"),
      link("n2", "n3", "failure"),
      link("w1", "w2", "time"),
      link("w2", "w3", "done"),
    ],
    revealOrder: ["b-n", "n1", "n2", "n3", "b-w", "w1", "w2", "w3", "inv"],
  },
  {
    id: "firing",
    eyebrow: "07 · THE EXECUTOR",
    title: "One task per node, four ways out",
    lede:
      "Every node is an asyncio Task in a TaskGroup. Awaiting inputs, then the firing gate, then dispatch — and four distinct exits, only one of them normal.",
    narration:
      "Now the runtime. Each node becomes one asyncio task inside a task group, so the group exit is the join — there is no queue and no gather anywhere in this engine. A node awaits its inputs, sleeps out its firing gate, dispatches, and executes. What is worth studying is the exit. A clean overflow stop ends the trajectory without scheduling successors. A dispatch failure is contained: it writes a type-correct sentinel so downstream fan-in still unblocks. A refusal is re-raised. And the finally block always runs.",
    caption:
      "graph/executor.py:219 _drive_frontier TaskGroup; :278 _fire; :345-424 the containment branch, which re-raises refusal and stickiness before the generic case; sentinel is [] for MESSAGES channels because add_messages rejects non-lists.",
    nodes: [
      band("b-l", "ONE TASK PER NODE", { col: 0, row: 0.9 }),
      card("await", "await inputs", "channel arrivals", "control", { col: 0, row: 0.9 }),
      card("gate", "firing gate", "clock sleep", "time", { col: 1.1, row: 0.9 }),
      card("exec", "execute", "dispatch to worker", "data", { col: 2.2, row: 0.9 }),

      band("b-x", "FOUR EXITS", { col: 3.4, row: -0.8 }),
      card("x1", "normal", "publish + successors", "done", { col: 3.4, row: -0.8 }),
      card("x2", "overflow terminate", "clean stop, no successors", "muted", { col: 3.4, row: 0.3 }),
      card("x3", "dispatch failure", "sentinel write, successors still fire", "time", { col: 3.4, row: 1.4 }),
      card("x4", "refusal / stickiness", "re-raised, trace fatal", "failure", { col: 3.4, row: 2.5 }),

      chip("fin", "finally: finalize", { col: 2.3, row: 2.5 }),
    ],
    edges: [
      link("await", "gate", "control"),
      link("gate", "exec", "time"),
      ...fanOut("exec", ["x1", "x2", "x3", "x4"], "control"),
      link("exec", "fin", "muted", "slow"),
    ],
    revealOrder: ["b-l", "await", "gate", "exec", "b-x", "x1", "x2", "x3", "x4", "fin"],
  },
  {
    id: "dualfan",
    eyebrow: "08 · DUAL FAN-OUT",
    title: "A node fans out twice, at different moments",
    lede:
      "Start-anchored children launch at the parent's dispatch and run alongside it. Completion children launch after. Two separate adjacency maps keep them apart.",
    narration:
      "This is the shape that makes the engine unusual. A node fans out twice. Start-anchored successors are scheduled the instant the parent dispatches, so they run concurrently with a parent that is still streaming. Completion successors are scheduled when it finishes. The scheduler keeps these in two separate maps on purpose — if a start-anchored child were also in the completion map it would be re-scheduled and trip the cycle guard.",
    caption:
      "graph/executor.py:271 schedules start_anchored_successors inside _prepare_node_inputs, at dispatch; :449 _schedule_successors at completion. scheduler.py:93 keeps _start_anchored_succ separate from _static_succ; :104 rejects mixed-anchor fan-in outright.",
    nodes: [
      card("parent", "parent node", "still streaming", "control", { col: 1.2, row: 1 }),
      band("b-d", "AT DISPATCH", { col: 2.5, row: -0.2 }),
      card("sa1", "start-anchored child", "runs alongside", "time", { col: 2.5, row: -0.2 }),
      card("sa2", "start-anchored child", "runs alongside", "time", { col: 2.5, row: 0.8 }),

      band("b-c", "AT COMPLETION", { col: 2.5, row: 2 }),
      card("cc1", "completion child", "after the reply", "done", { col: 2.5, row: 2 }),
      card("cc2", "completion child", "after the reply", "done", { col: 2.5, row: 3 }),

      note(
        "sep",
        "Why two maps",
        "A start-anchored child in the completion map would be scheduled twice and trip the cycle guard. Mixed-anchor fan-in is rejected at construction.",
        { col: 0, row: 2.4 },
      ),
    ],
    edges: [
      ...fanOut("parent", ["sa1", "sa2"], "time"),
      ...fanOut("parent", ["cc1", "cc2"], "done", "slow"),
    ],
    revealOrder: ["parent", "b-d", "sa1", "sa2", "b-c", "cc1", "cc2", "sep"],
  },
  {
    id: "lanes",
    eyebrow: "09 · CONCURRENCY",
    title: "Lanes with a feedback loop, not a queue",
    lede:
      "A fixed pool of lanes each loop: draw a template, run it, release, draw again. Two terminal conditions, and a bare run must not recycle forever.",
    narration:
      "Cross-trace concurrency is a fixed pool of lanes. Each lane loops — draw a template, run it to completion, release the slot, draw the next. The loop is the point: this is genuinely cyclic, not a pipeline. There are two ways it ends. With an explicit stop condition it recycles until that condition trips. Without one, each lane draws until every corpus position has been claimed exactly once, then stops, because a bare graph run must not recycle forever.",
    caption:
      "timing/strategies/graph_ir_replay.py:1259 _run_lanes; :1532 _recycle_has_stop_condition selects bounded vs single-pass; :857 _draw_index — random is coerced to shuffle because each recycle is one corpus pass.",
    nodes: [
      band("b-l", "LANE POOL", { col: 1.3, row: 0.9 }),
      card("corpus", "trace corpus", "N templates", "data", { col: 0, row: 0.9 }),
      card("l0", "lane 0", "draw - run - release", "control", { col: 1.3, row: 0 }),
      card("l1", "lane 1", "draw - run - release", "control", { col: 1.3, row: 0.9 }),
      card("l2", "lane N", "draw - run - release", "control", { col: 1.3, row: 1.8 }),
      chip("draw", "draw_index", { col: 2.6, row: 0.9 }),

      band("b-e", "TWO TERMINAL CONDITIONS", { col: 0, row: 2.8 }),
      card("t1", "stop condition set", "recycle until it trips", "done", { col: 0, row: 2.8 }),
      card("t2", "no stop condition", "single pass, then stop", "muted", { col: 1.4, row: 2.8 }),
      note(
        "loop",
        "The feedback arrow is the design",
        "A released lane re-draws onto its own freed slot, so concurrency stays pinned at the lane count rather than draining.",
        { col: 3, row: 2.4 },
      ),
    ],
    edges: [
      ...fanOut("corpus", ["l0", "l1", "l2"], "data"),
      ...fanIn(["l0", "l1", "l2"], "draw", "control"),
      link("draw", "l1", "time", "slow"),
    ],
    revealOrder: ["b-l", "corpus", "l0", "l1", "l2", "draw", "b-e", "t1", "t2", "loop"],
  },
  {
    id: "demux",
    eyebrow: "10 · RETURNS",
    title: "One observer, de-multiplexed to a parked Future",
    lede:
      "No queue. A return routes observer to adapter to the Future parked for one correlation id and turn, while first tokens travel a parallel track to an Event.",
    narration:
      "Finally, how a reply gets back to the coroutine that is awaiting it. There is no queue. A single observer receives every return, looks up the adapter for that trace, and resolves the future parked under a correlation id and turn index. That wakes exactly the one node coroutine that dispatched it. First tokens travel a completely parallel track — a stamp closure that sets an event, which is what a first-token-anchored successor is waiting on. Two tracks, one fabric.",
    caption:
      "graph/credit_dispatch_adapter.py:265 dispatch parks loop.create_future(); :355 resolve is called synchronously from the credit callback on the same loop, which is why the waiter dict needs no lock. First-token track: dispatch/llm.py:45 _make_first_token_stamp.",
    nodes: [
      card("obs", "graph return observer", "one entry point", "control", { col: 0, row: 1 }),
      chip("lookup", "adapters[trace_id]", { col: 1.15, row: 1 }),
      band("b-w", "PARKED FUTURES", { col: 2.1, row: 0 }),
      card("f1", "future (corr, turn 0)", "awaiting _fire", "data", { col: 2.1, row: 0 }),
      card("f2", "future (corr, turn 1)", "awaiting _fire", "data", { col: 2.1, row: 1 }),
      card("f3", "future (corr, turn 2)", "awaiting _fire", "data", { col: 2.1, row: 2 }),

      band("b-ft", "PARALLEL TRACK", { col: 0, row: 2.7 }),
      card("ft", "first token", "stamp closure", "time", { col: 0, row: 2.7 }),
      card("ev", "Event.set()", "wakes gated successor", "done", { col: 1.3, row: 2.7 }),

      note(
        "lock",
        "No lock on the waiter map",
        "resolve runs synchronously from the credit callback on the same loop, so the dict is thread-confined by construction.",
        { col: 3.4, row: 1.2 },
      ),
    ],
    edges: [
      link("obs", "lookup", "control"),
      ...fanOut("lookup", ["f1", "f2", "f3"], "data"),
      link("ft", "ev", "time"),
    ],
    revealOrder: ["obs", "lookup", "b-w", "f1", "f2", "f3", "b-ft", "ft", "ev", "lock"],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/python-graph-workload`. */
export const PYTHON_GRAPH_WORKLOAD_DECK: DeckDefinition = {
  id: "python-graph-workload",
  title: "Python Graph Workload: trace, build, dataflow",
  slides: SLIDES,
};
