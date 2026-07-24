/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Execution page: the async dataflow executor and the constructs that keep replay faithful.
//! Ports `ExecutorFiringDemo` (as a React Flow DAG driven by `useStepSimulator`), `ConcurrencyLanes`,
//! `ChannelLogVisual`, `ReducerVisual`, `MaterializationDecision`, `BarrierPolicyVisual`,
//! `LoopAggregatorVisual`, `SpawnLifetimeVisual`, and `BranchResolutionLadder`.

import { useMemo, useState } from "react";
import clsx from "clsx";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Toggle } from "../../prose/Toggle.js";
import { Select } from "../../prose/Select.js";
import { Legend } from "../../prose/Legend.js";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

interface TraceNode {
  id: string;
  kind: string;
  rank: number;
  x: number;
  y: number;
}
const TRACE_NODES: TraceNode[] = [
  { id: "START", kind: "sentinel", rank: 0, x: 120, y: 0 },
  { id: "plan", kind: "llm", rank: 1, x: 120, y: 100 },
  { id: "search", kind: "llm", rank: 2, x: 0, y: 200 },
  { id: "retrieve", kind: "tool", rank: 2, x: 240, y: 200 },
  { id: "draft", kind: "llm", rank: 3, x: 120, y: 300 },
  { id: "review", kind: "llm", rank: 4, x: 120, y: 400 },
  { id: "END", kind: "sentinel", rank: 5, x: 120, y: 500 },
];
const TRACE_EDGES: Edge[] = [
  { id: "e-start-plan", source: "START", target: "plan", type: "flow" },
  { id: "e-plan-search", source: "plan", target: "search", type: "flow" },
  { id: "e-plan-retrieve", source: "plan", target: "retrieve", type: "flow" },
  { id: "e-search-draft", source: "search", target: "draft", type: "flow" },
  { id: "e-retrieve-draft", source: "retrieve", target: "draft", type: "flow" },
  { id: "e-draft-review", source: "draft", target: "review", type: "flow" },
  { id: "e-review-end", source: "review", target: "END", type: "flow" },
];
const TICK_NOTES = [
  "START is the entry sentinel — the frontier begins here.",
  "plan fires first; its only predecessor is START.",
  "search (llm) and retrieve (tool) fire concurrently — both depend only on plan. This is in-trace parallelism.",
  "draft waits for BOTH search and retrieve (fan-in via count) before it fires.",
  "review fires once draft has published its output.",
  "review reaches END — the trace is complete.",
];
const TICKS = [0, 1, 2, 3, 4, 5];

// Top-to-bottom firing DAG. ELK computes the layout once from the (stable-id) structure; the
// per-tick node `data` (firing/done tints) is overlaid onto those positions each render.
const FIRING_LAYOUT: ElkOptions = { direction: "DOWN" };

// The step simulator regenerates `nodes` (new tint/state) every tick under stable ids. `useElkLayout`
// re-lays out only on id/edge changes, so its positions stay fixed; we merge them onto the live-data
// nodes so the animation still shows the frontier advancing.
function FiringCanvas({ nodes, edges }: { nodes: Node[]; edges: Edge[] }): React.JSX.Element {
  const { nodes: laid, laidOut } = useElkLayout(nodes, edges, FIRING_LAYOUT);
  const posById = useMemo(() => new Map(laid.map((n) => [n.id, n.position])), [laid]);
  const positioned = useMemo(
    () => nodes.map((n) => ({ ...n, position: posById.get(n.id) ?? n.position })),
    [nodes, posById],
  );
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={positioned}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.12 }}
      nodesDraggable={false}
      proOptions={{ hideAttribution: true }}
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

function ExecutorFiringDemo(): React.JSX.Element {
  const sim = useStepSimulator(TICKS, { autoPlayMs: 1200 });
  const tick = sim.index;

  const nodes: Node[] = useMemo(
    () =>
      TRACE_NODES.map((n) => {
        const state = n.rank < tick ? "done" : n.rank === tick ? "firing" : "pending";
        const sentinel = n.kind === "sentinel";
        const tint = sentinel ? surfaceClassName("panel") : state === "done" ? categoryBgTintClassName("green") : state === "firing" ? categoryBgTintClassName("blue") : surfaceClassName("elevated");
        return {
          id: n.id,
          type: "card",
          position: { x: n.x, y: n.y },
          data: {
            title: n.id,
            subtitle: sentinel ? undefined : n.kind,
            className: clsx(tint, state === "firing" && "border-l-4"),
            strokeRole: state === "firing" ? "primary" : "secondary",
          },
        };
      }),
    [tick],
  );

  return (
    <Stack gap={12}>
      <Row gap={16} align="start" wrap>
        <div style={{ width: 420, height: 560 }}>
          <ReactFlowProvider>
            <FiringCanvas nodes={nodes} edges={TRACE_EDGES} />
          </ReactFlowProvider>
        </div>
        <Stack gap={12} className="min-w-[240px] flex-1">
          <Row gap={8} align="center">
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>Tick {tick} / 5</span>
            <div className="flex-1" />
            <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>Prev</Button>
            <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>Advance</Button>
            <Button variant="ghost" onClick={sim.reset}>Reset</Button>
          </Row>
          <p className={clsx("text-sm", inkClassName("primary"))}>{TICK_NOTES[Math.min(tick, TICK_NOTES.length - 1)]}</p>
          <Legend
            entries={[
              { color: "gray", label: "pending" },
              { color: "blue", label: "firing" },
              { color: "green", label: "done" },
            ]}
          />
        </Stack>
      </Row>
      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        Illustrative trace · nodes at the same depth fire in parallel · the executor has no central queue — readiness is
        pure channel-input + timing gating.
      </p>
    </Stack>
  );
}

function ConcurrencyLanes(): React.JSX.Element {
  const [lanes, setLanes] = useState("4");
  const n = parseInt(lanes, 10);
  return (
    <Stack gap={12}>
      <Row gap={10} align="center">
        <span className={clsx("text-xs", inkClassName("tertiary"))}>Trace lanes</span>
        <Select value={lanes} onChange={setLanes} options={[{ value: "2", label: "2 lanes" }, { value: "4", label: "4 lanes" }, { value: "6", label: "6 lanes" }]} />
        <div className="flex-1" />
        <span className={clsx("text-xs", inkClassName("tertiary"))}>each lane runs one trace instance with its own ready-node fan-out</span>
      </Row>
      <Stack gap={6}>
        {Array.from({ length: n }).map((_, li) => (
          <Row key={li} gap={8} align="center">
            <div className={clsx("w-16 shrink-0 text-xs", inkClassName("tertiary"))}>lane {li + 1}</div>
            <Row gap={6} align="center">
              {[0, 1, 2, 3].map((d) => (
                <div key={d} className={clsx("h-3.5 w-3.5 rounded-md border shadow-sm", strokeClassName("secondary"), d === 1 || d === 2 ? categoryBgTintClassName("blue") : surfaceClassName("elevated"))} />
              ))}
              <span className={clsx("ml-1 text-xs", inkClassName("tertiary"))}>2 nodes ready</span>
            </Row>
          </Row>
        ))}
      </Stack>
      <Callout tone="info">
        Total in-flight graph work ≈ <strong>lanes × ready nodes per trace</strong>. With {n} lanes and ~2 ready nodes
        each, roughly {n * 2} node dispatches are live at once — size both dials when your graph fans out.
      </Callout>
    </Stack>
  );
}

function ChannelLog(): React.JSX.Element {
  const [count, setCount] = useState("all");
  const writes = [
    { seq: 0, who: "initial_state", seed: true },
    { seq: 1, who: "node A", seed: false },
    { seq: 2, who: "node B", seed: false },
    { seq: 3, who: "node C", seed: false },
  ];
  const declared = 3;
  const target = count === "all" ? declared : parseInt(count, 10);
  return (
    <Stack gap={10}>
      <Row gap={6} align="center" wrap>
        <span className={clsx("text-xs", inkClassName("tertiary"))}>ChannelRequirement.count =</span>
        {["1", "2", "all"].map((c) => (
          <button key={c} type="button" aria-pressed={count === c} onClick={() => setCount(c)} className={clsx("rounded-md border px-2.5 py-0.5 text-xs font-medium shadow-sm", strokeClassName(count === c ? "primary" : "secondary"), count === c ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}>{c}</button>
        ))}
      </Row>
      <Row gap={8} wrap>
        {writes.map((w) => {
          const on = !w.seed && w.seq <= target;
          return (
            <div key={w.seq} className="flex flex-col items-center gap-1">
              <div className={clsx("min-w-[88px] rounded-md border px-3 py-2 text-center text-xs font-semibold shadow-sm", strokeClassName("secondary"), w.seed ? clsx(surfaceClassName("panel"), inkClassName("primary")) : on ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("primary")))}>{w.who}</div>
              <span className={clsx("text-xs", inkClassName("tertiary"))}>seq {w.seq}{w.seed ? " · seed" : ""}</span>
            </div>
          );
        })}
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        The seed at <Code inline>seq 0</Code> feeds the reducer but does <strong>not</strong> count as a producer
        arrival. <Code inline>count=N</Code> captures the first N node writes; <Code inline>count=all</Code> resolves to
        the {declared} statically declared producers.
      </p>
    </Stack>
  );
}

const REDUCERS: Record<string, { in: string[]; out: string; note: string }> = {
  overwrite: { in: ["value X", "value Y"], out: "Y", note: "One writer value wins; multiple writers to an overwrite channel are a conflict." },
  add_messages: { in: ["[m1]", "[m2]"], out: "[m1, m2]", note: "Appends message lists; a later message with the same id replaces the earlier one." },
  stream_append: { in: ["chunk₁", "chunk₂"], out: "chunk₁chunk₂", note: "Append-only streaming channel; many writers may append chunks until close." },
};

function ReducerVisual(): React.JSX.Element {
  const [reducer, setReducer] = useState("add_messages");
  const r = REDUCERS[reducer]!;
  return (
    <Stack gap={10}>
      <Row gap={6} wrap>
        {Object.keys(REDUCERS).map((k) => (
          <button key={k} type="button" aria-pressed={reducer === k} onClick={() => setReducer(k)} className={clsx("rounded-md border px-3 py-1 text-xs font-medium shadow-sm", strokeClassName(reducer === k ? "primary" : "secondary"), reducer === k ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}>{k}</button>
        ))}
      </Row>
      <Row gap={10} align="center" wrap>
        <Row gap={6}>
          {r.in.map((v, i) => (
            <div key={i} className={clsx("rounded-md border px-2.5 py-1.5 text-xs shadow-sm", strokeClassName("secondary"), surfaceClassName("panel"), inkClassName("primary"))}>{v}</div>
          ))}
        </Row>
        <span className={inkClassName("tertiary")}>→</span>
        <div className={clsx("rounded-md px-3 py-1.5 text-xs font-semibold shadow-sm", categoryBgTintClassName("green"), inkClassName("primary"))}>{r.out}</div>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>{r.note}</p>
    </Stack>
  );
}

const MAT_BRANCHES = [
  { id: "uni-bytes", title: "Unified A2 · bytes", fn: "materialize_graph_request_unified_bytes", desc: "Interned unified store + no cache-bust → a pre-serialized body is built once from int-handle content slices and sent with zero decode/encode." },
  { id: "uni-dict", title: "Unified A2 · dict", fn: "materialize_graph_request_unified", desc: "Interned unified store + cache-bust → rebuild the messages dict from handles, layer run-level options, then stamp the per-instance cache-bust marker on the first user message." },
  { id: "delta-dict", title: "GRAPH_DELTA · dict", fn: "materialize_graph_request", desc: "No unified store (a native delta-mmap build) → accumulate the node's ancestor-path messages_delta from the GRAPH_DELTA mmap, then apply dispatch overrides." },
];

function MaterializationDecision(): React.JSX.Element {
  const [unified, setUnified] = useState(true);
  const [cacheBust, setCacheBust] = useState(false);
  const activeId = unified && !cacheBust ? "uni-bytes" : unified && cacheBust ? "uni-dict" : "delta-dict";
  return (
    <Stack gap={12}>
      <Row gap={16} wrap align="center">
        <Row gap={8} align="center">
          <Toggle checked={unified} onChange={setUnified} />
          <span className={clsx("text-sm", inkClassName("secondary"))}>Unified store present (trie build)</span>
        </Row>
        <Row gap={8} align="center">
          <Toggle checked={cacheBust} onChange={setCacheBust} />
          <span className={clsx("text-sm", inkClassName("secondary"))}>Cache-busting on</span>
        </Row>
      </Row>
      <Grid columns={3} gap={12}>
        {MAT_BRANCHES.map((b) => {
          const on = b.id === activeId;
          return (
            <div key={b.id} className={clsx("rounded-lg border px-4 py-3 shadow-sm", strokeClassName(on ? "primary" : "secondary"), on ? categoryBgTintClassName("blue") : surfaceClassName("elevated"))}>
              <Row align="center" justify="space-between">
                <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{b.title}</span>
                {on && <span className={clsx("rounded-md border px-1.5 py-0.5 text-[10px] font-medium shadow-sm", strokeClassName("primary"), inkClassName("primary"))}>selected</span>}
              </Row>
              <div className="mt-1"><Code inline>{b.fn}</Code></div>
              <p className={clsx("mt-1 text-xs", inkClassName("secondary"))}>{b.desc}</p>
            </div>
          );
        })}
      </Grid>
      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        The interned A2 unified store is the sole trie store shape — every weka and dynamo trie build writes it. A
        pre-serialized body cannot be mutated, so any cache-busting forces the dict path; only native delta-mmap builds
        (no unified store) take the GRAPH_DELTA ancestor path.
      </p>
    </Stack>
  );
}

type BarrierPolicy = "all" | "any" | "quorum" | "timeout";
const BARRIER_DETAIL: Record<BarrierPolicy, string> = {
  all: "policy='all', no timeout: the standard input gate already proved every predecessor wrote, so the barrier is pass-through — it just stamps reason='all' and emits outputs. Nothing is cancelled.",
  any: "policy='any': threshold is 1. The first arrival closes the barrier; every still-running loser is race-cancelled via task.cancel(); already-done siblings simply go uncounted.",
  quorum: "policy='quorum': threshold is the validated quorum_count (here 2). Once 2 predecessors complete the barrier closes and cancels the rest. Cancelled/errored predecessors still count toward the threshold.",
  timeout: "policy='all' WITH timeout_ms: the barrier races predecessor task handles and returns reason='timeout' if the deadline fires before all arrive. Losers are cancelled; the closure reason is published to each output channel.",
};

function BarrierPolicyVisual(): React.JSX.Element {
  const [policy, setPolicy] = useState<BarrierPolicy>("quorum");
  const preds = [
    { id: "p1", done: true },
    { id: "p2", done: true },
    { id: "p3", done: false },
    { id: "p4", done: false },
  ];
  const relaxed = policy !== "all";
  const closedCount = policy === "any" ? 1 : policy === "quorum" ? 2 : policy === "timeout" ? 2 : preds.length;
  return (
    <Stack gap={12}>
      <Row gap={6} wrap align="center">
        <span className={clsx("text-xs", inkClassName("tertiary"))}>BarrierNode.policy =</span>
        {(["all", "any", "quorum", "timeout"] as BarrierPolicy[]).map((p) => (
          <button key={p} type="button" aria-pressed={policy === p} onClick={() => setPolicy(p)} className={clsx("rounded-md border px-3 py-1 text-xs font-medium shadow-sm", strokeClassName(policy === p ? "primary" : "secondary"), policy === p ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}>{p === "timeout" ? "all + timeout" : p}</button>
        ))}
      </Row>
      <Row gap={10} wrap>
        {preds.map((p, i) => {
          const counted = i < closedCount;
          const state = counted ? "counted" : p.done ? "uncounted" : relaxed ? "cancelled" : "waiting";
          const color: CategoryRole | null = state === "counted" ? "green" : state === "cancelled" ? "orange" : null;
          return (
            <div key={p.id} className="flex flex-col items-center gap-1">
              <div className={clsx("min-w-[56px] rounded-md border px-4 py-2 text-center text-xs font-semibold shadow-sm", strokeClassName("secondary"), color ? clsx(categoryBgTintClassName(color), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("primary")))}>{p.id}</div>
              <span className={clsx("text-xs", inkClassName("tertiary"))}>{state}</span>
            </div>
          );
        })}
      </Row>
      <Callout tone={policy === "all" ? "info" : "warning"} title={`closure reason = "${policy === "timeout" ? "timeout" : policy}"`}>
        {BARRIER_DETAIL[policy]}
      </Callout>
    </Stack>
  );
}

type LoopAgg = "last" | "list" | "concat";
const LOOP_NOTE: Record<LoopAgg, string> = {
  last: "aggregator='last' returns only the final iteration's body write (UNSET → None). One value per output channel.",
  list: "aggregator='list' returns one Write per output channel whose value is the per-iteration list.",
  concat: "aggregator='concat' folds the per-iteration values through the channel's typed reducer — string concat for overwrite/text, list extend for add_messages, one Write per chunk for streaming reducers.",
};

function LoopAggregatorVisual(): React.JSX.Element {
  const [agg, setAgg] = useState<LoopAgg>("concat");
  const iters = ["A", "B", "C"];
  const out = agg === "last" ? "C" : agg === "list" ? '["A", "B", "C"]' : '"ABC"';
  return (
    <Stack gap={12}>
      <Row gap={6} wrap>
        {(["last", "list", "concat"] as LoopAgg[]).map((a) => (
          <button key={a} type="button" aria-pressed={agg === a} onClick={() => setAgg(a)} className={clsx("rounded-md border px-3 py-1 text-xs font-medium shadow-sm", strokeClassName(agg === a ? "primary" : "secondary"), agg === a ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}>{a}</button>
        ))}
      </Row>
      <Row gap={10} align="center" wrap>
        <Row gap={6}>
          {iters.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <div className={clsx("rounded-md border px-3 py-1.5 text-xs font-semibold shadow-sm", strokeClassName("secondary"), surfaceClassName("panel"), inkClassName("primary"))}>{v}</div>
              <span className={clsx("text-xs", inkClassName("tertiary"))}>iter {i + 1}</span>
            </div>
          ))}
        </Row>
        <span className={inkClassName("tertiary")}>→</span>
        <div className={clsx("rounded-md px-3.5 py-2 text-xs font-bold shadow-sm", categoryBgTintClassName("green"), inkClassName("primary"))}>{out}</div>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>{LOOP_NOTE[agg]}</p>
      <Callout tone="info">
        Each iteration runs the body in a fresh child <Code inline>TraceExecutor</Code> on a fresh{" "}
        <Code inline>TraceRecord</Code> (no state leaks). A counted loop runs exactly <Code inline>max_iterations</Code>;
        a conditional loop with <Code inline>break_on_channel</Code> that never fires the break raises{" "}
        <Code inline>LoopMaxIterationsError</Code>.
      </Callout>
    </Stack>
  );
}

function SpawnLifetime(): React.JSX.Element {
  const [cascade, setCascade] = useState(true);
  return (
    <Stack gap={12}>
      <Row align="center" gap={10}>
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>SpawnNode.cancel_with_parent = {String(cascade)}</span>
        <div className="flex-1" />
        <span className={clsx("text-xs", inkClassName("tertiary"))}>cancel with parent</span>
        <Toggle checked={cascade} onChange={setCascade} />
      </Row>
      <Grid columns={2} gap={12}>
        <div className={clsx("rounded-lg border shadow-sm", strokeClassName("secondary"))}>
          <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>Owning TaskGroup</span>
            <Code inline>{cascade ? "ctx.tg" : "ctx.phase_tg"}</Code>
          </div>
          <p className={clsx("px-4 py-3 text-sm", inkClassName("secondary"))}>
            The child task is created on <Code inline>{cascade ? "ctx.tg" : "ctx.phase_tg"}</Code>.{" "}
            {cascade ? "It lives inside the per-trace TaskGroup, so a parent unwind cascades cancellation to the child." : "It lives on the phase-scoped TaskGroup, so the child outlives its parent trace and is only cancelled at phase teardown."}
          </p>
        </div>
        <div className={clsx("rounded-lg border shadow-sm", strokeClassName("secondary"))}>
          <div className={clsx("flex items-center justify-between border-b px-4 py-2", strokeClassName("secondary"))}>
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>Await + timeout</span>
            <Code inline>AwaitNode</Code>
          </div>
          <p className={clsx("px-4 py-3 text-sm", inkClassName("secondary"))}>
            An <Code inline>AwaitNode</Code> reads the <Code inline>SpawnHandle</Code> and awaits it under{" "}
            <Code inline>asyncio.wait_for(asyncio.shield(task), …)</Code>. On timeout it writes a{" "}
            <Code inline>{"<node::timeout>"}</Code> marker and raises <Code inline>_NodeExpectedExit</Code>; the shielded
            spawn keeps running unless it is a <Code inline>cancel_with_parent</Code> child (then it is cancelled to
            avoid a TaskGroup deadlock).
          </p>
        </div>
      </Grid>
      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        SpawnNode returns immediately with a <Code inline>SpawnHandle</Code> on its <Code inline>handle_channel</Code>;
        the child id is minted by <Code inline>make_spawn_child_id</Code> so loop-body and nested spawns never collide.
      </p>
    </Stack>
  );
}

function BranchResolutionLadder(): React.JSX.Element {
  const [explicit, setExplicit] = useState(false);
  const [perTrace, setPerTrace] = useState(false);
  const [mixTag, setMixTag] = useState(true);
  const [edgeDefault, setEdgeDefault] = useState(true);
  const steps = [
    { on: explicit, label: "trace.selected_branches[source]", sub: "explicit per-trace pick" },
    { on: perTrace, label: "trace.branch_distributions[source]", sub: "per-trace sampler" },
    { on: mixTag, label: "mix_record.branch_weights[tag][source]", sub: "first matching trace tag" },
    { on: edgeDefault, label: "edge.branch_weights", sub: "graph-level default sampler" },
    { on: true, label: "None", sub: "no resolution — trace ends here" },
  ];
  const winner = steps.findIndex((s) => s.on);
  return (
    <Stack gap={12}>
      <Row gap={16} wrap>
        {[
          { c: explicit, set: setExplicit, l: "selected_branches" },
          { c: perTrace, set: setPerTrace, l: "branch_distributions" },
          { c: mixTag, set: setMixTag, l: "mix tag weights" },
          { c: edgeDefault, set: setEdgeDefault, l: "edge default" },
        ].map((t) => (
          <Row key={t.l} gap={8} align="center">
            <Toggle checked={t.c} onChange={t.set} />
            <span className={clsx("text-sm", inkClassName("secondary"))}>{t.l}</span>
          </Row>
        ))}
      </Row>
      <Stack gap={6}>
        {steps.map((s, i) => {
          const isWinner = i === winner;
          const skipped = i < winner;
          return (
            <Row key={i} gap={10} align="center">
              <div className={clsx("flex h-6 w-6 shrink-0 items-center justify-center rounded-md border text-[11px] font-bold shadow-sm", strokeClassName("secondary"), isWinner ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("tertiary")))}>{i + 1}</div>
              <div>
                <Code inline>{s.label}</Code>
                <div className={clsx("text-xs", skipped ? clsx(inkClassName("quaternary"), "line-through") : inkClassName("tertiary"))}>
                  {s.sub}{isWinner ? " · resolves" : skipped ? " · skipped (higher layer won)" : ""}
                </div>
              </div>
            </Row>
          );
        })}
      </Stack>
      <Callout tone="info">
        The three sampler layers (2/3/4) share one SHA-256 seed derivation over{" "}
        <Code inline>(workload_seed, trace_id, edge.source)</Code>, so a given fork re-runs byte-identically across
        processes regardless of which layer supplied the weights — independent of <Code inline>PYTHONHASHSEED</Code>.
      </Callout>
    </Stack>
  );
}

export function ExecutionPage(): React.JSX.Element {
  return (
    <Stack gap={20}>
      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Walkthrough: a trace executing</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Advance the tick to watch the frontier move. Nodes at the same depth fire concurrently; fan-in nodes wait for
          every producer.
        </p>
        <ExecutorFiringDemo />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Two concurrency dials, live</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Cross-trace lanes and in-trace ready-node fan-out multiply. Change the lane count to see how in-flight work
          scales.
        </p>
        <ConcurrencyLanes />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Channel dataflow: versioned writes &amp; fan-in</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Every node write appends a versioned log entry. A node&apos;s <Code inline>count</Code> requirement decides
          how many arrivals it waits for. Change it to see which writes are captured.
        </p>
        <ChannelLog />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Reducers: how writes merge</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A channel&apos;s reducer decides how concurrent writes combine into the value a reader sees.
        </p>
        <ReducerVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Walkthrough: which materialization path?</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A worker picks one of four rebuild paths per credit. Flip the switches to see which branch a given store +
          endpoint config selects.
        </p>
        <MaterializationDecision />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Barrier join policies</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A <Code inline>BarrierNode</Code> synchronizes predecessors under an all/any/quorum policy, with an optional
          timeout. Relaxed policies race the predecessor tasks and cancel the losers.
        </p>
        <BarrierPolicyVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Loop aggregation</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A <Code inline>LoopNode</Code> re-runs its body subgraph and folds the per-iteration writes. Switch the
          aggregator to see how the parent-visible value is built.
        </p>
        <LoopAggregatorVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Spawn &amp; await lifetime</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A <Code inline>SpawnNode</Code> detaches a child executor and returns a handle immediately. Whether the child
          dies with its parent depends on which TaskGroup owns it.
        </p>
        <SpawnLifetime />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Conditional branch resolution</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          When an edge forks, the runtime walks a five-layer ladder to pick a branch. Toggle which layers are present to
          see which one wins.
        </p>
        <BranchResolutionLadder />
      </Stack>
    </Stack>
  );
}
