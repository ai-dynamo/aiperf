/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Table } from "../../prose/Table.js";
import { Stat } from "../../prose/Stat.js";
import { Legend } from "../../prose/Legend.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

// Ported from
// ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-weka-ir-v1/canvases/upcoming-async-dataflow.canvas.tsx.
// Single-view canvas — no internal page tabs. Source: docs/reference/graph-async-dataflow-runtime.md
// (reused mechanics) + Step/Emit strategy spec §2/§4C/§12.

type Lane = "dispatch" | "emit" | "core" | "resolve";

// Tailwind's compiler only picks up classes it can see as literal strings in source, so a
// dynamically interpolated `border-l-category-${lane}` would be purged from the production
// build. Keep every lane's className as a whole literal string (mirrors GraphFanInDeck.tsx /
// AiperfGraphEngineDeck.tsx).
const LANE_CLASSES: Record<Lane, string | undefined> = {
  dispatch: "border-l-4 border-l-category-blue",
  emit: "border-l-4 border-l-category-orange",
  resolve: "border-l-4 border-l-category-purple",
  core: undefined,
};

const graphNodes: Node[] = [
  {
    id: "entry",
    type: "panel",
    position: { x: 340, y: 0 },
    data: { title: "Scheduler: entry Steps", detail: "START successors · END suppressed", className: LANE_CLASSES.core },
  },
  {
    id: "gate",
    type: "panel",
    position: { x: 340, y: 110 },
    data: { title: "Await inputs + timing gate", detail: "channel requirements · edge/node delay", className: LANE_CLASSES.core },
  },
  {
    id: "fire",
    type: "panel",
    position: { x: 340, y: 220 },
    data: { title: "Fire Step", detail: "capture causal input snapshot", className: LANE_CLASSES.core },
  },
  {
    id: "dispatch",
    type: "panel",
    position: { x: 20, y: 330 },
    data: { title: "effect: Dispatch", detail: "live · measured · consumes credit", className: LANE_CLASSES.dispatch },
  },
  {
    id: "emit",
    type: "panel",
    position: { x: 700, y: 330 },
    data: { title: "effect: Emit", detail: "canned Duration · no net · no credit", className: LANE_CLASSES.emit },
  },
  {
    id: "adapter",
    type: "panel",
    position: { x: 20, y: 440 },
    data: { title: "CreditDispatchAdapter", detail: "park Future · mint correlation id", className: LANE_CLASSES.dispatch },
  },
  {
    id: "issuer",
    type: "panel",
    position: { x: 20, y: 550 },
    data: { title: "CreditIssuer.issue_graph_credit", detail: "acquire prefill slot · sent counter", className: LANE_CLASSES.dispatch },
  },
  {
    id: "router",
    type: "panel",
    position: { x: 20, y: 660 },
    data: { title: "StickyCreditRouter -> worker", detail: "materialize req from segment pool", className: LANE_CLASSES.dispatch },
  },
  {
    id: "ret",
    type: "panel",
    position: { x: 20, y: 770 },
    data: { title: "Graph return observer", detail: "route by credit.trace_id", className: LANE_CLASSES.dispatch },
  },
  {
    id: "sleep",
    type: "panel",
    position: { x: 700, y: 440 },
    data: { title: "sleep(Duration.micros)", detail: "category: tool | think | network", className: LANE_CLASSES.emit },
  },
  {
    id: "write",
    type: "panel",
    position: { x: 340, y: 880 },
    data: { title: "VersionedChannelStore write", detail: "append log entry · write_seq", className: LANE_CLASSES.core },
  },
  {
    id: "resolve",
    type: "panel",
    position: { x: 340, y: 990 },
    data: { title: "Producer resolves once", detail: "real | FAILED | WILL_NOT_PRODUCE", className: LANE_CLASSES.resolve },
  },
  {
    id: "succ",
    type: "panel",
    position: { x: 340, y: 1100 },
    data: { title: "Schedule successors", detail: "fan-in inputs gate · detached = ungated", className: LANE_CLASSES.core },
  },
];

const graphEdges: Edge[] = [
  { id: "e-entry-gate", source: "entry", target: "gate", type: "flow" },
  { id: "e-gate-fire", source: "gate", target: "fire", type: "flow" },
  { id: "e-fire-dispatch", source: "fire", target: "dispatch", type: "flow" },
  { id: "e-fire-emit", source: "fire", target: "emit", type: "flow" },
  { id: "e-dispatch-adapter", source: "dispatch", target: "adapter", type: "flow" },
  { id: "e-adapter-issuer", source: "adapter", target: "issuer", type: "flow" },
  { id: "e-issuer-router", source: "issuer", target: "router", type: "flow" },
  { id: "e-router-ret", source: "router", target: "ret", type: "flow" },
  { id: "e-ret-write", source: "ret", target: "write", type: "flow" },
  { id: "e-emit-sleep", source: "emit", target: "sleep", type: "flow" },
  { id: "e-sleep-write", source: "sleep", target: "write", type: "flow" },
  { id: "e-write-resolve", source: "write", target: "resolve", type: "flow" },
  { id: "e-resolve-succ", source: "resolve", target: "succ", type: "flow" },
  {
    id: "e-succ-gate",
    source: "succ",
    target: "gate",
    type: "flow",
    label: "frontier loop (back-edge)",
    data: { speed: "slow" },
  },
];

function Header(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Row align="center" gap={10} wrap>
        <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
          Upcoming Async Dataflow — Step/Emit runtime
        </h1>
        <span
          className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
        >
          target
        </span>
      </Row>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        The firing engine is{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>node-kind-agnostic</span> and is
        reused unchanged. The only structural change is the dispatch leaf: today&apos;s{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>singledispatch over 13 NodeKinds</span>{" "}
        collapses to a two-way branch on{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>effect: Dispatch | Emit</span>, plus a
        typed producer-resolution model at the channel store.
      </p>
      <Grid columns={4} gap={12}>
        <Stat value="1" label="Vertex type (Step)" />
        <Stat value="2" label="Effects" tone="neutral" />
        <Stat value="3" label="Producer resolutions" tone="negative" />
        <Stat value="0" label="Firing-loop rewrites" tone="positive" />
      </Grid>
    </Stack>
  );
}

function GraphSection(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>End-to-end firing lifecycle</h2>
      <Row gap={16} wrap align="center">
        <Legend
          entries={[
            { color: "blue", label: "Dispatch lane (server, credit)" },
            { color: "orange", label: "Emit lane (replayed latency)" },
            { color: "purple", label: "Typed resolution" },
            { color: "gray", label: "Reused firing-loop core" },
          ]}
        />
        <Row gap={6} align="center">
          <div className={`h-0 w-4 border-t border-dashed ${strokeClassName("tertiary")}`} />
          <span className={`text-sm ${inkClassName("secondary")}`}>frontier loop (back-edge)</span>
        </Row>
      </Row>
      <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
        <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
            TraceExecutor frontier · one Step firing
          </span>
          <span
            className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
          >
            scrolls
          </span>
        </div>
        <div style={{ height: 640 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={graphNodes}
            edges={graphEdges}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
      </div>
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        Source: docs/reference/graph-async-dataflow-runtime.md (reused mechanics) + Step/Emit strategy
        spec §2/§4C/§12. No central ready queue — readiness is channel waiters, futures, and TaskGroup
        task creation.
      </p>
    </Stack>
  );
}

function EffectSplit(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        The dispatch leaf — the only thing that changes
      </h2>
      <Grid columns={2} gap={16}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>effect: Dispatch</span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              weka + dynamo
            </span>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Builds a <span className={`font-semibold ${inkClassName("primary")}`}>DispatchRequest</span>,
                parks a Future in the{" "}
                <span className={`font-semibold ${inkClassName("primary")}`}>CreditDispatchAdapter</span>,
                awaits the injected credit issuer. Live/measured timing, consumes credit, uses a prefill
                slot, routes through the normal credit router — but bypasses linear session slots.
              </p>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                Bounded by GRAPH.DISPATCH_TIMEOUT_S once it reaches the adapter.
              </p>
            </Stack>
          </div>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>effect: Emit</span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              dynamo only
            </span>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                No network, no credit, no adapter. Sleeps for a typed{" "}
                <span className={`font-semibold ${inkClassName("primary")}`}>
                  Duration&#123;micros, category, source&#125;
                </span>{" "}
                then writes replayed/synthesized content to its output channels.
              </p>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                Where per-tool-type timing scaling (WebFetch 2x, Grep 0.5x) will key on
                metadata.tool_class.
              </p>
            </Stack>
          </div>
        </div>
      </Grid>
      <Callout tone="info" title="Everything below the leaf is unchanged">
        The firing loop, VersionedChannelStore, Scheduler adjacency, edge-delay / t-star gate, fan-in
        dedupe, and the watchdog are all reused. The 11 complicated kinds survive only as{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>metadata kind tags</span>: spawn =
        detached Step, barrier = fan-in gate, await = join, loop = pre-unrolled DAG.
      </Callout>
    </Stack>
  );
}

function TypedResolution(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Typed producer resolution — the new failure model
      </h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        Every producer resolves{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>exactly once</span> into one of three
        states. Consumers gate on the resolution, not the value, so a failed upstream cannot deadlock a
        waiter. Drain is a monotone, forward-only flip — never a reset arc.
      </p>
      <Grid columns={3} gap={16}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Produced a value</span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              real
            </span>
          </div>
          <div className="px-4 py-3">
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Normal write appended to the channel log; the value participates in the reducer.
            </p>
          </div>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Ran, produced nothing</span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              FAILED
            </span>
          </div>
          <div className="px-4 py-3">
            <p className={`text-sm ${inkClassName("secondary")}`}>
              None-sentinel to the output channel; gate-only readers continue. FAILED writers contribute
              nothing to multi-writer joins.
            </p>
          </div>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Never will run</span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              WILL_NOT_PRODUCE
            </span>
          </div>
          <div className="px-4 py-3">
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Untaken branch, or forced by the F-1 gate-timeout watchdog. Waiters whose count is now
              unreachable orphan deterministically.
            </p>
          </div>
        </div>
      </Grid>
      <Grid columns={2} gap={16}>
        <Callout tone="warning" title="F-1 · runtime watchdog is the real backstop">
          The static check is a decidable sound-by-construction subclass (block-structured, acyclic
          post-unroll) —{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>not a proof</span>. The wall-clock{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>EXECUTOR_WATCHDOG_TIMEOUT_S</span>{" "}
          (prototyped this session) forces WILL_NOT_PRODUCE so pre-dispatch deadlocks are bounded.
        </Callout>
        <Callout tone="warning" title="F-3 · relaxed gates need an escape">
          Any <span className={`font-semibold ${inkClassName("primary")}`}>any</span>/
          <span className={`font-semibold ${inkClassName("primary")}`}>quorum</span> gate must resolve
          itself to FAILED/WILL_NOT_PRODUCE once n-k+1 producers resolve non-real, so the gate always
          resolves. Validator rejects relaxed gates lacking it.
        </Callout>
      </Grid>
    </Stack>
  );
}

function ReusedVsChanged(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Reused unchanged vs. changed (§4C / §4D)
      </h2>
      <Table
        columns={[
          { key: "element", label: "Runtime element" },
          { key: "underStepEmit", label: "Under Step/Emit" },
          { key: "status", label: "Status" },
        ]}
        rows={[
          {
            element: "Frontier firing loop",
            underStepEmit: "node-kind-agnostic task creation",
            status: "reused",
            tone: "success",
          },
          {
            element: "VersionedChannelStore",
            underStepEmit: "append log + reducers + producer counts",
            status: "reused",
            tone: "success",
          },
          {
            element: "Scheduler adjacency",
            underStepEmit: "START successors, fan-in dedupe",
            status: "reused",
            tone: "success",
          },
          {
            element: "Edge-delay / t-star gate",
            underStepEmit: "max of edge + node min_start_delay",
            status: "reused",
            tone: "success",
          },
          {
            element: "Watchdog",
            underStepEmit: "EXECUTOR_WATCHDOG_TIMEOUT_S",
            status: "reused (F-1 backstop)",
            tone: "success",
          },
          {
            element: "Dispatch table",
            underStepEmit: "singledispatch(13 kinds) -> branch on effect",
            status: "collapsed",
            tone: "neutral",
          },
          {
            element: "Worker materialize / manifest",
            underStepEmit: "keyed on prompt_segment_ids",
            status: "rewire (mostly renames)",
            tone: "neutral",
          },
          {
            element: "Validation",
            underStepEmit: "cycle / fan-in / no-branch-on-live-output on Step/Edge",
            status: "re-expressed",
            tone: "warning",
          },
        ]}
      />
    </Stack>
  );
}

function Layers(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Concurrency &amp; backpressure layers (reused)
      </h2>
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        Seven independent controls — do not collapse into one concept. All carry over unchanged into the
        Step/Emit runtime.
      </p>
      <Table
        columns={[
          { key: "layer", label: "Layer" },
          { key: "owner", label: "Owner" },
          { key: "bounds", label: "Bounds" },
        ]}
        rows={[
          {
            layer: "Node tasks",
            owner: "TraceExecutor",
            bounds: "in-trace dataflow tasks whose inputs are ready",
          },
          {
            layer: "Trace lanes",
            owner: "GraphIRReplayStrategy",
            bounds: "concurrent trace instances admitted",
          },
          {
            layer: "Graph credit issue",
            owner: "CreditIssuer + stop checker",
            bounds: "whether another graph request may send",
          },
          {
            layer: "Prefill slots",
            owner: "CreditIssuer + callback",
            bounds: "in-flight prefill pressure per request",
          },
          {
            layer: "Adapter waiters",
            owner: "CreditDispatchAdapter",
            bounds: "graph requests awaiting worker returns",
          },
          {
            layer: "Replay barrier",
            owner: "TraceReplayBarrier",
            bounds: "optional cross-stream issue order in a trace",
          },
          {
            layer: "Router load",
            owner: "StickyCreditRouter",
            bounds: "worker choice + in-flight credit load",
          },
        ]}
      />
      <Callout tone="info" title="Not the same thing">
        <span className={`font-semibold ${inkClassName("primary")}`}>TraceReplayBarrier</span> (gates
        credit issue order) is not an IR barrier (a node kind). Analysis cohorts are planning views, not
        runtime synchronization.
      </Callout>
    </Stack>
  );
}

/**
 * Ports `upcoming-async-dataflow.canvas.tsx` (a real, hand-authored Cursor Canvas) onto
 * aiperf-flow's component vocabulary. Single-view canvas — explains the upcoming Step/Emit
 * dataflow runtime: a node-kind-agnostic firing loop reused unchanged, collapsing today's
 * 13-NodeKind `singledispatch` leaf into a two-way `effect: Dispatch | Emit` branch, plus a
 * typed three-state producer-resolution model (`real | FAILED | WILL_NOT_PRODUCE`) at the
 * `VersionedChannelStore`.
 */
export function UpcomingAsyncDataflowDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Upcoming Async Dataflow" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={28}>
            <Header />
            <div className={`border-t ${strokeClassName("secondary")}`} />
            <GraphSection />
            <EffectSplit />
            <TypedResolution />
            <ReusedVsChanged />
            <Layers />
          </Stack>
        </div>
      </div>
    </div>
  );
}
