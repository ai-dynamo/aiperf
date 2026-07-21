/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Stat } from "../../prose/Stat.js";
import { Table, type TableRowTone } from "../../prose/Table.js";
import { Legend } from "../../prose/Legend.js";
import {
  surfaceClassName,
  strokeClassName,
  inkClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

// Ported from ~/.cursor/projects/.../canvases/aiperf-graph-engine.canvas.tsx
// (AIPerf v2 — Async-Dataflow Graph Engine). Single-view canvas: no page tabs
// in the source, so this ports as one component file. Source of truth for the
// runtime it explains: src/aiperf/graph/** + src/aiperf/dataset/loader/graph/**.

type NodeGroup = "io" | "content" | "control" | "terminal";

interface GNode {
  id: string;
  label: string;
  group: NodeGroup;
  sub: string;
  file?: string;
}

interface GEdge {
  from: string;
  to: string;
  label?: string;
  style: "static" | "spawn" | "stream" | "join";
}

// A representative single-trace agentic graph exercising the node taxonomy: the
// planner fans out into a streamed tool path AND a detached sub-agent, then both
// arms re-join at context compaction before the terminal turn.
const GRAPH_NODES: GNode[] = [
  { id: "START", label: "START", group: "terminal", sub: "entry" },
  {
    id: "bootstrap",
    label: "bootstrap",
    group: "content",
    sub: "BootstrapNode",
    file: "src/aiperf/graph/dispatch/replay.py",
  },
  {
    id: "planner",
    label: "planner",
    group: "io",
    sub: "LlmNode · chat",
    file: "src/aiperf/graph/dispatch/llm.py",
  },
  {
    id: "toolcall",
    label: "parse tool call",
    group: "content",
    sub: "ToolCallNode",
    file: "src/aiperf/graph/dispatch/replay.py",
  },
  {
    id: "tool",
    label: "exec tool",
    group: "io",
    sub: "ToolNode · tool:*",
    file: "src/aiperf/graph/dispatch/tool.py",
  },
  {
    id: "toolresult",
    label: "splice result",
    group: "content",
    sub: "ToolResultNode",
    file: "src/aiperf/graph/dispatch/replay.py",
  },
  {
    id: "critic",
    label: "spawn critic",
    group: "control",
    sub: "SpawnNode",
    file: "src/aiperf/graph/dispatch/spawn.py",
  },
  {
    id: "await_critic",
    label: "join critic",
    group: "control",
    sub: "AwaitNode",
    file: "src/aiperf/graph/dispatch/await_node.py",
  },
  {
    id: "compact",
    label: "compact ctx",
    group: "content",
    sub: "CompactNode",
    file: "src/aiperf/graph/dispatch/replay.py",
  },
  {
    id: "final",
    label: "final answer",
    group: "io",
    sub: "LlmNode · terminal_for_user",
    file: "src/aiperf/graph/dispatch/llm.py",
  },
  { id: "END", label: "END", group: "terminal", sub: "exit" },
];

const GRAPH_EDGES: GEdge[] = [
  { from: "START", to: "bootstrap", style: "static" },
  { from: "bootstrap", to: "planner", style: "static", label: "@messages" },
  { from: "planner", to: "toolcall", style: "stream", label: "tool_call_stream" },
  { from: "planner", to: "critic", style: "spawn", label: "spawn (fresh ctx)" },
  { from: "toolcall", to: "tool", style: "static" },
  { from: "tool", to: "toolresult", style: "static" },
  { from: "toolresult", to: "compact", style: "static", label: "@messages" },
  { from: "critic", to: "await_critic", style: "spawn", label: "SpawnHandle" },
  { from: "await_critic", to: "compact", style: "join", label: "count=all" },
  { from: "compact", to: "final", style: "static" },
  { from: "final", to: "END", style: "static" },
];

// Hand-placed positions: main spine at x=140, spawned-critic arm at x=440,
// converging back at "compact". Mirrors the source canvas's vertical DAG layout.
const GRAPH_POSITIONS: Record<string, { x: number; y: number }> = {
  START: { x: 140, y: 0 },
  bootstrap: { x: 140, y: 90 },
  planner: { x: 140, y: 190 },
  toolcall: { x: 0, y: 300 },
  critic: { x: 440, y: 300 },
  tool: { x: 0, y: 400 },
  await_critic: { x: 440, y: 400 },
  toolresult: { x: 0, y: 500 },
  compact: { x: 140, y: 610 },
  final: { x: 140, y: 710 },
  END: { x: 140, y: 800 },
};

// Every possible group -> Tailwind class, kept literal so Tailwind's JIT scanner sees
// them (see aiperf-flow-diagrams SKILL.md "The Tailwind JIT trap").
const GROUP_CATEGORY: Record<NodeGroup, CategoryRole> = {
  io: "blue",
  content: "green",
  control: "yellow",
  terminal: "gray",
};

const CATEGORY_BORDER_L_CLASSES: Record<CategoryRole, string> = {
  green: "border-l-category-green",
  yellow: "border-l-category-yellow",
  purple: "border-l-category-purple",
  blue: "border-l-category-blue",
  red: "border-l-category-red",
  orange: "border-l-category-orange",
  cyan: "border-l-category-cyan",
  gray: "border-l-category-gray",
};

const CATEGORY_CSS_VAR: Record<CategoryRole, string> = {
  green: "var(--color-category-green)",
  yellow: "var(--color-category-yellow)",
  purple: "var(--color-category-purple)",
  blue: "var(--color-category-blue)",
  red: "var(--color-category-red)",
  orange: "var(--color-category-orange)",
  cyan: "var(--color-category-cyan)",
  gray: "var(--color-category-gray)",
};

const EDGE_STYLE_CATEGORY: Record<GEdge["style"], CategoryRole> = {
  static: "gray",
  spawn: "yellow",
  stream: "blue",
  join: "purple",
};

const flowNodes: Node[] = GRAPH_NODES.map((n) => ({
  id: n.id,
  type: "card",
  position: GRAPH_POSITIONS[n.id],
  data: {
    title: n.label,
    subtitle: n.group === "terminal" ? undefined : n.sub,
    className: clsx(
      "border-l-4",
      CATEGORY_BORDER_L_CLASSES[GROUP_CATEGORY[n.group]],
      n.group === "terminal" && "min-w-[100px] text-center",
    ),
  },
}));

const flowEdges: Edge[] = GRAPH_EDGES.map((e) => ({
  id: `e-${e.from}-${e.to}`,
  source: e.from,
  target: e.to,
  label: e.label,
  ...(e.style === "static"
    ? {}
    : {
        type: "flow" as const,
        data: { color: CATEGORY_CSS_VAR[EDGE_STYLE_CATEGORY[e.style]], speed: "normal" as const },
      }),
}));

const NODE_BY_ID = new Map(GRAPH_NODES.map((n) => [n.id, n]));

// All 14 dispatch-registered node kinds + edge handling.
const NODE_KINDS: {
  kind: string;
  role: string;
  module: string;
  span: string;
  group: NodeGroup;
}[] = [
  {
    kind: "LlmNode",
    role: "Dispatch one LLM call → one credit, one request, one record. Streaming path fans out parsed tool-calls mid-stream.",
    module: "dispatch/llm.py",
    span: "CLIENT · chat",
    group: "io",
  },
  {
    kind: "ToolNode",
    role: "Execute an external tool call as real wire traffic.",
    module: "dispatch/tool.py",
    span: "CLIENT · tool:<tool>",
    group: "io",
  },
  {
    kind: "SpawnNode",
    role: "Detach a sub-agent child trace with fresh context; may outlive the parent (phase_tg).",
    module: "dispatch/spawn.py",
    span: "INTERNAL · spawn:<ref>",
    group: "control",
  },
  {
    kind: "AwaitNode",
    role: "Join on a SpawnHandle; gate the consumer turn until the child drains.",
    module: "dispatch/await_node.py",
    span: "INTERNAL · await:<id>",
    group: "control",
  },
  {
    kind: "SubgraphNode",
    role: "Invoke a nested ParsedGraph via a child TraceExecutor.",
    module: "dispatch/subgraph.py",
    span: "INTERNAL · subgraph:<ref>",
    group: "control",
  },
  {
    kind: "LoopNode",
    role: "Iterate a body subgraph internally; exempt from the cycle guard.",
    module: "dispatch/loop.py",
    span: "INTERNAL · loop:<id>",
    group: "control",
  },
  {
    kind: "BarrierNode",
    role: "Relaxed policy / timeout races predecessor handles and cancels losers; plain all+no-timeout uses replay semantics.",
    module: "dispatch/barrier.py",
    span: "INTERNAL · barrier:<policy>",
    group: "control",
  },
  {
    kind: "ReplayNode",
    role: "Splice a recorded message delta; may overwrite an accumulating channel.",
    module: "dispatch/replay.py",
    span: "INTERNAL",
    group: "content",
  },
  {
    kind: "ToolCallNode",
    role: "Parse / emit a structured tool call into the tool_call channel.",
    module: "dispatch/replay.py",
    span: "INTERNAL · tool_call",
    group: "content",
  },
  {
    kind: "ToolResultNode",
    role: "Splice a tool result back into the message accumulator.",
    module: "dispatch/replay.py",
    span: "INTERNAL",
    group: "content",
  },
  {
    kind: "CompactNode",
    role: "Context compaction / reset turn (delta becomes the new channel base).",
    module: "dispatch/replay.py",
    span: "INTERNAL",
    group: "content",
  },
  {
    kind: "BootstrapNode",
    role: "Seed the initial state / accumulator at trace entry.",
    module: "dispatch/replay.py",
    span: "INTERNAL",
    group: "content",
  },
  {
    kind: "DelayNode",
    role: "Idle / pacing delay between turns (replay-faithful edge timing).",
    module: "dispatch/replay.py",
    span: "INTERNAL",
    group: "content",
  },
  {
    kind: "edges",
    role: "START/END handling, ConditionalEdge branch resolution, terminal_for_user accounting.",
    module: "dispatch/edges.py",
    span: "—",
    group: "terminal",
  },
];

const CHANNEL_TYPES = [
  "text",
  "image",
  "audio",
  "video",
  "json",
  "messages",
  "tool_calls",
  "tool_call_stream",
];

const REDUCERS = [
  { name: "overwrite", note: "last write wins (default)" },
  { name: "add_messages", note: "append to the accumulator" },
  { name: "stream_append", note: "append one element per write" },
  { name: "stream_passthrough", note: "forward stream elements as-is" },
];

// Table only knows neutral/success/warning/danger (no direct "info" tone), so the
// node-kind group -> row tone mapping is domain-chosen: io stays neutral (already
// distinguished by the Handler/Span columns), content -> success, control -> warning.
const NODE_KIND_ROW_TONE: Record<NodeGroup, TableRowTone> = {
  io: "neutral",
  content: "success",
  control: "warning",
  terminal: "neutral",
};

function PipelineStage({
  step,
  title,
  service,
  body,
  files,
}: {
  step: string;
  title: string;
  service: string;
  body: string;
  files: string[];
}): React.JSX.Element {
  return (
    <div
      className={clsx(
        "flex flex-col gap-2 rounded-lg border px-4 py-3 shadow-sm",
        surfaceClassName("elevated"),
        strokeClassName("secondary"),
      )}
    >
      <Row justify="space-between" align="center">
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>
          {step} · {title}
        </span>
        <span className={clsx("text-xs font-medium", inkClassName("tertiary"))}>{service}</span>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>{body}</p>
      <Row gap={6} wrap>
        {files.map((f) => (
          <Code key={f} inline>
            {f}
          </Code>
        ))}
      </Row>
    </div>
  );
}

function Collaborator({
  name,
  role,
  file,
}: {
  name: string;
  role: string;
  file: string;
}): React.JSX.Element {
  return (
    <div
      className={clsx(
        "flex flex-col gap-2 rounded-lg border px-4 py-3 shadow-sm",
        surfaceClassName("elevated"),
        strokeClassName("secondary"),
      )}
    >
      <Row justify="space-between" align="center">
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{name}</span>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>{role}</p>
      <Code inline>{file}</Code>
    </div>
  );
}

function GraphDiagram(): React.JSX.Element {
  const [selectedId, setSelectedId] = useState<string | null>("planner");
  const selectedNode = selectedId ? NODE_BY_ID.get(selectedId) : undefined;
  const selectedKind = selectedNode
    ? NODE_KINDS.find((k) => selectedNode.sub.startsWith(k.kind))
    : undefined;

  return (
    <Stack gap={12}>
      <Legend
        entries={[
          { color: "blue", label: "Wire I/O (credit · request · record)" },
          { color: "green", label: "Content / replay (no wire traffic)" },
          { color: "yellow", label: "Control flow (spawn · await · loop · barrier)" },
          { color: "gray", label: "START / END" },
        ]}
      />
      <Row gap={16} align="start">
        <div style={{ height: 620, flex: "1 1 auto", minWidth: 0 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={flowNodes}
            edges={flowEdges}
            onNodeClick={(_, node) => setSelectedId(node.id)}
            fitView
            fitViewOptions={{ padding: 0.1 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
        <div
          className={clsx(
            "flex flex-shrink-0 flex-col gap-2 rounded-lg border px-4 py-3 shadow-sm",
            surfaceClassName("elevated"),
            strokeClassName("secondary"),
          )}
          style={{ width: 260 }}
        >
          <div className={clsx("text-sm font-semibold", inkClassName("primary"))}>
            {selectedNode ? selectedNode.label : "Select a node"}
          </div>
          {selectedNode ? (
            <Stack gap={10}>
              <p className={clsx("text-sm", inkClassName("secondary"))}>
                {selectedKind?.role ??
                  "Dataflow node — fires when its input channels satisfy AND-fan-in."}
              </p>
              {selectedNode.file ? (
                <Row gap={6} align="center">
                  <span className={clsx("text-xs", inkClassName("tertiary"))}>handler</span>
                  <Code inline>{selectedNode.file}</Code>
                </Row>
              ) : null}
            </Stack>
          ) : (
            <p className={clsx("text-sm", inkClassName("secondary"))}>
              Click any node to see its dispatch handler.
            </p>
          )}
        </div>
      </Row>
      <p className={clsx("text-sm", inkClassName("tertiary"))}>
        Source: an illustrative single-trace agentic graph (not a literal trace file). Solid = static
        edge; animated blue = mid-stream tool fan-out; animated yellow = spawn; animated purple = join.
        Nodes fire as soon as their input channels are ready — there is no turn counter.
      </p>
    </Stack>
  );
}

/**
 * AIPerf v2 async-dataflow graph engine explainer.
 *
 * Ports the single-view Cursor canvas `aiperf-graph-engine.canvas.tsx` onto
 * aiperf-flow's component vocabulary: the hand-drawn SVG DAG becomes a real
 * React Flow node/edge graph with click-to-inspect node detail, and the
 * surrounding prose sections (pipeline stages, collaborators, node kind
 * taxonomy, channel types, reducers) become `Stack`/`Grid`/`Table`/`Callout`.
 */
export function AiperfGraphEngineDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Graph Engine" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={24}>
            <Stack gap={8}>
              <h1 className={clsx("text-2xl font-bold", inkClassName("primary"))}>
                AIPerf v2 — Async-Dataflow Graph Engine
              </h1>
              <p className={clsx("max-w-3xl text-sm", inkClassName("secondary"))}>
                The graph engine executes benchmark conversations as a{" "}
                <span className={clsx("font-semibold", inkClassName("primary"))}>
                  directed dataflow graph
                </span>{" "}
                instead of a linear turn loop. Nodes fire the instant their input channels are
                ready, so tool calls, sub-agents, and branches run with real concurrency — each{" "}
                <Code inline>LlmNode</Code> still becomes exactly one credit, one request, and one
                record on the existing worker pipeline.
              </p>
            </Stack>

            <Grid columns={4} gap={16}>
              <Stat value="14" label="Node kinds" />
              <Stat value="8" label="Channel types" />
              <Stat value="4" label="Reducers" />
              <Stat value="2" label="Planes (build · schedule)" />
            </Grid>

            <Callout tone="info" title='What "v2" means here'>
              Triggered by a weka KV-cache IR trace (auto-detected) or a{" "}
              <Code inline>--custom-dataset-type dag_jsonl</Code> file. The{" "}
              <Code inline>TimingMode.GRAPH_IR</Code> strategy drives a{" "}
              <Code inline>TraceExecutor</Code> per trace; non-graph inputs keep the unchanged
              linear pipeline.
            </Callout>

            <Stack gap={12}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                Build → Schedule → Execute
              </h2>
              <Grid columns={3} gap={16}>
                <PipelineStage
                  step="1"
                  title="Build IR"
                  service="DatasetManager"
                  body="Parse the weka trace / dag_jsonl into a ParsedGraph IR, synthesize real content, and write each (trace, node) payload into an O(N) GRAPH_DELTA mmap. A structural graph_meta.msgpack sidecar is written beside it."
                  files={[
                    "src/aiperf/dataset/loader/graph/parser.py",
                    "src/aiperf/dataset/loader/graph/dataset_builder.py",
                    "src/aiperf/dataset/loader/graph/graph_meta_sidecar.py",
                  ]}
                />
                <PipelineStage
                  step="2"
                  title="Schedule"
                  service="TimingManager"
                  body="Load the structural sidecar (or re-parse on miss), hand the ParsedGraph to the graph timing strategy. Cross-trace concurrency and stop conditions are owned here, not by per-turn counters."
                  files={[
                    "src/aiperf/dataset/loader/graph/validator.py",
                    "src/aiperf/dataset/loader/graph/models.py",
                  ]}
                />
                <PipelineStage
                  step="3"
                  title="Execute"
                  service="TraceExecutor"
                  body="Run the dataflow firing loop per trace over the credit / worker / records pipeline. Workers rebuild each node payload from the shared GRAPH_DELTA mmap by (trace_id, node_ordinal), so any worker serves any node."
                  files={["src/aiperf/graph/executor.py", "src/aiperf/graph/credit_dispatch_adapter.py"]}
                />
              </Grid>
            </Stack>

            <Stack gap={12}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                The dataflow graph
              </h2>
              <GraphDiagram />
            </Stack>

            <Stack gap={12}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                Runtime collaborators
              </h2>
              <p className={clsx("text-sm", inkClassName("secondary"))}>
                The <Code inline>TraceExecutor</Code> is shared across every trace in a phase (all
                graph-derived state is immutable); only <Code inline>_TraceContext</Code> holds
                mutable per-trace state.
              </p>
              <Grid columns={3} gap={16}>
                <Collaborator
                  name="TraceExecutor"
                  role="Drives the per-trace firing loop: prepare inputs → execute → publish writes → schedule successors. Dispatch is a singledispatch table populated by the dispatch/ modules."
                  file="src/aiperf/graph/executor.py"
                />
                <Collaborator
                  name="Scheduler"
                  role="Pure graph adjacency: entry nodes, static + conditional successors, incoming edges for firing-delay gates. Immutable, shared across traces."
                  file="src/aiperf/graph/scheduler.py"
                />
                <Collaborator
                  name="VersionedChannelStore"
                  role="Per-trace append-only log per channel. await_inputs() gates AND-fan-in (count=N or 'all'); reducers merge concurrent writes in commit order."
                  file="src/aiperf/graph/channel_store.py"
                />
                <Collaborator
                  name="_TraceContext"
                  role="Per-trace mutable state: scheduled ids, task handles, cancelled nodes, finish-wall times, overflow flag. Passed into every _fire call."
                  file="src/aiperf/graph/context.py"
                />
                <Collaborator
                  name="CreditIssuer adapter"
                  role="Bridges an LlmNode/ToolNode dispatch to a real credit → request → record on the worker pipeline, honoring placement hints (sticky / free)."
                  file="src/aiperf/graph/credit_dispatch_adapter.py"
                />
                <Collaborator
                  name="Dispatch table"
                  role="One module per node kind registers its _execute body at import time. dispatch/__init__.py auto-imports them all before executor instantiation."
                  file="src/aiperf/graph/dispatch/__init__.py"
                />
              </Grid>
            </Stack>

            <Stack gap={12}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                Node kind taxonomy
              </h2>
              <Table
                columns={[
                  { key: "kind", label: "Node kind" },
                  { key: "role", label: "Role" },
                  { key: "module", label: "Handler" },
                  { key: "span", label: "Span" },
                ]}
                rows={NODE_KINDS.map((k) => ({
                  kind: <Code inline>{k.kind}</Code>,
                  role: <span className={inkClassName("secondary")}>{k.role}</span>,
                  module: <Code inline>{k.module}</Code>,
                  span: <span className={inkClassName("tertiary")}>{k.span}</span>,
                  tone: NODE_KIND_ROW_TONE[k.group],
                }))}
              />
            </Stack>

            <Grid columns="1fr 1fr" gap={24}>
              <Stack gap={12}>
                <h3 className={clsx("text-base font-semibold", inkClassName("primary"))}>
                  Channel types
                </h3>
                <p className={clsx("text-sm", inkClassName("secondary"))}>
                  Typed, versioned channels carry values between nodes. <Code inline>tool_call_stream</Code>{" "}
                  is append-only so a <Code inline>ToolCallNode</Code> can fire on the first chunk of a
                  still-streaming response.
                </p>
                <Row gap={6} wrap>
                  {CHANNEL_TYPES.map((c) => (
                    <span
                      key={c}
                      className={clsx(
                        "rounded-md border px-2 py-1 text-xs font-medium shadow-sm",
                        surfaceClassName("panel"),
                        strokeClassName("secondary"),
                        inkClassName("secondary"),
                      )}
                    >
                      {c}
                    </span>
                  ))}
                </Row>
              </Stack>
              <Stack gap={12}>
                <h3 className={clsx("text-base font-semibold", inkClassName("primary"))}>Reducers</h3>
                <p className={clsx("text-sm", inkClassName("secondary"))}>
                  How concurrent writes to one channel are merged on read.
                </p>
                <Stack gap={6}>
                  {REDUCERS.map((r) => (
                    <Row key={r.name} gap={8} align="center">
                      <Code inline>{r.name}</Code>
                      <span className={clsx("text-sm", inkClassName("tertiary"))}>{r.note}</span>
                    </Row>
                  ))}
                </Stack>
              </Stack>
            </Grid>

            <Divider />
            <Row gap={8} align="center">
              <span className={clsx("text-sm", inkClassName("tertiary"))}>Reference docs:</span>
              <Code inline>docs/benchmark-modes/dag.md</Code>
              <Code inline>docs/reference/weka-graph-structural-handoff.md</Code>
            </Row>
          </Stack>
        </div>
      </div>
    </div>
  );
}
