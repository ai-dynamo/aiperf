/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `step-dispatch-emit-system.canvas.tsx` (a real, hand-authored Cursor Canvas) onto
//! aiperf-flow's component vocabulary. Single-view canvas — no internal page tabs. Explains the
//! greenfield two-effect workload IR: one vertex kind (`Step`) whose `effect` is either a
//! `Dispatch` (hits the server, live timing, consumes credit) or an `Emit` (canned/replayed
//! latency, no network). Source: `src/aiperf/dataset/loader/graph/step_emit.py`,
//! `step_emit_weka.py`, `step_emit_validate.py`, `src/aiperf/graph/executor.py`.

import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { Stat } from "../../prose/Stat.js";
import { Code } from "../../prose/Code.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

function FieldTable({ rows }: { rows: [string, string, string][] }): React.JSX.Element {
  return (
    <Table
      columns={[
        { key: "name", label: "Field" },
        { key: "type", label: "Type" },
        { key: "note", label: "Note" },
      ]}
      rows={rows.map(([name, type, note]) => ({
        name: <Code inline>{name}</Code>,
        type: <Code inline>{type}</Code>,
        note: <span className={`text-sm ${inkClassName("secondary")}`}>{note}</span>,
      }))}
    />
  );
}

function EffectCard({
  kind,
  tag,
  tagline,
  fields,
  traits,
}: {
  kind: string;
  tag: string;
  tagline: string;
  fields: [string, string][];
  traits: string[];
}): React.JSX.Element {
  return (
    <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
      <div
        className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}
      >
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{kind}</span>
        <span
          className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
        >
          effect = &quot;{tag}&quot;
        </span>
      </div>
      <div className="px-4 py-3">
        <Stack gap={10}>
          <p className={`text-sm ${inkClassName("secondary")}`}>{tagline}</p>
          <div>
            {fields.map(([name, type]) => (
              <Row key={name} gap={8} className="py-0.5">
                <span className={`min-w-[130px] font-mono text-xs ${inkClassName("primary")}`}>
                  {name}
                </span>
                <span className={`font-mono text-xs ${inkClassName("tertiary")}`}>{type}</span>
              </Row>
            ))}
          </div>
          <div className={`border-t ${strokeClassName("secondary")}`} />
          <Stack gap={4}>
            {traits.map((trait) => (
              <p key={trait} className={`text-sm ${inkClassName("secondary")}`}>
                • {trait}
              </p>
            ))}
          </Stack>
        </Stack>
      </div>
    </div>
  );
}

function FiringStep({
  n,
  label,
  desc,
}: {
  n: number;
  label: string;
  desc: string;
}): React.JSX.Element {
  return (
    <Row gap={12} align="start" className="py-1.5">
      <div
        className={`flex h-6 min-w-6.5 items-center justify-center rounded-full border text-xs font-semibold ${inkClassName("secondary")} ${strokeClassName("primary")}`}
      >
        {n}
      </div>
      <span className={`min-w-[92px] font-mono text-xs font-semibold ${inkClassName("primary")}`}>
        {label}
      </span>
      <span className={`text-sm ${inkClassName("secondary")}`}>{desc}</span>
    </Row>
  );
}

// ---------------------------------------------------------------------------
// Diagram 1 — IR object model
// ---------------------------------------------------------------------------

const objectModelNodes: Node[] = [
  { id: "wl", type: "card", position: { x: 320, y: 0 }, data: { title: "Workload", subtitle: "graph + seed + traces" } },
  { id: "g", type: "card", position: { x: 60, y: 130 }, data: { title: "Graph", subtitle: "Plane 1: topology" } },
  { id: "t", type: "card", position: { x: 580, y: 130 }, data: { title: "Trace[]", subtitle: "control pins" } },
  { id: "s", type: "card", position: { x: 0, y: 260 }, data: { title: "Step{}", subtitle: "the only vertex" } },
  { id: "e", type: "card", position: { x: 220, y: 260 }, data: { title: "Edge[]", subtitle: "waits-for" } },
  { id: "c", type: "card", position: { x: 440, y: 260 }, data: { title: "Channel{}", subtitle: "Plane 2 decls" } },
  { id: "d", type: "card", position: { x: 0, y: 390 }, data: { title: "Dispatch", subtitle: "server + credit" } },
  { id: "m", type: "card", position: { x: 220, y: 390 }, data: { title: "Emit", subtitle: "canned latency" } },
];

const objectModelEdges: Edge[] = [
  { id: "e-wl-g", source: "wl", target: "g", type: "flow" },
  { id: "e-wl-t", source: "wl", target: "t", type: "flow" },
  { id: "e-g-s", source: "g", target: "s", type: "flow" },
  { id: "e-g-e", source: "g", target: "e", type: "flow" },
  { id: "e-g-c", source: "g", target: "c", type: "flow" },
  { id: "e-s-d", source: "s", target: "d", type: "flow", label: "effect =" },
  { id: "e-s-m", source: "s", target: "m", type: "flow" },
  { id: "e-t-c", source: "t", target: "c", style: { strokeDasharray: "4 4" } },
];

// Top-to-bottom object-model tree; ELK layers Workload → Graph/Trace → Step/Edge/Channel → effects.
const OBJECT_MODEL_LAYOUT: ElkOptions = { direction: "DOWN" };

// Each diagram is a self-contained `AutoLayoutFlow` (its own `ReactFlowProvider`), so the two
// React Flow instances on this view never collide on a shared store.
function ObjectModelDiagram(): React.JSX.Element {
  return (
    <AutoLayoutFlow
      className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}
      nodes={objectModelNodes}
      edges={objectModelEdges}
      layout={OBJECT_MODEL_LAYOUT}
      height={480}
    />
  );
}

// ---------------------------------------------------------------------------
// Diagram 2 — Projection pipeline
// ---------------------------------------------------------------------------

const projectionNodes: Node[] = [
  { id: "parsed", type: "card", position: { x: 0, y: 0 }, data: { title: "ParsedGraph", subtitle: "trie IR" } },
  {
    id: "projector",
    type: "card",
    position: { x: 260, y: 0 },
    data: { title: "weka_trie_to_workload", subtitle: "pure projection" },
  },
  { id: "workload2", type: "card", position: { x: 580, y: 0 }, data: { title: "Workload", subtitle: "Step/Emit IR" } },
  { id: "store", type: "card", position: { x: 840, y: 0 }, data: { title: "unified store", subtitle: "byte-parity build" } },
];

const projectionEdges: Edge[] = [
  { id: "e-parsed-projector", source: "parsed", target: "projector", type: "flow", label: "weka / dynamo" },
  {
    id: "e-projector-workload",
    source: "projector",
    target: "workload2",
    type: "flow",
    label: "LlmNode -> Dispatch, else -> Emit + kind tag",
  },
  { id: "e-workload-store", source: "workload2", target: "store", type: "flow", label: "mirrors interned builder" },
];

// Left→right projection chain: ParsedGraph → projector → Workload → unified store.
const PROJECTION_LAYOUT: ElkOptions = { direction: "RIGHT" };

function ProjectionDiagram(): React.JSX.Element {
  return (
    <AutoLayoutFlow
      className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}
      nodes={projectionNodes}
      edges={projectionEdges}
      layout={PROJECTION_LAYOUT}
      height={220}
    />
  );
}

// ---------------------------------------------------------------------------
// Data: firing steps, lowering table, source files
// ---------------------------------------------------------------------------

const FIRING_STEPS: [string, string][] = [
  ["Gate", "Wait for AND-fan-in ChannelReq arrivals (relaxed barriers bypass)"],
  ["Snapshot", "Capture channel-store sequence, compute causal input snapshot"],
  ["Timing", "Apply static / conditional / node-level delay gates"],
  ["Span", "Open the node span when span collection is wired"],
  ["Execute", "Run the registered dispatch body for the effect"],
  ["Write", "Publish result writes to the VersionedChannelStore"],
  ["Producers", "Mark all declared output producers done"],
  ["Successors", "Schedule static + selected conditional successors"],
  ["Orphan", 'Mark untaken branch producers done so count="all" waiters wake'],
];

const LOWERING_ROWS: [string, string, string][] = [
  ["llm", "Dispatch", "prompt, output -> response_channel + writes; trie block verbatim"],
  ["replay", "Emit", "canned outputs, recorded/authored duration"],
  ["tool", "Emit", 'kind="tool"; may bypass adapter via endpoint dispatch_tool'],
  ["tool_call", "Emit", "model-initiated call; SATF payload preserved"],
  ["tool_result", "Emit", "tool result injected back into context"],
  ["subgraph", "Emit", "kind + graph_ref; body recursed for ordinals"],
  ["spawn / await", "Emit", "detached child handle / join"],
  ["delay", "Emit", "no-op pass-through latency marker"],
  ["barrier", "Emit", "all / any / quorum sync marker"],
  ["compact / bootstrap", "Emit", "conversation compaction / system markers"],
  ["loop", "Emit", "pre-unrolled; not a Step/Emit type at runtime"],
];

const SOURCE_FILES: [string, string, string][] = [
  ["src/aiperf/dataset/loader/graph/step_emit.py", "step_emit.py", "the IR structs"],
  ["src/aiperf/dataset/loader/graph/step_emit_weka.py", "step_emit_weka.py", "projection + store build"],
  ["src/aiperf/dataset/loader/graph/step_emit_validate.py", "step_emit_validate.py", "structural validators"],
  ["src/aiperf/graph/executor.py", "graph/executor.py", "runtime firing"],
];

const loweringColumns: TableColumn[] = [
  { key: "kind", label: "Authored node kind" },
  { key: "effect", label: "Effect" },
  { key: "notes", label: "Notes" },
];

const loweringTableRows: TableRow[] = LOWERING_ROWS.map(([kind, effect, notes]) => ({
  kind: <Code inline>{kind}</Code>,
  effect: <span className="text-sm font-semibold">{effect}</span>,
  notes: <span className={`text-sm ${inkClassName("secondary")}`}>{notes}</span>,
  tone: effect === "Dispatch" ? "warning" : "neutral",
}));

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function Header(): React.JSX.Element {
  return (
    <Stack gap={12}>
      <Row gap={10} align="center" wrap>
        <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>Step / Dispatch / Emit IR</h1>
        <span
          className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
        >
          graph-lane-internal
        </span>
      </Row>
      <p className={`max-w-3xl text-sm ${inkClassName("secondary")}`}>
        The greenfield two-effect workload IR. One vertex kind (<Code inline>Step</Code>) whose{" "}
        <Code inline>effect</Code> is either a <Code inline>Dispatch</Code> (hits the server, live timing, consumes
        credit) or an <Code inline>Emit</Code> (canned/replayed latency, no network). Every complicated node kind
        collapses to these primitives plus a <Code inline>metadata</Code> tag.
      </p>
      <Grid columns={4} gap={12}>
        <Stat value="1" label="vertex kind (Step)" />
        <Stat value="2" label="2 effects (Dispatch / Emit)" tone="positive" />
        <Stat value="2" label="planes (timing / content)" />
        <Stat value="AND" label="fan-in gate semantics" />
      </Grid>
    </Stack>
  );
}

function OneVertexTwoEffects(): React.JSX.Element {
  return (
    <Stack gap={12}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>One vertex, two effects</h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        A <Code inline>Step</Code> fires when its AND-fan-in <Code inline>inputs</Code> gate is satisfied and its
        edge / <Code inline>pre_wait</Code> delays elapse. The single irreducible split is whether its effect
        touches the network.
      </p>
      <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
        <Stack gap={6}>
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Step</span>
          <FieldTable
            rows={[
              ["id", "str", "namespaced id encodes nesting (parent/child/step)"],
              ["effect", "Dispatch | Emit", "the one irreducible split (tagged on 'effect')"],
              ["inputs", "ChannelReq[]", "AND-fan-in gate; empty = fire on edge readiness"],
              ["writes", "str[]", "channels this Step produces"],
              ["metadata", "dict", "round-trip kind tag / span attrs; never runtime-branching"],
            ]}
          />
        </Stack>
      </div>
      <Grid columns={2} gap={12}>
        <EffectCard
          kind="Dispatch"
          tag="dispatch"
          tagline="Server-hitting: live timing, consumes credit. Service time is measured, never a field."
          fields={[
            ["prompt", "list[Any]"],
            ["response_channel", "str"],
            ["endpoint", "str | None"],
            ["streaming", "bool = True"],
            ["pre_wait", "Duration | None"],
            ["overrides", "dict | None"],
            ["expected", "ExpectedTokens | None"],
          ]}
          traits={[
            "Routes through CreditDispatchAdapter -> CreditIssuer",
            "Acquires a prefill slot; bypasses session slots",
            "Latency is observed at the wire, not authored",
          ]}
        />
        <EffectCard
          kind="Emit"
          tag="emit"
          tagline="Canned/replayed: authored/recorded latency, no network, no credit."
          fields={[
            ["outputs", "list[str]"],
            ["duration", "Duration"],
          ]}
          traits={[
            "Tools, delays, compaction/bootstrap lower here",
            "Optionally sleeps, writes replayed/synth outputs",
            "Never mints a trie manifest (no prompt_segment_ids)",
          ]}
        />
      </Grid>
    </Stack>
  );
}

function ObjectModel(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>IR object model</h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        A <Code inline>Workload</Code> is one shared <Code inline>Graph</Code> plus a bounded population of
        control-pinning <Code inline>Trace</Code>s. Plane 2 (the <Code inline>SegmentPool</Code>) is a separate
        runtime companion, not a field of the serialized IR.
      </p>
      <ObjectModelDiagram />
      <Grid columns={2} gap={12}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Edge</span>
            <FieldTable
              rows={[
                ["source", "str", "producing Step id"],
                ["target", "str | END", "unconditional successor, or exit sentinel"],
                ["branches", "dict | None", "keyed successors at a branch point"],
                ["weights", "dict | None", "seeded control distribution over keys"],
                ["gap", "Duration | None", "scheduling delay (input timing)"],
              ]}
            />
          </Stack>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Trace</span>
            <FieldTable
              rows={[
                ["id", "str", 'stable trace id (e.g. "t-1#0")'],
                ["arrival_time", "float", "wall-clock admission offset (FIXED_SCHEDULE)"],
                ["initial_state", "dict", "seed channel values (not a producer arrival)"],
                ["selected_branches", "dict", "source_id -> branch_key control pin"],
                ["replay_outputs", "dict", "canned content for Emit steps"],
              ]}
            />
          </Stack>
        </div>
      </Grid>
    </Stack>
  );
}

function TwoPlanes(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Two planes</h2>
      <Grid columns={2} gap={12}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div
            className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}
          >
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
              Plane 1 — dependency / timing
            </span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              serialized IR
            </span>
          </div>
          <p className={`px-4 py-3 text-sm ${inkClassName("secondary")}`}>
            The <Code inline>Step</Code> / <Code inline>Edge</Code> graph plus <Code inline>Channel</Code>{" "}
            declarations. All topology and population-wide timing. This is what{" "}
            <Code inline>validate_workload</Code> checks and what gets serialized with <Code inline>msgspec</Code>.
          </p>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div
            className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}
          >
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
              Plane 2 — content / cache
            </span>
            <span
              className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}
            >
              runtime companion
            </span>
          </div>
          <p className={`px-4 py-3 text-sm ${inkClassName("secondary")}`}>
            The content-addressed <Code inline>SegmentPool</Code> — a separate on-disk artifact carried alongside,
            NOT a field of the IR. Drained into the unified store in insertion order for identical int handles.
          </p>
        </div>
      </Grid>
    </Stack>
  );
}

function NodeKindLowering(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Node-kind lowering</h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        Projection (<Code inline>weka_trie_to_workload</Code>) maps a trie <Code inline>ParsedGraph</Code> — weka
        or dynamo — onto Step/Emit. Only <Code inline>LlmNode</Code> becomes a <Code inline>Dispatch</Code>; every
        other kind becomes an <Code inline>Emit</Code> carrying its <Code inline>metadata[&quot;kind&quot;]</Code>{" "}
        tag.
      </p>
      <Table columns={loweringColumns} rows={loweringTableRows} />
    </Stack>
  );
}

function ProjectionPipeline(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Projection pipeline</h2>
      <ProjectionDiagram />
      <Callout tone="info" title="Byte-parity contract">
        The pool drain order, the per-trace ordinal order (mirrors <Code inline>subgraph_aware_trie_ordinals</Code>
        ), and the manifest inputs (mirror <Code inline>_trie_envelope</Code>) all match the interned builder — so{" "}
        <Code inline>build_unified_trie_store_from_workload</Code> reproduces its ordinals and handles
        byte-for-byte.
      </Callout>
    </Stack>
  );
}

function RuntimeFiring(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Runtime firing path</h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        There is no central ready queue — readiness is expressed through channel waiters, futures, and{" "}
        <Code inline>TaskGroup</Code> task creation. Each node walks the same nine-step path in the{" "}
        <Code inline>TraceExecutor</Code>.
      </p>
      <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-2`}>
        <Stack gap={0}>
          {FIRING_STEPS.map(([label, desc], i) => (
            <FiringStep key={label} n={i + 1} label={label} desc={desc} />
          ))}
        </Stack>
      </div>
      <Grid columns={2} gap={12}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
            Dispatch execution
          </div>
          <p className={`px-4 py-3 text-sm ${inkClassName("secondary")}`}>
            Builds a <Code inline>DispatchRequest</Code>, parks a future in the{" "}
            <Code inline>CreditDispatchAdapter</Code>, and awaits <Code inline>issue_graph_credit</Code>. The
            return observer routes the worker result back by <Code inline>credit.trace_id</Code>. Bounded by{" "}
            <Code inline>AIPERF_GRAPH_DISPATCH_TIMEOUT_S</Code> once it reaches the adapter.
          </p>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
            Emit execution
          </div>
          <p className={`px-4 py-3 text-sm ${inkClassName("secondary")}`}>
            Issues no graph credit. Optionally sleeps for the authored / recorded <Code inline>Duration</Code>,
            then writes replayed or synthesized outputs. Tool, delay, compact, bootstrap, and tool-result nodes
            all take this path.
          </p>
        </div>
      </Grid>
    </Stack>
  );
}

function StructuralValidation(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Structural validation</h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        <Code inline>validate_workload</Code> returns a list of issues; empty means valid. It enforces the
        cleanly-checkable base-IR invariants.
      </p>
      <Grid columns={3} gap={12}>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Effect coherence</span>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              <Code inline>Dispatch.response_channel</Code> must be in <Code inline>writes</Code>; every{" "}
              <Code inline>Emit.outputs</Code> entry must be in <Code inline>writes</Code>.
            </p>
          </Stack>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Reference integrity</span>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Total producer accounting: every input channel has a producer; every edge endpoint is a known Step
              or <Code inline>END</Code>.
            </p>
          </Stack>
        </div>
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Acyclicity</span>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Kahn-style cycle detection over static + branch edges. Step/Emit graphs must be acyclic post-unroll.
            </p>
          </Stack>
        </div>
      </Grid>
      <Callout tone="info" title="Deferred by design (not gaps)">
        The full free-choice / block-structured soundness subclass (F-1) belongs to the lowering/unroll stage;
        relaxed-gate escape transitions (F-3) are N/A because this IR is AND-only; the static loop-iteration cap
        (M-7) is a lowering constraint since loops pre-unroll.
      </Callout>
    </Stack>
  );
}

function SourceFiles(): React.JSX.Element {
  return (
    <Stack gap={8}>
      <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>Source files</h3>
      <Grid columns={2} gap={8}>
        {SOURCE_FILES.map(([path, label, description]) => (
          <Row key={path} gap={8} align="center">
            <span title={path}>
              <Code inline>{label}</Code>
            </span>
            <span className={`text-sm ${inkClassName("tertiary")}`}>— {description}</span>
          </Row>
        ))}
      </Grid>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

/**
 * Ports `docs/canvases/step-dispatch-emit-system.canvas.tsx` (a real, hand-authored Cursor
 * Canvas) onto aiperf-flow's component vocabulary. Single-view canvas — explains the greenfield
 * two-effect (`Dispatch` / `Emit`) workload IR: one vertex kind, AND-only fan-in, two planes
 * (dependency/timing vs. content/cache), node-kind lowering from trie `ParsedGraph`, the
 * nine-step runtime firing path, and structural validation.
 */
export function StepDispatchEmitSystemDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Step / Dispatch / Emit" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={28}>
            <Header />
            <Callout tone="info" title="Structural invariant (spec §0)">
              Control flow never branches on live model output — the IR has no field by which a branch could read
              a channel value, so this holds structurally. Fan-in gates are{" "}
              <span className="font-semibold">AND-only</span>.
            </Callout>
            <div className={`border-t ${strokeClassName("secondary")}`} />
            <OneVertexTwoEffects />
            <ObjectModel />
            <TwoPlanes />
            <NodeKindLowering />
            <ProjectionPipeline />
            <RuntimeFiring />
            <StructuralValidation />
            <div className={`border-t ${strokeClassName("secondary")}`} />
            <SourceFiles />
          </Stack>
        </div>
      </div>
    </div>
  );
}
