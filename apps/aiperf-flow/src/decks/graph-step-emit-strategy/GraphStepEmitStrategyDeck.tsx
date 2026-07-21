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
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Pill } from "../../prose/Pill.js";
import { Table } from "../../prose/Table.js";
import { Stat } from "../../prose/Stat.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

// Ported from
// canvases/graph-step-emit-strategy.canvas.tsx. Single-view canvas — no
// internal page tabs. North-star strategy for the graph-IR subsystem:
// collapse a 13-kind NodeUnion into a two-effect Step/Emit IR, scoped to a
// graph lane kept permanently separate from the legacy Conversation/Turn
// runtime. Source: specs/2026-06-30-graph-step-emit-separated-runtime-strategy.md.

// -----------------------------------------------------------------------
// Header
// -----------------------------------------------------------------------

function DeckHeader(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Stack gap={4}>
        <Row align="center" gap={10} wrap>
          <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
            Separated Graph Runtime on Step/Emit
          </h1>
          <Pill>Strategy spec</Pill>
        </Row>
        <p className={`text-sm ${inkClassName("secondary")}`}>
          North-star strategy for the graph-IR subsystem. Collapse a 13-kind{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>NodeUnion</span> into a
          two-effect <span className={`font-semibold ${inkClassName("primary")}`}>Step/Emit</span>{" "}
          IR — scoped to a graph lane kept permanently separate from the legacy{" "}
          <span className={`font-semibold ${inkClassName("primary")}`}>Conversation/Turn</span>{" "}
          runtime.
        </p>
        <p className={`text-xs ${inkClassName("tertiary")}`}>
          Source: specs/2026-06-30-graph-step-emit-separated-runtime-strategy.md · consolidates 5
          prior specs + a 2026-06-30 design session
        </p>
      </Stack>
      <Grid columns={4} gap={12}>
        <Stat value="SOUND" label="Base IR adjudication" tone="positive" />
        <Stat value="0 / 6" label="Refutations landed" tone="positive" />
        <Stat value="72" label="Agents · whole-spec review" />
        <Stat value="Greenfield" label="Step/Emit in code" tone="negative" />
      </Grid>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §1 · Governing decision — two separate runtimes
// -----------------------------------------------------------------------

const laneNodes: Node[] = [
  { id: "lane-graph-header", type: "header", position: { x: 0, y: 0 }, data: { title: "Graph lane", caption: "self-contained" } },
  {
    id: "lane-graph-adapters",
    type: "panel",
    position: { x: 0, y: 80 },
    data: { title: "Graph adapters", detail: "weka · dynamo · native", className: "border-l-4 border-l-category-green" },
  },
  {
    id: "lane-graph-ir",
    type: "panel",
    position: { x: 0, y: 200 },
    data: {
      title: "Graph IR + SegmentPool",
      detail: "Step/Emit · Plane 1 + Plane 2",
      className: "border-l-4 border-l-category-green",
    },
  },
  {
    id: "lane-graph-source",
    type: "panel",
    position: { x: 0, y: 320 },
    data: { title: "graph_ir_source", detail: "schedule plane", className: "border-l-4 border-l-category-green" },
  },
  {
    id: "lane-graph-replay",
    type: "card",
    position: { x: 0, y: 440 },
    data: {
      title: "GraphIRReplayStrategy",
      detail: "+ TraceExecutor + channel store",
      className: "border-l-4 border-l-category-green",
    },
  },

  { id: "lane-legacy-header", type: "header", position: { x: 420, y: 0 }, data: { title: "Legacy lane", caption: "unchanged" } },
  {
    id: "lane-legacy-turn",
    type: "panel",
    position: { x: 420, y: 80 },
    data: {
      title: "Conversation / Turn",
      detail: "13-mode god-object",
      className: "border-l-4 border-l-category-gray",
    },
  },
  {
    id: "lane-legacy-timing",
    type: "panel",
    position: { x: 420, y: 200 },
    data: { title: "Linear timing strategies", className: "border-l-4 border-l-category-gray" },
  },
  {
    id: "lane-legacy-session",
    type: "card",
    position: { x: 420, y: 320 },
    data: { title: "session_manager / worker", detail: "linear path", className: "border-l-4 border-l-category-gray" },
  },
];

const laneEdges: Edge[] = [
  { id: "e-lane-graph-adapters-ir", source: "lane-graph-adapters", target: "lane-graph-ir", type: "flow" },
  { id: "e-lane-graph-ir-source", source: "lane-graph-ir", target: "lane-graph-source", type: "flow" },
  { id: "e-lane-graph-source-replay", source: "lane-graph-source", target: "lane-graph-replay", type: "flow" },
  { id: "e-lane-legacy-turn-timing", source: "lane-legacy-turn", target: "lane-legacy-timing", type: "flow" },
  { id: "e-lane-legacy-timing-session", source: "lane-legacy-timing", target: "lane-legacy-session", type: "flow" },
];

function GoverningDecision(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        §1 · Governing decision — two separate runtimes
      </h2>
      <Callout tone="warning" title="Owner directive (2026-06-30)">
        Keep the graph content and graph runtime{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>permanently separate</span>{" "}
        from the legacy <span className={`font-semibold ${inkClassName("primary")}`}>Conversation/Turn</span>{" "}
        runtime. Do not consolidate, merge, or make one a view over the other. This{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>reverses</span> the
        feasibility doc&apos;s &quot;consolidation spine&quot; (approach C), which is removed from
        the roadmap.
      </Callout>
      <div style={{ height: 560 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={laneNodes}
          edges={laneEdges}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>
      <Callout tone="info" title="Accepted tradeoff">
        Two lanes means permanently accepting the duplication the feasibility doc wanted to
        remove. Deliberate: the graph lane never inherits the{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>Turn</span> god-object&apos;s
        unenforced 4-mode union or the lazy-view /{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>model_dump_json</span>{" "}
        problems. It is already the more capable model.
      </Callout>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §2 · Target IR — Step/Emit (graph-internal)
// -----------------------------------------------------------------------

function StepEmitIR(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        §2 · Target IR — Step/Emit (graph-internal)
      </h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        One vertex, one two-way effect. The 11 complicated kinds (spawn / await / subgraph / loop
        / barrier / tool-call / tool-result / compact / bootstrap / delay) survive only as{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>metadata kind tags</span>,
        never runtime types.
      </p>
      <Grid columns={2} gap={16}>
        <div className={`rounded-none border ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>effect: Dispatch</span>
            <Pill>weka + dynamo</Pill>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Hits the server. Live / measured timing. Consumes credit.
              </p>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                weka is <span className={`font-semibold ${inkClassName("primary")}`}>pure Dispatch</span>{" "}
                (LlmNode only).
              </p>
            </Stack>
          </div>
        </div>
        <div className={`rounded-none border ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>effect: Emit</span>
            <Pill>dynamo only</Pill>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Canned / replayed latency. No network, no credit. Carries a typed{" "}
                <span className={`font-semibold ${inkClassName("primary")}`}>Duration</span>.
              </p>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                Exercised by dynamo tool / think nodes.
              </p>
            </Stack>
          </div>
        </div>
      </Grid>
      <Grid columns={2} gap={16}>
        <div className={`rounded-none border ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={4}>
            <Row align="center" gap={8}>
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
                Plane 1 — dependency / timing
              </span>
              <div className="flex-1" />
              <Pill>graph</Pill>
            </Row>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Steps + Edges + gaps. Concurrency = absence of an edge. Join = fan-in{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>inputs</span> gate.
              Detached child = a Step the parent doesn&apos;t gate on.
            </p>
          </Stack>
        </div>
        <div className={`rounded-none border ${strokeClassName("secondary")} px-4 py-3`}>
          <Stack gap={4}>
            <Row align="center" gap={8}>
              <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Plane 2 — content</span>
              <div className="flex-1" />
              <Pill>the trie</Pill>
            </Row>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Content-addressed <span className={`font-semibold ${inkClassName("primary")}`}>SegmentPool</span>.
              The weka segment pool/trie{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>IS</span> Plane 2 — a
              projection, never a rival IR.{" "}
              <span className={`font-semibold ${inkClassName("primary")}`}>content_parent</span> ≠
              temporal-predecessor.
            </p>
          </Stack>
        </div>
      </Grid>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §4-5 · Adapter lowering to Step/Emit
// -----------------------------------------------------------------------

function Lowering(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        §4–5 · Adapter lowering to Step/Emit
      </h2>
      <Table
        columns={[
          { key: "adapter", label: "Adapter" },
          { key: "kinds", label: "Node kinds today" },
          { key: "effects", label: "Effect(s)" },
          { key: "framing", label: "Framing" },
          { key: "deltas", label: "Deltas / cost" },
        ]}
        rows={[
          {
            tone: "success",
            adapter: "weka",
            kinds: "LlmNode + StaticEdge",
            effects: "Dispatch only",
            framing: "Reshape under golden tests (mostly 1:1 renames)",
            deltas: "Gate half done; typed 3-resolution model, F-1 watchdog, F-3 escape still missing",
          },
          {
            tone: "neutral",
            adapter: "dynamo",
            kinds: "LlmNode / ReplayNode / SubgraphNode",
            effects: "Dispatch + Emit",
            framing: "Phase B (content emission) is the hard prerequisite",
            deltas: "Tools built as ReplayNode(duration=0) — latency dropped today",
          },
          {
            tone: "warning",
            adapter: "native",
            kinds: "every NodeKind (spawn/loop/barrier/await…)",
            effects: "Dispatch + Emit + reduced kinds",
            framing: "Convert (re-express capabilities) or scope out",
            deltas: "Blocks blanket deletion of the dispatch zoo",
          },
        ]}
      />
      <Callout tone="info" title="Leverage: weka-interval-order-causality is a down-payment">
        Its <span className={`font-semibold ${inkClassName("primary")}`}>A→B iff A.end≤B.start ∧ rank(A)&lt;rank(B)</span>{" "}
        edges are §9&apos;s &quot;completed-before + api_time&quot; edges, and it already splits
        content-lineage from timing. Land it first — the weka conversion (item B) shrinks.
      </Callout>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §8 · Sequencing
// -----------------------------------------------------------------------

const SEQUENCING_STEPS: Array<{ n: string; title: string; note: string; tag: string }> = [
  { n: "1", title: "weka interval-order-causality", note: "Establish two-plane (timing vs content) edges — down-payment on §4.", tag: "weka" },
  { n: "2", title: "Phase B — dynamo content emission", note: "dynamo produces records for the first time; add §6.4 tool-data hook.", tag: "dynamo" },
  { n: "3", title: "Define the Step/Emit IR", note: "Dispatch-only subset first (§4A).", tag: "IR" },
  { n: "4", title: "weka → Step/Emit", note: "Behind a flag, A/B under content_byte_exact_vs_v04, then flip default.", tag: "weka" },
  {
    n: "4b",
    title: "Decouple the graph carrier",
    note: "Give the graph lane its own end-to-end node/segment/manifest carrier so it stops borrowing Turn.",
    tag: "§1 real",
  },
  { n: "5", title: "dynamo → Step/Emit", note: "Lowering (§5.2); tools become Emit(category=tool).", tag: "dynamo" },
  {
    n: "6",
    title: "Per-tool-type timing scaling",
    note: "Graph→Graph transform keyed on metadata.tool_class (dynamo-only).",
    tag: "timing",
  },
  { n: "7", title: "Node-zoo / dead-code collapse", note: "Once lanes no longer emit the complicated kinds.", tag: "cleanup" },
];

function Sequencing(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>§8 · Sequencing</h2>
      <p className={`text-xs ${inkClassName("tertiary")}`}>
        Each step is its own flag-gated spec → plan → implement cycle under byte-parity / fidelity
        gates. Order is a proposal, not resourced.
      </p>
      <Stack gap={0}>
        {SEQUENCING_STEPS.map((s, i) => (
          <div key={s.n}>
            <Row gap={12} align="start" className="py-2">
              <div
                className="flex h-6 min-w-6.5 items-center justify-center rounded-full border text-xs font-semibold text-accent-primary"
                style={{ borderColor: "var(--color-accent-primary)" }}
              >
                {s.n}
              </div>
              <Stack gap={2} className="min-w-0 flex-1">
                <Row gap={8} align="center" wrap>
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{s.title}</span>
                  <Pill>{s.tag}</Pill>
                </Row>
                <span className={`text-sm ${inkClassName("secondary")}`}>{s.note}</span>
              </Stack>
            </Row>
            {i < SEQUENCING_STEPS.length - 1 && <Divider />}
          </div>
        ))}
      </Stack>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §7 · Node-zoo collapse
// -----------------------------------------------------------------------

function NodeZoo(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        §7 · Node-zoo collapse — types collapse, capabilities do not
      </h2>
      <Callout tone="info" title="Correction from prior drafts">
        Converting to Step/Emit collapses node{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>types</span>, it does{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>not</span> remove
        capabilities. spawn → detached step · barrier → fan-in gate · await → join · loop →
        pre-unrolled DAG (needs a static max-iteration cap).
      </Callout>
      <Grid columns={3} gap={16}>
        <div className={`rounded-none border ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Deletable (Tier 1)</span>
            <Pill>now</Pill>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Zero-reference dead code, unrelated to the node-zoo question.
              </p>
              <Row gap={8} align="start">
                <span className={`min-w-24 text-sm font-medium ${inkClassName("tertiary")}`}>errors.py</span>
                <span className={`text-sm ${inkClassName("primary")}`}>unused Llm/Retriable/Retries error classes</span>
              </Row>
              <Row gap={8} align="start">
                <span className={`min-w-24 text-sm font-medium ${inkClassName("tertiary")}`}>adapter</span>
                <span className={`text-sm ${inkClassName("primary")}`}>CreditDispatchAdapter dead fields</span>
              </Row>
              <Row gap={8} align="start">
                <span className={`min-w-24 text-sm font-medium ${inkClassName("tertiary")}`}>channel</span>
                <span className={`text-sm ${inkClassName("primary")}`}>populate_initial_state / apply_writes</span>
              </Row>
              <Row gap={8} align="start">
                <span className={`min-w-24 text-sm font-medium ${inkClassName("tertiary")}`}>misc</span>
                <span className={`text-sm ${inkClassName("primary")}`}>
                  endpoint/branch resolvers, TraceTimeline.streams, StreamReducer.reduce
                </span>
              </Row>
            </Stack>
          </div>
        </div>
        <div className={`rounded-none border ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Judgment (Tier 2)</span>
            <Pill>decide</Pill>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>Unwired subsystems — wire or delete.</p>
              <Row gap={8} align="start">
                <span className={`min-w-24 text-sm font-medium ${inkClassName("tertiary")}`}>span_builder</span>
                <span className={`text-sm ${inkClassName("primary")}`}>
                  SpanBuilder never receives spans= (OTel: wire or drop)
                </span>
              </Row>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                If kept under §1, it is graph-lane-internal.
              </p>
            </Stack>
          </div>
        </div>
        <div className={`rounded-none border ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-4 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Not deletable</span>
            <Pill>output, not precondition</Pill>
          </div>
          <div className="px-4 py-3">
            <Stack gap={6}>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                dispatch/&#123;spawn, await, subgraph, loop, barrier, tool…&#125;, replay_barrier,
                graph_ir_barriers, spawn_keys, handles.
              </p>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                native can author their kinds; capabilities must be re-expressed first. Removal is
                an <span className={`font-semibold ${inkClassName("primary")}`}>output</span> of
                the conversion.
              </p>
            </Stack>
          </div>
        </div>
      </Grid>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §12 · Adjudicated refinements the conversion must honor
// -----------------------------------------------------------------------

function FailureRefinements(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        §12 · Adjudicated refinements the conversion must honor
      </h2>
      <Table
        columns={[
          { key: "ref", label: "Ref" },
          { key: "requirement", label: "Requirement" },
          { key: "status", label: "Status" },
        ]}
        rows={[
          {
            tone: "success",
            ref: "F-1",
            requirement:
              "Decidable block-structured / acyclic-post-unroll subclass + runtime gate-timeout watchdog → forced WILL_NOT_PRODUCE. Stop calling the static check a proof.",
            status: "Watchdog prototyped as EXECUTOR_WATCHDOG_TIMEOUT_S",
          },
          {
            tone: "neutral",
            ref: "F-3",
            requirement:
              "any/quorum gates need an escape transition — resolve self to FAILED once n−k+1 producers resolve non-real.",
            status: "Validator must reject relaxed gates lacking it",
          },
          {
            tone: "neutral",
            ref: "M-1",
            requirement:
              "Deterministic multi-writer joins — add_messages merges in static Step-id order; reject undeclared multi-writer live joins.",
            status: "Bites only if graph lane grows source=live",
          },
          {
            tone: "warning",
            ref: "M-3",
            requirement: "--request-count N is an atomic counter at the credit issuer, NOT a dataflow barrier (~15k/s ceiling).",
            status: "Prefer pre-assigned dispatch ordinals",
          },
          {
            tone: "neutral",
            ref: "M-7",
            requirement: "Loops need a static max-iteration cap to pre-unroll; reject uncapped loops.",
            status: "GRAPH.LOOP_MAX_ITERATIONS already exists",
          },
          {
            tone: "neutral",
            ref: "F-4",
            requirement:
              "WITHDRAWN — user-doubling on FAILED live channels is correct benchmark behavior, not a bug. Do not fix.",
            status: "merge_consecutive_user stays a lowering knob",
          },
        ]}
      />
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §13 · Verification & confidence
// -----------------------------------------------------------------------

function Confidence(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>§13 · Verification &amp; confidence</h2>
      <Table
        columns={[
          { key: "claim", label: "Claim" },
          { key: "confidence", label: "Confidence" },
          { key: "basis", label: "Basis" },
        ]}
        rows={[
          { tone: "success", claim: "Step/Emit IR sound as a design", confidence: "High", basis: "5 adjudication files; 72-agent SOUND; 0/6 refutations" },
          { tone: "success", claim: "Code facts (§11 provenance)", confidence: "High", basis: "Grepped/read on the working tree this session" },
          { tone: "success", claim: "Trie = Plane 2, not an alternative", confidence: "High", basis: "Direct quotes, feasibility + unified specs" },
          { tone: "success", claim: "Per-tool-scaling mechanism (§6)", confidence: "High", basis: "Metadata-keyed Graph→Graph transform" },
          { tone: "success", claim: "Phase B is content-plane / prerequisite", confidence: "High", basis: "Read Phase B plan + design in full" },
          {
            tone: "danger",
            claim: "weka conversion is a mostly-renames reshape",
            confidence: "LOW — unverified",
            basis: "manifest/catalog/ordinal/worker coupling not audited",
          },
          { tone: "neutral", claim: "§7 removability", confidence: "Corrected", basis: "Now scoped to Tier-1 only (native + capability-vs-type)" },
          {
            tone: "neutral",
            claim: "weka already failure-robust the spec's way",
            confidence: "Corrected",
            basis: "Gate half only; typed model + F-1 + F-3 are deltas",
          },
          { tone: "warning", claim: "§1 fully realized today", confidence: "Partial", basis: "Separate at dispatch, not at dataset/transport" },
          { tone: "neutral", claim: "§8 sequencing order", confidence: "Proposal", basis: "Idealized order; not resourced/validated" },
        ]}
      />
      <Callout tone="danger" title="One remaining pre-execution action">
        Audit the runtime plumbing coupling (manifest / catalog / ordinal /{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>worker_materialize</span>) to
        the concrete node types before treating §4&apos;s effort estimate as real. Everything else
        is verified or honestly fenced.
      </Callout>
    </Stack>
  );
}

// -----------------------------------------------------------------------
// §9 · Hard exclusions
// -----------------------------------------------------------------------

const NON_GOALS: string[] = [
  "No merge of graph lane with legacy Conversation/Turn.",
  "No branch-on-live-model-output (validator-enforced).",
  "No node-kind zoo / second IR inside the graph lane.",
  "weka tool-delay scaling is out (dynamo-only).",
  "No universal-IR ambition — Turn is not replaced.",
  "Multi-root dynamo files remain rejected.",
  "agentx external_event stays out of scope.",
];

function NonGoals(): React.JSX.Element {
  return (
    <Stack gap={8}>
      <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>§9 · Hard exclusions</h3>
      <Grid columns={2} gap={8}>
        {NON_GOALS.map((line) => (
          <span key={line} className={`text-sm ${inkClassName("secondary")}`}>
            {`• ${line}`}
          </span>
        ))}
      </Grid>
    </Stack>
  );
}

/**
 * Ports `graph-step-emit-strategy.canvas.tsx` (a real, hand-authored Cursor Canvas) onto
 * aiperf-flow's component vocabulary. Single-view canvas — north-star strategy for collapsing
 * the graph-IR subsystem's 13-kind `NodeUnion` into a two-effect `Step`/`Emit` IR, scoped to a
 * graph lane kept permanently separate from the legacy `Conversation`/`Turn` runtime.
 */
export function GraphStepEmitStrategyDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Graph Step/Emit Strategy" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={28}>
            <DeckHeader />
            <Divider />
            <GoverningDecision />
            <StepEmitIR />
            <Lowering />
            <Sequencing />
            <NodeZoo />
            <FailureRefinements />
            <Confidence />
            <Divider />
            <NonGoals />
          </Stack>
        </div>
      </div>
    </div>
  );
}
