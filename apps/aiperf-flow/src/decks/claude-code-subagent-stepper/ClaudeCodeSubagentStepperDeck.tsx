/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ported from `docs/canvases/claude-code-subagent-stepper.canvas.tsx` (a real, hand-authored
//! Cursor Canvas), single-view (no in-deck page tabs in the source). Walks a Claude Code
//! conversation trace where the main agent fans out three subagents plus a detached background
//! agent from the same instant, two of the subagents join an early AND-fan-in (`synthesize`), and
//! the third — slowed by a WebFetch tool call — misses that join and instead merges into a later
//! `report` turn. A shared wall-clock `t` per step drives node/edge highlighting, two gate
//! progress trackers, and an in-flight LLM request log, so a request stays visibly "in flight" for
//! its whole latency window instead of flashing for a single step.

import { useMemo } from "react";
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
import { Stat } from "../../prose/Stat.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";

// --- trace data, ported verbatim from the canvas source ------------------

type NodeState = "pending" | "ready" | "firing" | "done";
type Kind = "dispatch" | "emit";

/**
 * Every trace node carries a real wall-clock window `[start, end]` in ms — the single source of
 * truth the whole page derives from: node state, gate progress, and the in-flight request log
 * are all computed from where the current step's clock lands in these windows.
 */
export type TraceNodeDef = {
  id: string;
  label: string;
  kind: Kind;
  start: number;
  end: number;
  detached?: boolean;
  isl?: number;
  osl?: number;
};

export const TRACE_NODES: TraceNodeDef[] = [
  { id: "START", label: "START", kind: "dispatch", start: 0, end: 0 },
  { id: "M1", label: "main: plan", kind: "dispatch", start: 200, end: 1600, isl: 420, osl: 180 },
  { id: "S1a", label: "sub1: reason", kind: "dispatch", start: 1800, end: 3400, isl: 1600, osl: 240 },
  { id: "S1t", label: "sub1: Grep", kind: "emit", start: 3400, end: 3700 },
  { id: "S1b", label: "sub1: summarize", kind: "dispatch", start: 3700, end: 5900, isl: 2100, osl: 300 },
  { id: "S2a", label: "sub2: reason", kind: "dispatch", start: 1800, end: 3300, isl: 1500, osl: 220 },
  { id: "S2t", label: "sub2: Read", kind: "emit", start: 3300, end: 3600 },
  { id: "S2b", label: "sub2: summarize", kind: "dispatch", start: 3600, end: 6000, isl: 2000, osl: 280 },
  { id: "S3a", label: "sub3: reason", kind: "dispatch", start: 1800, end: 3500, isl: 1700, osl: 260 },
  { id: "S3t", label: "sub3: WebFetch", kind: "emit", start: 3500, end: 8200 },
  { id: "S3b", label: "sub3: summarize", kind: "dispatch", start: 8300, end: 10900, isl: 2600, osl: 340 },
  { id: "B1a", label: "bg: scan", kind: "dispatch", detached: true, start: 1800, end: 3000, isl: 900, osl: 120 },
  { id: "M2", label: "main: synthesize", kind: "dispatch", start: 6100, end: 9700, isl: 3200, osl: 520 },
  { id: "M2t", label: "main: Edit", kind: "emit", start: 9700, end: 10600 },
  { id: "M3", label: "main: report", kind: "dispatch", start: 11000, end: 15200, isl: 4100, osl: 640 },
];

export type TraceEdgeDef = { from: string; to: string; detached?: boolean };

export const TRACE_EDGES: TraceEdgeDef[] = [
  { from: "START", to: "M1" },
  { from: "M1", to: "S1a" },
  { from: "M1", to: "S2a" },
  { from: "M1", to: "S3a" },
  { from: "M1", to: "B1a", detached: true },
  { from: "S1a", to: "S1t" },
  { from: "S1t", to: "S1b" },
  { from: "S2a", to: "S2t" },
  { from: "S2t", to: "S2b" },
  { from: "S3a", to: "S3t" },
  { from: "S3t", to: "S3b" },
  { from: "S1b", to: "M2" },
  { from: "S2b", to: "M2" },
  { from: "M2", to: "M2t" },
  { from: "M2t", to: "M3" },
  { from: "S3b", to: "M3" },
];

export type TraceFrame = { t: number; desc: string };

// Frames are sampled instants along the shared clock, chosen so each turn is observed mid-flight
// for at least a couple of steps and concurrent requests visibly overlap.
export const TRACE_FRAMES: TraceFrame[] = [
  { t: 0, desc: "A user task arrives. The main-agent plan turn (M1) is queued as the entry Step." },
  {
    t: 900,
    desc: "M1 is in flight — the main agent streams its plan and will call Task to launch three explore subagents plus a detached background scan.",
  },
  { t: 1500, desc: "M1 is still streaming (~1.3s into a 1.4s turn). One request outstanding." },
  {
    t: 2000,
    desc: "M1 returned. sub1, sub2, sub3 AND the background agent were all issued at the SAME instant — four concurrent LLM requests are now in flight.",
  },
  { t: 2700, desc: "The four first turns keep streaming in parallel; there are no edges between them." },
  {
    t: 3600,
    desc: "The reason turns returned. Each subagent fired a tool (Emit): sub1 Grep, sub2 Read, sub3 WebFetch. No LLM request is in flight right now — only tools. The background turn is done and ignored.",
  },
  {
    t: 4500,
    desc: "sub1 Grep and sub2 Read resolved fast, so sub1 and sub2 are streaming their summarize turns (2 concurrent requests). sub3's WebFetch is still running.",
  },
  { t: 5400, desc: "sub1 and sub2 summarize turns keep streaming; sub3 is still fetching." },
  {
    t: 6050,
    desc: "sub1 and sub2 returned — the synthesize gate (M2) is satisfied 2/2. sub3's WebFetch is still in flight, so sub3 will miss this join.",
  },
  { t: 7000, desc: "M2 is in flight — the main agent synthesizes sub1 + sub2. sub3 has NOT joined; its slow fetch continues." },
  {
    t: 8600,
    desc: "sub3's WebFetch finally resolved, so sub3 is streaming its summarize turn — now M2 and sub3 are both in flight at once.",
  },
  { t: 9600, desc: "M2 and sub3's summarize both keep streaming (two overlapping requests)." },
  { t: 10200, desc: "M2 returned and called Edit (Emit), which feeds M3's chain input. sub3 is still summarizing." },
  {
    t: 11000,
    desc: "The Edit output and sub3's late result both arrived (M3 gate 2/2). M3 is in flight — the report AND-joins the chain output with sub3's late merge.",
  },
  { t: 13000, desc: "M3 keeps streaming the final report." },
  { t: 15500, desc: "M3 returned. The conversation is complete; the detached background agent was never joined." },
];

export const M2_INPUTS = ["S1b", "S2b"];
export const M3_INPUTS = ["M2t", "S3b"];

const MODEL = "meta-llama/Llama-3.1-70B-Instruct";

const NODE_BY_ID = new Map(TRACE_NODES.map((n) => [n.id, n]));
const PREDS_OF = new Map<string, string[]>();
for (const e of TRACE_EDGES) {
  PREDS_OF.set(e.to, [...(PREDS_OF.get(e.to) ?? []), e.from]);
}

/** Node state as of wall-clock `t`, derived purely from each node's `[start, end]` window. */
export function nodeStateOf(t: number, id: string): NodeState {
  const n = NODE_BY_ID.get(id)!;
  if (id === "START") return "done";
  if (t >= n.end) return "done";
  if (t >= n.start) return "firing";
  const preds = PREDS_OF.get(id) ?? [];
  const inputsReady = preds.every((p) => t >= NODE_BY_ID.get(p)!.end);
  return inputsReady ? "ready" : "pending";
}

/** Count of `ids` whose node has completed (`t >= end`) — drives the two AND-fan-in gate bars. */
export function arrivedCount(t: number, ids: string[]): number {
  return ids.filter((id) => t >= NODE_BY_ID.get(id)!.end).length;
}

/** Count of completed nodes of a given `kind`, excluding the synthetic START node. */
export function doneCountByKind(t: number, kind: Kind): number {
  return TRACE_NODES.filter((n) => n.id !== "START" && n.kind === kind && t >= n.end).length;
}

type ReqState = "in-flight" | "done";

function reqStateOf(t: number, n: TraceNodeDef): ReqState | null {
  if (t >= n.end) return "done";
  if (t >= n.start) return "in-flight";
  return null;
}

// Dispatch turns (excluding START) are the only nodes that issue an LLM API call, listed in issue
// order so the log grows downward as turns fire.
const REQUEST_NODES = TRACE_NODES.filter((n) => n.id !== "START" && n.kind === "dispatch").sort(
  (a, b) => a.start - b.start,
);

// --- diagram layout --------------------------------------------------------

// Manual left-to-right layout: rank (x) tracks pipeline stage, lane (y) tracks which
// subagent/background thread a node belongs to. No dagre-style auto layout helper exists in this
// app, so positions are hand-placed like `OverviewPage.tsx`/`DispatchPage.tsx`.
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  START: { x: 0, y: 150 },
  M1: { x: 220, y: 150 },
  S1a: { x: 460, y: 0 },
  S2a: { x: 460, y: 140 },
  S3a: { x: 460, y: 280 },
  B1a: { x: 460, y: 420 },
  S1t: { x: 700, y: 0 },
  S2t: { x: 700, y: 140 },
  S3t: { x: 700, y: 280 },
  S1b: { x: 940, y: 0 },
  S2b: { x: 940, y: 140 },
  S3b: { x: 940, y: 280 },
  M2: { x: 1180, y: 70 },
  M2t: { x: 1420, y: 70 },
  M3: { x: 1660, y: 175 },
};

const NODE_STYLE_CLASSES: Record<Kind, Record<NodeState, string>> = {
  dispatch: {
    pending: "!border-stroke-tertiary !bg-surface-panel",
    ready: "!border-accent-primary !bg-surface-elevated",
    firing: "!border-2 !border-accent-primary !bg-surface-elevated",
    done: "!border-stroke-secondary !bg-surface-elevated",
  },
  emit: {
    pending: "!border-dashed !border-stroke-tertiary !bg-surface-panel",
    ready: "!border-dashed !border-stroke-tertiary !bg-surface-panel",
    firing: "!border-2 !border-dashed !border-stroke-secondary !bg-surface-panel",
    done: "!border-dashed !border-stroke-tertiary !bg-surface-panel",
  },
};

const STATE_LABEL: Record<NodeState, string> = {
  pending: "pending",
  ready: "ready",
  firing: "firing",
  done: "done (already ran)",
};

function buildDiagram(t: number): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = TRACE_NODES.map((def) => {
    const pos = NODE_POSITIONS[def.id];
    const state = nodeStateOf(t, def.id);
    if (def.id === "START") {
      return {
        id: def.id,
        type: "card",
        position: pos,
        data: { title: "START", detail: "entry", className: NODE_STYLE_CLASSES.dispatch.done },
      };
    }
    const tag = `${def.kind === "emit" ? "Emit · tool" : "Dispatch · LLM"}${def.detached ? " · detached" : ""}`;
    return {
      id: def.id,
      type: "card",
      position: pos,
      data: {
        title: def.label,
        subtitle: tag,
        detail: STATE_LABEL[state],
        className: NODE_STYLE_CLASSES[def.kind][state],
      },
    };
  });

  const edges: Edge[] = TRACE_EDGES.map((e) => {
    const active = nodeStateOf(t, e.from) === "done" && ["firing", "ready"].includes(nodeStateOf(t, e.to));
    const id = `${e.from}-${e.to}`;
    if (active) {
      return {
        id,
        source: e.from,
        target: e.to,
        type: "flow",
        data: { speed: "fast" },
      };
    }
    return {
      id,
      source: e.from,
      target: e.to,
      style: {
        stroke: "var(--color-ink-quaternary)",
        strokeWidth: 1.25,
        strokeDasharray: e.detached ? "4 4" : undefined,
      },
    };
  });

  return { nodes, edges };
}

// Module-level (stable identity) so `useElkLayout` never re-lays-out from an inline options object.
// The graph structure (node ids + edge ids) is fixed across steps — only node styling/data changes —
// so ELK computes positions once and they hold while the step clock advances.
const GRAPH_LAYOUT: ElkOptions = { direction: "RIGHT" };

/** Auto-laid-out conversation graph. Runs the ELK hook inside its own provider (per-instance). */
function ConversationGraph({ nodes: inputNodes, edges }: { nodes: Node[]; edges: Edge[] }): React.JSX.Element {
  const { nodes, laidOut } = useElkLayout(inputNodes, edges, GRAPH_LAYOUT);
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      nodesDraggable={false}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

// --- in-flight request sidebar ---------------------------------------------

function RequestLog({ t }: { t: number }): React.JSX.Element {
  const rows = REQUEST_NODES.map((n) => ({ n, st: reqStateOf(t, n) })).filter(
    (x): x is { n: TraceNodeDef; st: ReqState } => x.st !== null,
  );
  const live = rows.filter((r) => r.st === "in-flight").length;

  return (
    <Stack gap={8}>
      <Row align="center" gap={8}>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>POST /v1/chat/completions</span>
        <span className={`ml-auto text-xs ${live > 0 ? inkClassName("secondary") : inkClassName("quaternary")}`}>
          {live} live
        </span>
      </Row>
      {rows.length === 0 ? (
        <span className={`text-xs ${inkClassName("tertiary")}`}>no requests issued yet</span>
      ) : (
        rows.map(({ n, st }) => {
          const isLive = st === "in-flight";
          const total = n.end - n.start;
          const elapsed = Math.max(0, Math.min(total, t - n.start));
          const pct = total > 0 ? Math.round((elapsed / total) * 100) : 100;
          return (
            <div
              key={n.id}
              className={`rounded-lg px-3 py-2 shadow-sm ${isLive ? "border-2 border-accent-primary" : "border border-stroke-secondary"} ${
                isLive ? surfaceClassName("elevated") : surfaceClassName("panel")
              }`}
            >
              <Stack gap={4}>
                <Row gap={8} align="center">
                  <span className={`text-xs font-semibold ${isLive ? inkClassName("primary") : inkClassName("tertiary")}`}>
                    {n.label}
                  </span>
                  <span className="ml-auto" />
                  {isLive ? (
                    <span className="text-xs font-semibold text-accent-primary">{(elapsed / 1000).toFixed(1)}s</span>
                  ) : (
                    <span className={`text-xs ${inkClassName("tertiary")}`}>200 OK</span>
                  )}
                </Row>
                {isLive ? (
                  <div className={`h-1 overflow-hidden rounded-md ${surfaceClassName("chrome")}`}>
                    <div className="h-full bg-accent-primary" style={{ width: `${pct}%` }} />
                  </div>
                ) : null}
                <Row gap={8} align="center" wrap>
                  <span className={`text-xs ${inkClassName("quaternary")}`}>req_{n.id.toLowerCase()}</span>
                  <span className="ml-auto" />
                  <span className={`text-xs ${inkClassName("tertiary")}`}>
                    in {(n.isl ?? 0).toLocaleString()} · out {isLive ? `→ ${n.osl}` : (n.osl ?? 0).toLocaleString()} tok
                  </span>
                </Row>
              </Stack>
            </div>
          );
        })
      )}
    </Stack>
  );
}

// --- AND-fan-in gate bar -----------------------------------------------------

function GateBar({
  title,
  sub,
  have,
  need,
  releasedNote,
  waitNote,
}: {
  title: string;
  sub: string;
  have: number;
  need: number;
  releasedNote: string;
  waitNote: string;
}): React.JSX.Element {
  const pct = need > 0 ? Math.round((have / need) * 100) : 100;
  return (
    <Stack gap={6}>
      <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</span>
      <Row align="center" gap={8}>
        <span className={`text-xs ${inkClassName("tertiary")}`}>{sub}</span>
        <span className={`ml-auto text-xs ${inkClassName("tertiary")}`}>
          {have} / {need} arrived
        </span>
      </Row>
      <div className={`h-2 overflow-hidden rounded-md ${surfaceClassName("chrome")}`}>
        <div className="h-full bg-accent-primary" style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs ${inkClassName("tertiary")}`}>{have < need ? waitNote : releasedNote}</span>
    </Stack>
  );
}

// --- main deck ---------------------------------------------------------------

/**
 * Claude Code subagent flow explainer: a step-through trace of a main agent that fans out three
 * subagents plus a detached background agent at the same instant, where two subagents join an
 * early AND-fan-in and the third (slowed by a tool call) merges into a later join instead.
 *
 * Ported from `docs/canvases/claude-code-subagent-stepper.canvas.tsx`, which has no in-deck page
 * tabs, so this is a single component file rather than a `PageTabs` composition.
 */
export function ClaudeCodeSubagentStepperDeck(): React.JSX.Element {
  const sim = useStepSimulator(TRACE_FRAMES, { autoPlayMs: 1400 });
  const index = sim.index;
  const frame = sim.current ?? TRACE_FRAMES[0];
  const t = frame.t;

  function gotoStep(target: number): void {
    const delta = target - sim.index;
    // `next()`/`back()` each schedule one state update rather than mutating `sim` in place, so a
    // loop keyed off a live/stale `sim.index` inside the loop body would misbehave. `delta` is
    // computed once from the index at click time, then applied as a fixed, bounded number of
    // calls — the same pattern `PoolPage.tsx`'s "Run all" uses for a bounded batch of `next()`s.
    if (delta > 0) {
      for (let i = 0; i < delta; i++) sim.next();
    } else if (delta < 0) {
      for (let i = 0; i < -delta; i++) sim.back();
    }
  }

  const { nodes, edges } = useMemo(() => buildDiagram(t), [t]);
  const dispatches = doneCountByKind(t, "dispatch");
  const emits = doneCountByKind(t, "emit");
  const m2Have = arrivedCount(t, M2_INPUTS);
  const m3Have = arrivedCount(t, M3_INPUTS);

  return (
    <div className={`min-h-full p-6 ${surfaceClassName("page")}`}>
      <Stack gap={20} className="mx-auto max-w-6xl 2xl:max-w-[1728px]">
        <Stack gap={8}>
          <Row align="center" gap={10} wrap>
            <h1 className={`text-xl font-bold ${inkClassName("primary")}`}>
              Claude Code subagent flow — concurrent spawn, staggered join
            </h1>
          </Row>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            The main agent launches sub1, sub2, and sub3 at the same instant. sub1 and sub2 finish
            fast and join the <strong>synthesize</strong> turn (M2). sub3 runs a slow WebFetch,
            misses that join, and instead <strong>merges into the later report</strong> turn (M3) —
            two AND-fan-ins at different points in the same trace. <strong>Dispatch</strong> turns
            are solid and accent-colored; <strong>Emit</strong> tool steps are dashed and muted,
            because they are not real agent turns. The right rail tracks every LLM request on a
            shared clock, so a request stays in-flight for its whole latency window and concurrent
            calls overlap. Use Next/Prev or click a step.
          </p>
        </Stack>

        <Stack gap={10}>
          <Row gap={8} align="center" wrap>
            <Button variant="secondary" disabled={sim.isFirst} onClick={sim.back}>
              Prev
            </Button>
            <Button variant="primary" disabled={sim.isLast} onClick={sim.next}>
              Next
            </Button>
            <Button variant="ghost" disabled={sim.isFirst} onClick={sim.reset}>
              Reset
            </Button>
            <span className={`ml-auto text-xs ${inkClassName("tertiary")}`}>
              step {index + 1} / {TRACE_FRAMES.length} · t = {(t / 1000).toFixed(1)}s
            </span>
          </Row>
          <Row gap={6} wrap>
            {TRACE_FRAMES.map((_, i) => (
              <button
                key={i}
                type="button"
                aria-pressed={i === index}
                aria-label={`step ${i + 1}`}
                onClick={() => gotoStep(i)}
                className={
                  i === index
                    ? "rounded-full border border-accent-primary bg-accent-primary px-2.5 py-0.5 text-xs font-semibold text-white"
                    : `rounded-full border px-2.5 py-0.5 text-xs font-medium ${strokeClassName("secondary")} ${inkClassName("secondary")}`
                }
              >
                {i + 1}
              </button>
            ))}
          </Row>
        </Stack>

        <Callout tone="info" title={`Step ${index + 1} · t = ${(t / 1000).toFixed(1)}s`}>
          {frame.desc}
        </Callout>

        <Grid columns="1fr minmax(0, 300px)" gap={20} align="start">
          <Stack gap={16} className="min-w-0">
            <div className={`border ${strokeClassName("secondary")} rounded-lg shadow-sm`}>
              <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
                Conversation graph
              </div>
              <div style={{ height: 520 }}>
                <ReactFlowProvider>
                  <ConversationGraph nodes={nodes} edges={edges} />
                </ReactFlowProvider>
              </div>
            </div>

            <Grid columns={2} gap={12}>
              <Stat value={dispatches} label="LLM dispatches (credit)" />
              <Stat value={emits} label="Tool emits (no credit)" />
            </Grid>

            <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Two AND-fan-in gates</h2>
            <Grid columns={2} gap={20}>
              <GateBar
                title="M2 · synthesize (early join)"
                sub="await_inputs: sub1_out, sub2_out"
                have={m2Have}
                need={M2_INPUTS.length}
                waitNote="Parked until sub1 and sub2 resolve. sub3 is still running and is not an input here."
                releasedNote="sub1 + sub2 resolved — M2 fired without waiting for sub3."
              />
              <GateBar
                title="M3 · report (late join)"
                sub="await_inputs: main_edit_out, sub3_out"
                have={m3Have}
                need={M3_INPUTS.length}
                waitNote="Waiting on the Edit output (chain) and sub3's late summarize result."
                releasedNote="Edit output + sub3's late result both arrived — the report turn fires."
              />
            </Grid>

            <Callout tone="info" title="Same spawn instant, different join turns">
              sub3 is spawned concurrently with sub1/sub2 from M1, so all three share the
              START-relative fan-out. Whether a subagent joins M2 or M3 is decided by{" "}
              <strong>completed-before</strong> timing: a cause becomes an edge into the first
              later turn that started after the cause finished. sub3&apos;s slow WebFetch pushes
              its completion past M2, so its edge lands on M3. The detached background agent, by
              contrast, joins nothing.
            </Callout>

            <div className={`border-t ${strokeClassName("secondary")}`} />
            <Stack gap={10}>
              <Row gap={16} wrap align="center">
                <span className={`min-w-24 text-xs font-medium ${inkClassName("tertiary")}`}>node kind</span>
                <Row gap={6} align="center">
                  <div className="h-3.5 w-5 rounded-md border border-accent-primary bg-surface-elevated" />
                  <span className={`text-xs ${inkClassName("secondary")}`}>Dispatch — real agent turn (LLM, credit)</span>
                </Row>
                <Row gap={6} align="center">
                  <div className={`h-3.5 w-5 rounded-md border border-dashed ${strokeClassName("tertiary")} ${surfaceClassName("panel")}`} />
                  <span className={`text-xs ${inkClassName("secondary")}`}>Emit — tool call, not a turn (no credit)</span>
                </Row>
              </Row>
              <Row gap={16} wrap align="center">
                <span className={`min-w-24 text-xs font-medium ${inkClassName("tertiary")}`}>state</span>
                <Row gap={6} align="center">
                  <div className="h-3.5 w-5 rounded-md border-2 border-accent-primary bg-surface-elevated" />
                  <span className={`text-xs ${inkClassName("secondary")}`}>firing (thick border)</span>
                </Row>
                <Row gap={6} align="center">
                  <div className={`h-3.5 w-5 rounded-md border ${strokeClassName("secondary")} ${surfaceClassName("panel")}`} />
                  <span className={`text-xs ${inkClassName("secondary")}`}>already ran</span>
                </Row>
                <Row gap={6} align="center">
                  <div className={`h-3.5 w-5 rounded-md border ${strokeClassName("tertiary")} ${surfaceClassName("panel")}`} />
                  <span className={`text-xs ${inkClassName("secondary")}`}>pending</span>
                </Row>
                <Row gap={6} align="center">
                  <div
                    className="h-0 w-4 border-t border-dashed"
                    style={{ borderColor: "var(--color-ink-quaternary)" }}
                  />
                  <span className={`text-xs ${inkClassName("secondary")}`}>detached spawn (not joined)</span>
                </Row>
              </Row>
            </Stack>
          </Stack>

          <div>
            <div className={`border ${strokeClassName("secondary")} rounded-lg shadow-sm`}>
              <div className={`border-b px-4 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`}>
                In-flight requests
              </div>
              <div className="p-3">
                <span className={`mb-2 block text-xs ${inkClassName("quaternary")}`}>{MODEL}</span>
                <RequestLog t={t} />
              </div>
            </div>
          </div>
        </Grid>
      </Stack>
    </div>
  );
}
