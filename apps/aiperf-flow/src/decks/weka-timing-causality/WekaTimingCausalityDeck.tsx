/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports the real Cursor canvas `weka-timing-causality.canvas.tsx` onto aiperf-flow's component
//! vocabulary. The weka trace is lossy — timestamps, KV-block hashes, and subagent markers, but no
//! causal graph — so causality is reconstructed purely from interval timing: each recorded request
//! is an interval on the warped clock, and `A -> B` is drawn only when A provably finished before B
//! began. This single-view deck keeps the source's interactive interval-order lab (slider + async
//! toggle re-deriving a frontier-reduced DAG live), the zero-duration rank tie-break demo, the
//! content-contract role-segmentation comparison, the timing/content data-flow pipeline, and the
//! locked-regression table.

import { useState } from "react";
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
import { Table } from "../../prose/Table.js";
import { Stat } from "../../prose/Stat.js";
import { Legend } from "../../prose/Legend.js";
import { Code } from "../../prose/Code.js";
import { Toggle } from "../../prose/Toggle.js";
import { Button } from "../../prose/Button.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";
import { Eyebrow } from "../../prose/Eyebrow.js";

// --- interval-order logic (ported verbatim from the source canvas) ----------

// A recorded request as its interval on the warped clock.
type IvNode = {
  id: string;
  label: string;
  start: number;
  end: number;
  kind: "parent" | "agent" | "spawn";
};

// The worked example from the design (§ "Worked example"). C0's start is the
// one adjustable knob; its 1.8s duration is preserved as it slides.
const C0_DUR = 1.8;
const BASE: IvNode[] = [
  { id: "P0", label: "parent", start: 0, end: 1.0, kind: "parent" },
  { id: "A0", label: "Explore #1", start: 1.2, end: 4.0, kind: "agent" },
  { id: "B0", label: "Explore #2", start: 1.3, end: 5.0, kind: "agent" },
  { id: "C0", label: "spawned", start: 5.2, end: 5.2 + C0_DUR, kind: "spawn" },
  { id: "P1", label: "parent resume", start: 7.5, end: 8.0, kind: "parent" },
];

// SVG stroke/fill colors, sourced from the theme CSS custom properties so no raw hex leaks in.
const KIND_VAR: Record<IvNode["kind"], string> = {
  parent: "var(--color-category-blue)",
  agent: "var(--color-category-green)",
  spawn: "var(--color-category-purple)",
};
const ASYNC_VAR = "var(--color-category-orange)";

function kindVar(kind: IvNode["kind"], async: boolean): string {
  return async ? ASYNC_VAR : KIND_VAR[kind];
}

type DerivedEdge = { from: string; to: string; delayMs: number; binding: boolean };

// Interval-order edge derivation, faithful to the design:
//   A -> B iff A.end <= B.start AND rank(A) < rank(B), rank = sort(start,end,id)
//   frontier reduction keeps only the maximal finished-before set
//   async-launched leaves are excluded as out-of-subtree predecessors
//   binding cause = latest-ending frontier pred; carries the firing delay
function deriveIntervalOrder(nodes: IvNode[], asyncIds: Set<string>) {
  const sorted = [...nodes].sort(
    (a, b) => a.start - b.start || a.end - b.end || (a.id < b.id ? -1 : 1),
  );
  const rank = new Map(sorted.map((n, i) => [n.id, i]));
  const rk = (id: string) => rank.get(id)!;
  const edges: DerivedEdge[] = [];
  const roots = new Set<string>();

  for (const B of nodes) {
    let completed = nodes.filter(
      (A) => A.id !== B.id && A.end <= B.start && rk(A.id) < rk(B.id),
    );
    // Exclusion: an async leaf is dropped as a predecessor of a target that
    // does not share its async subtree (here: any non-async target).
    completed = completed.filter((A) => !(asyncIds.has(A.id) && !asyncIds.has(B.id)));
    // Frontier reduction: drop any A that finished before another completed A2.
    const frontier = completed.filter(
      (A) => !completed.some((A2) => A2.id !== A.id && A.end <= A2.start && rk(A.id) < rk(A2.id)),
    );
    if (frontier.length === 0) {
      roots.add(B.id);
      continue;
    }
    const binding = frontier.reduce((a, b) => (b.end > a.end ? b : a));
    for (const f of frontier) {
      edges.push({
        from: f.id,
        to: B.id,
        delayMs: f.id === binding.id ? Math.max(0, B.start - binding.end) * 1000 : 0,
        binding: f.id === binding.id,
      });
    }
  }
  return { edges, roots, rank };
}

function fmt(n: number): string {
  return Number.isInteger(n) ? `${n}` : n.toFixed(1);
}

// --- interval timeline (a Gantt-style chart, scoped to this deck) -----------

function IntervalTimeline({
  nodes,
  asyncIds,
  rank,
}: {
  nodes: IvNode[];
  asyncIds: Set<string>;
  rank: Map<string, number>;
}): React.JSX.Element {
  const maxEnd = Math.max(...nodes.map((n) => n.end)) + 0.5;
  const LEFT = 78;
  const px = Math.max(40, Math.min(78, Math.floor((720 - LEFT) / maxEnd)));
  const x = (t: number) => LEFT + t * px;
  const rowH = 26;
  const rowGap = 8;
  const top = 12;
  const width = x(maxEnd) + 16;
  const bottom = top + nodes.length * (rowH + rowGap);
  const svgH = bottom + 26;
  const gridMax = Math.ceil(maxEnd);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={width} height={svgH} role="img" aria-label="Intervals on the warped clock">
        {Array.from({ length: gridMax + 1 }, (_, t) => (
          <g key={`g-${t}`}>
            <line
              x1={x(t)}
              y1={top - 4}
              x2={x(t)}
              y2={bottom}
              stroke="var(--color-stroke-tertiary)"
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={bottom + 14}
              textAnchor="middle"
              fontSize={10}
              fill="var(--color-ink-tertiary)"
            >
              {t}s
            </text>
          </g>
        ))}
        {nodes.map((n, i) => {
          const y = top + i * (rowH + rowGap);
          const isAsync = asyncIds.has(n.id);
          const c = kindVar(n.kind, isAsync);
          const w = Math.max((n.end - n.start) * px, 12);
          return (
            <g key={n.id}>
              <text x={8} y={y + rowH / 2 + 4} fontSize={11} fill="var(--color-ink-secondary)" fontWeight={600}>
                {n.id}
              </text>
              <rect
                x={x(n.start)}
                y={y}
                width={w}
                height={rowH}
                rx={0}
                fill="var(--color-surface-panel)"
                stroke={c}
                strokeWidth={isAsync ? 2 : 1.5}
                strokeDasharray={isAsync ? "5 3" : undefined}
              />
              <text x={x(n.start) + 8} y={y + rowH / 2 + 4} fontSize={10.5} fill="var(--color-ink-primary)" fontWeight={600}>
                {n.label}
              </text>
              <circle cx={x(n.end)} cy={y + rowH / 2} r={8} fill="var(--color-surface-page)" stroke={c} strokeWidth={1.5} />
              <text x={x(n.end)} y={y + rowH / 2 + 3.5} textAnchor="middle" fontSize={9} fill="var(--color-ink-secondary)" fontWeight={700}>
                {rank.get(n.id)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// --- derived causality DAG (live React Flow graph) --------------------------

// Longest-path layering: START at depth 0, roots at depth 1, every other node one deeper than its
// latest-ending predecessor. Positions are recomputed each render from the derived edges, so the
// graph re-lays-out live as the slider/toggle change the interval order.
function layoutDag(
  ids: string[],
  roots: Set<string>,
  edges: DerivedEdge[],
): { nodes: Node[]; edges: Edge[] } {
  const depth = new Map<string, number>();
  depth.set("START", 0);
  ids.forEach((id) => depth.set(id, 1));
  for (let guard = 0; guard < 20; guard++) {
    let changed = false;
    for (const e of edges) {
      const d = (depth.get(e.from) ?? 1) + 1;
      if (d > (depth.get(e.to) ?? 1)) {
        depth.set(e.to, d);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const byDepth = new Map<number, string[]>();
  for (const id of ["START", ...ids]) {
    const d = depth.get(id) ?? 1;
    const list = byDepth.get(d) ?? [];
    list.push(id);
    byDepth.set(d, list);
  }

  const rfNodes: Node[] = ["START", ...ids].map((id) => {
    const d = depth.get(id) ?? 1;
    const idx = (byDepth.get(d) ?? []).indexOf(id);
    const iv = BASE.find((n) => n.id === id);
    return {
      id,
      type: "panel",
      position: { x: d * 190 + 10, y: idx * 78 + 10 },
      data:
        id === "START"
          ? { title: "START", detail: "root" }
          : { title: id, detail: iv?.label },
    } satisfies Node;
  });

  const rfEdges: Edge[] = [
    ...[...roots].map((r) => ({
      id: `START-${r}`,
      source: "START",
      target: r,
      style: { stroke: "var(--color-ink-tertiary)", strokeDasharray: "4 3" },
    })),
    ...edges.map((e) =>
      e.binding
        ? {
            id: `${e.from}-${e.to}`,
            source: e.from,
            target: e.to,
            type: "flow" as const,
            label: e.delayMs > 0 ? `+${fmt(e.delayMs)}ms` : undefined,
          }
        : {
            id: `${e.from}-${e.to}`,
            source: e.from,
            target: e.to,
            style: { stroke: "var(--color-ink-tertiary)", strokeDasharray: "4 3" },
          },
    ),
  ];

  return { nodes: rfNodes, edges: rfEdges };
}

function CausalityLab(): React.JSX.Element {
  const [c0start, setC0] = useState<number>(5.2);
  const [bAsync, setBAsync] = useState<boolean>(false);

  const nodes: IvNode[] = BASE.map((n) =>
    n.id === "C0" ? { ...n, start: c0start, end: c0start + C0_DUR } : n,
  );
  const asyncIds = new Set<string>(bAsync ? ["B0"] : []);
  const { edges, roots, rank } = deriveIntervalOrder(nodes, asyncIds);

  const c0Preds = edges.filter((e) => e.to === "C0").map((e) => e.from);
  const b0OverlapsC0 = 1.3 < c0start + C0_DUR && c0start < 5.0; // B0 [1.3,5.0]

  const dag = layoutDag(
    nodes.map((n) => n.id),
    roots,
    edges,
  );

  const edgeRows = nodes.map((n) => {
    const preds = edges.filter((e) => e.to === n.id);
    const binding = preds.find((p) => p.binding);
    return {
      node: n.id,
      frontier: roots.has(n.id) ? "START (root)" : preds.map((p) => p.from).join(" + "),
      binding: binding ? binding.from : "—",
      delay: roots.has(n.id)
        ? `min_start ${fmt(n.start * 1000)}ms`
        : binding
          ? `${fmt(binding.delayMs)}ms`
          : "0ms",
    };
  });

  return (
    <Stack gap={14}>
      <div className={clsxBorder()}>
        <div className={borderHeader()}>
          Controls — {bAsync ? "B0 async-launched" : "all blocking"}
        </div>
        <div className="p-3">
          <Stack gap={12}>
            <Row gap={12} align="center" wrap>
              <span className={`w-24 text-sm font-semibold ${inkClassName("primary")}`}>C0 start</span>
              <input
                type="range"
                min={2}
                max={6}
                step={0.1}
                value={c0start}
                aria-label="C0 start"
                onChange={(e) => setC0(Number(e.target.value))}
                style={{ width: 260, accentColor: "var(--color-accent-primary)" }}
              />
              <span className={`text-sm ${inkClassName("secondary")}`}>{fmt(c0start)}s</span>
              <div className="flex-1" />
              <Button
                variant="ghost"
                onClick={() => {
                  setC0(5.2);
                  setBAsync(false);
                }}
              >
                Reset
              </Button>
            </Row>
            <Row gap={12} align="center" wrap>
              <span className={`w-24 text-sm font-semibold ${inkClassName("primary")}`}>B0 async</span>
              <Toggle checked={bAsync} onChange={setBAsync} />
              <span className={`text-sm ${inkClassName("tertiary")}`}>
                fire-and-forget: exclude B0 as an out-of-subtree predecessor
              </span>
            </Row>
          </Stack>
        </div>
      </div>

      <Grid columns={4} gap={12}>
        <Stat value={nodes.length} label="requests" />
        <Stat value={edges.length} label="edges (reduced)" />
        <Stat value={roots.size} label="START roots" />
        <Stat
          value={c0Preds.length}
          label="C0 fan-in width"
          tone={c0Preds.length >= 2 ? "positive" : "neutral"}
        />
      </Grid>

      <Callout
        tone={b0OverlapsC0 ? "warning" : "info"}
        title={
          b0OverlapsC0
            ? "C0 overlaps B0 — concurrent, no edge"
            : "C0 starts after A0 and B0 — AND-join"
        }
      >
        {b0OverlapsC0
          ? "B0 is still running when C0 starts, so B0 -> C0 is dropped (overlap = concurrent). C0 now joins only the predecessors that had finished."
          : bAsync
            ? "B0 is async-launched, so it never serializes C0 even though it finished first. C0 AND-joins the remaining finished-before frontier."
            : "Both A0 and B0 finished before C0 started, so C0 AND-joins both with no parent turn required (fan-in k = 2)."}
      </Callout>

      <Grid columns="minmax(0, 1fr) minmax(0, 360px)" gap={16} align="start">
        <div className={clsxBorder()}>
          <div className={borderHeader()}>Intervals on the warped clock — rank badge on each end</div>
          <div className="p-3">
            <IntervalTimeline nodes={nodes} asyncIds={asyncIds} rank={rank} />
          </div>
        </div>
        <div className={clsxBorder()}>
          <div className={borderHeader()}>Derived causality DAG — accent = binding cause</div>
          <div style={{ height: 340 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={dag.nodes}
              edges={dag.edges}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              proOptions={{ hideAttribution: true }}
              nodesDraggable={false}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
            </ReactFlow>
          </div>
        </div>
      </Grid>

      <Table
        columns={[
          { key: "node", label: "Node" },
          { key: "frontier", label: "Finished-before frontier" },
          { key: "binding", label: "Binding cause" },
          { key: "delay", label: "delay_after_pred", align: "end" },
        ]}
        rows={edgeRows}
      />
    </Stack>
  );
}

// --- rank / zero-duration 2-cycle demo (React Flow) -------------------------

function RankDemo(): React.JSX.Element {
  const [useRank, setUseRank] = useState<boolean>(true);
  const cyc = !useRank;

  const rfNodes: Node[] = [
    { id: "X", type: "panel", position: { x: 0, y: 40 }, data: { title: "X [3,3]" } },
    { id: "Y", type: "panel", position: { x: 230, y: 40 }, data: { title: "Y [3,3]" } },
  ];
  const rfEdges: Edge[] = [
    {
      id: "x-y",
      source: "X",
      target: "Y",
      type: "flow",
      data: { color: cyc ? "var(--color-category-red)" : "var(--color-accent-primary)" },
    },
    ...(cyc
      ? [
          {
            id: "y-x",
            source: "Y",
            target: "X",
            type: "flow" as const,
            data: { color: "var(--color-category-red)" },
          },
        ]
      : []),
  ];

  return (
    <div className={clsxBorder()}>
      <div className={borderHeader()}>
        <Row gap={8} align="center" justify="space-between">
          <span>Two coincident zero-duration requests (same t, api_time = None)</span>
          <Row gap={8} align="center">
            <span className={`text-xs ${inkClassName("tertiary")}`}>rank tie-break</span>
            <Toggle checked={useRank} onChange={setUseRank} />
          </Row>
        </Row>
      </div>
      <div className="p-3">
        <Stack gap={10}>
          <div style={{ height: 160 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={rfNodes}
              edges={rfEdges}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              proOptions={{ hideAttribution: true }}
              nodesDraggable={false}
            />
          </div>
          <div
            className={`text-sm font-semibold ${inkClassName("secondary")}`}
            style={cyc ? { color: "var(--color-category-red)" } : undefined}
          >
            {cyc ? "2-cycle: await_inputs deadlock" : "single edge: X -> Y (rank by node_id)"}
          </div>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            The naive predicate <Code inline>A.end &le; B.start</Code> holds both ways for coincident
            zero-duration intervals, so it is not antisymmetric — no DAG. The total-order tie-break{" "}
            <Code inline>rank = sort(start, end, node_id)</Code> resolves the exact tie to exactly one
            direction, so the cycle is prevented at build time (not merely caught by the executor).
          </p>
        </Stack>
      </div>
    </div>
  );
}

// --- content contract: frozen per-block tags --------------------------------

type BlockRole = "user" | "assistant";

function BlockStrip({ roles, diverge }: { roles: BlockRole[]; diverge: number }): React.JSX.Element {
  return (
    <div className="flex flex-wrap gap-[2px]">
      {roles.map((r, i) => (
        <span
          key={i}
          className="inline-block h-4 w-[13px]"
          style={{
            background: r === "user" ? "var(--color-category-blue)" : "var(--color-category-purple)",
            opacity: i === diverge ? 1 : 0.55,
            outline: i === diverge ? "2px solid var(--color-category-orange)" : undefined,
          }}
        />
      ))}
    </div>
  );
}

function ContentContract(): React.JSX.Element {
  // 23-block shared prefix; block 20 is the receipt divergence (57f2a77e...).
  const N = 23;
  const parentOld: BlockRole[] = Array.from({ length: N }, (_, i) => (i < 20 ? "user" : "assistant"));
  const subOld: BlockRole[] = Array.from({ length: N }, () => "user");
  const frozen: BlockRole[] = Array.from({ length: N }, () => "user");

  return (
    <Stack gap={16}>
      <Grid columns={2} gap={16} align="start">
        <div className={clsxBorder()}>
          <div className={borderHeader()}>advance_turn relabels block 20 — per-turn (old)</div>
          <div className="p-3">
            <Stack gap={10}>
              <Stack gap={4}>
                <span className={`text-sm ${inkClassName("tertiary")}`}>parent chain</span>
                <BlockStrip roles={parentOld} diverge={20} />
              </Stack>
              <Stack gap={4}>
                <span className={`text-sm ${inkClassName("tertiary")}`}>forking subagent</span>
                <BlockStrip roles={subOld} diverge={20} />
              </Stack>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Block 20 is <span style={{ color: "var(--color-category-purple)", fontWeight: 600 }}>assistant</span>{" "}
                on the parent but <span style={{ color: "var(--color-category-blue)", fontWeight: 600 }}>user</span>{" "}
                on the subagent — different tokenization, KV-cache miss on the shared prefix.
              </p>
            </Stack>
          </div>
        </div>
        <div className={clsxBorder()}>
          <div className={borderHeader()}>role fixed at creation — frozen per-block (new)</div>
          <div className="p-3">
            <Stack gap={10}>
              <Stack gap={4}>
                <span className={`text-sm ${inkClassName("tertiary")}`}>parent chain</span>
                <BlockStrip roles={frozen} diverge={20} />
              </Stack>
              <Stack gap={4}>
                <span className={`text-sm ${inkClassName("tertiary")}`}>forking subagent</span>
                <BlockStrip roles={frozen} diverge={20} />
              </Stack>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Block 20&apos;s <Code inline>(role, message_index)</Code> is set by its creating node and
                inherited verbatim — both paths emit an identical segment-id chain, so the server&apos;s KV
                prefix hits.
              </p>
            </Stack>
          </div>
        </div>
      </Grid>
      <Legend
        entries={[
          { color: "blue", label: "user" },
          { color: "purple", label: "assistant" },
          { color: "orange", label: "block 20 (divergence point)" },
        ]}
      />
    </Stack>
  );
}

// --- data flow (React Flow: timing lane over content lane) ------------------

const FLOW_NODES: Node[] = [
  { id: "recorded", type: "panel", position: { x: 0, y: 0 }, data: { title: "recorded requests", detail: "t, hash_ids, api_time" } },
  { id: "flatten", type: "panel", position: { x: 200, y: 0 }, data: { title: "flatten + warp", detail: "DFS order, async_ancestors, idle-gap" } },
  { id: "rank", type: "panel", position: { x: 400, y: 0 }, data: { title: "global rank", detail: "sort(start, end, node_id)", strokeRole: "primary" } },
  { id: "interval", type: "panel", position: { x: 600, y: 0 }, data: { title: "interval order", detail: "finished-before, frontier-reduced", strokeRole: "primary" } },
  { id: "parsed", type: "panel", position: { x: 800, y: 0 }, data: { title: "ParsedGraph", detail: "nodes + StaticEdges" } },

  { id: "trie", type: "panel", position: { x: 0, y: 130 }, data: { title: "scope-blind trie", detail: "full-prefix then partial-LCP" } },
  { id: "blocktag", type: "panel", position: { x: 200, y: 130 }, data: { title: "block-tag pass", detail: "role + starts_new_message, frozen", strokeRole: "primary" } },
  { id: "message-unit", type: "panel", position: { x: 400, y: 130 }, data: { title: "message-unit emission", detail: "one pool id per message" } },
  { id: "metrics", type: "panel", position: { x: 600, y: 130 }, data: { title: "metrics", detail: "prefix share, async-cross, boundary-divergence" } },
];

const FLOW_EDGES: Edge[] = [
  { id: "f-recorded-flatten", source: "recorded", target: "flatten", type: "flow" },
  { id: "f-flatten-rank", source: "flatten", target: "rank", type: "flow" },
  { id: "f-rank-interval", source: "rank", target: "interval", type: "flow" },
  { id: "f-interval-parsed", source: "interval", target: "parsed", type: "flow" },
  { id: "f-trie-blocktag", source: "trie", target: "blocktag", type: "flow" },
  { id: "f-blocktag-message", source: "blocktag", target: "message-unit", type: "flow" },
  { id: "f-message-metrics", source: "message-unit", target: "metrics", type: "flow" },
];

function DataFlow(): React.JSX.Element {
  return (
    <div style={{ height: 300 }}>
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={FLOW_NODES}
        edges={FLOW_EDGES}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
      </ReactFlow>
    </div>
  );
}

// --- shared small helpers for the open bordered sections --------------------

function clsxBorder(): string {
  return `rounded-lg border shadow-sm ${strokeClassName("secondary")}`;
}

function borderHeader(): string {
  return `border-b px-3 py-2 text-sm font-semibold ${strokeClassName("secondary")} ${inkClassName("primary")}`;
}

// --- source links (informational; the host openFile action is not available) ---

const SOURCE_FILES: Array<[string, string]> = [
  ["design spec", "/home/anthony/.aiperf/docs/superpowers/specs/2026-06-30-weka-interval-order-causality-design.md"],
  ["_weka_trie_build.py", "src/aiperf/dataset/loader/graph/adapters/_weka_trie_build.py"],
  ["_weka_content.py", "src/aiperf/dataset/loader/graph/adapters/_weka_content.py"],
];

/**
 * Single-view port of the `weka-timing-causality` Cursor canvas: an interval-order causality
 * explainer with a live interactive lab, a rank tie-break demo, a content-contract comparison, a
 * timing/content data-flow pipeline, and a locked-regression table. Self-contained; takes no props.
 */
export function WekaTimingCausalityDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Timing & Causality" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl 2xl:max-w-[1728px] bg-surface-page px-10 py-8">
          <Stack gap={26}>
            <Stack gap={10}>
              <Row align="center" gap={10} wrap>
                <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
                  Weka interval-order causality
                </h1>
                <span className="rounded-full border border-accent-primary bg-accent-primary px-3 py-1 text-xs font-medium text-white">
                  design
                </span>
                <span className={`rounded-full border px-3 py-1 text-xs font-medium ${strokeClassName("secondary")} ${inkClassName("secondary")}`}>
                  synthesize mode
                </span>
              </Row>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                The weka trace is <strong>lossy</strong> — it records timestamps, KV-block hashes, and
                subagent markers, but no causal graph. This reconstructs causality purely from timing: plot
                each request as an interval on the warped clock and draw <Code inline>A → B</Code> only when
                A had <strong>provably finished</strong> before B began. Overlapping intervals stay
                concurrent.
              </p>
              <Row gap={8} wrap align="center">
                <Eyebrow>Source</Eyebrow>
                {SOURCE_FILES.map(([label, path]) => (
                  <span key={path} className="inline-flex items-center gap-2">
                    <span className={`text-xs font-semibold ${inkClassName("secondary")}`}>{label}</span>
                    <Code inline>{path}</Code>
                  </span>
                ))}
              </Row>
            </Stack>

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Two failure modes it replaces
              </h2>
              <Grid columns={2} gap={16} align="start">
                <Callout tone="danger" title="Racing siblings get chained">
                  Two subagents that prefill the same shared prefix carry identical <Code inline>hash_ids</Code>.
                  The scope-blind <Code inline>content_parent</Code> rule (longest-full-prefix, most-recent)
                  threads the second racer through the first&apos;s <strong>output</strong> — a lineage that
                  never existed. Interval-order forks both off the common ancestor instead.
                </Callout>
                <Callout tone="danger" title="Parentless spawn chains collapse">
                  An agent spawned <strong>because another finished</strong>, with no parent turn between, has no
                  structural candidate cause — <Code inline>spawner</Code> is positional and joins only attach to
                  a later parent leaf. The join edge is silently dropped even though the timing to infer it is
                  present.
                </Callout>
              </Grid>
              <Callout tone="info" title="One root assumption">
                Both trace to <strong>causality flows only through parent leaves</strong>, and to overloading one
                scope-blind rule for two questions it cannot separate: &ldquo;do we share an input prefix?&rdquo;
                vs. &ldquo;did one consume the other&apos;s output?&rdquo;
              </Callout>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Interval-order lab</h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                The worked example: parent <Code inline>P0</Code>, two overlapping explorers{" "}
                <Code inline>A0</Code>/<Code inline>B0</Code>, a spawned <Code inline>C0</Code>, and a parent
                resume <Code inline>P1</Code>. Slide <Code inline>C0</Code>&apos;s start to watch overlap flip an
                edge on and off, or mark <Code inline>B0</Code> async to drop it as a predecessor. The DAG is
                re-derived live with frontier reduction.
              </p>
              <CausalityLab />
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Strict order &amp; the time-consistent rank
              </h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                <Code inline>A.end ≤ B.start</Code> alone is not antisymmetric, so it does not yield a DAG.
                Zero-duration intervals are legal (<Code inline>api_time</Code> is <Code inline>None</Code>,{" "}
                <Code inline>t</Code> only needs <Code inline>ge=0</Code>), so two coincident requests would
                produce a mutual edge. The fix is a <strong>time-consistent rank</strong>, not{" "}
                <Code inline>_Node.order</Code> (the DFS index), which would drop real cross-scope edges.
              </p>
              <Grid columns="minmax(0, 340px) minmax(0, 1fr)" gap={16} align="start">
                <RankDemo />
                <Stack gap={10}>
                  <Callout tone="success" title="No dropped real edges (monotone-transfer)">
                    Completed-before is decided on the <strong>raw</strong> clock; <Code inline>rank</Code> is
                    keyed on the <strong>warped</strong> clock. Because <Code inline>_ActiveIdleWarp</Code> is
                    monotone and never cuts inside an active interval, every raw finished-before pair is also
                    warped-finished-before, so <Code inline>key(A) &lt; key(B)</Code> and the edge is emitted.
                  </Callout>
                  <Callout tone="info" title="DAG guaranteed">
                    <Code inline>rank = sort(start, end, node_id)</Code> is a strict total order, so the
                    conjunction is irreflexive, antisymmetric, and transitive. Coincident zero-duration pairs
                    resolve to exactly one direction — no 2-cycle can form.
                  </Callout>
                </Stack>
              </Grid>
              <Callout tone="warning" title="Timing fidelity, not causal minimality">
                The interval order is a <strong>sufficient over-approximation</strong>: it never drops a real
                dependency, but two independent scopes that happen to be sequential within a trace acquire a
                finished-before edge across the scope boundary. That re-serializes independent-but-sequential
                work at replay — accepted and pinned by a regression, with a §4 statistic measuring how often it
                binds.
              </Callout>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Content contract — trace-global role-segmentation uniqueness
              </h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                For any block prefix in the entire trace, exactly one role-segmentation exists. A block&apos;s
                role and message boundaries are decided by the node that <strong>creates</strong> it, fixed at
                creation, and inherited immutably. Any two requests sharing a block prefix re-emit the identical
                segment-id chain — same blocks, roles, and boundaries — so the server&apos;s KV cache hits.
              </p>
              <ContentContract />
              <Callout tone="info" title="Message-unit emission (not block-unit)">
                Consecutive blocks with the same frozen <Code inline>(role, message_index)</Code> concatenate
                into one message; each message is one content-addressed pool entry via the existing{" "}
                <Code inline>segment_id(parent_id, role, tokens)</Code> — no store-schema change. A trailing-user
                cap is frozen at block creation, never coerced at assembly.
              </Callout>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Data flow</h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Timing (top) and content (bottom) are cleanly separated — the interval order is independent of
                the content contract, and both feed one <Code inline>ParsedGraph</Code>. The same scope-blind
                trie also answers read-only prefix-sharing and divergence metrics.
              </p>
              <DataFlow />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Locked regressions</h2>
              <Table
                columns={[
                  { key: "regression", label: "Regression" },
                  { key: "asserts", label: "Asserts" },
                ]}
                rows={[
                  {
                    regression: "Zero-duration coincidence",
                    asserts:
                      "same t + api_time=None → exactly one edge, no mutual edge, no cycle; executor runs without deadlock",
                  },
                  {
                    regression: "No DFS-order tie-break",
                    asserts:
                      "a finished-before parent turn with a higher _Node.order than a later subagent leaf still gets its edge",
                  },
                  {
                    regression: "Cross-scope serialization",
                    asserts:
                      "two sequential independent scopes in one trace DO get a finished-before edge (pins the accepted over-serialization)",
                  },
                  {
                    regression: "Shared-prefix identity",
                    asserts: "any two requests sharing a block prefix emit an identical per-block id chain",
                  },
                  {
                    regression: "No-relabel-across-chains",
                    asserts:
                      "the 57f2a77e receipt topology — a shared block's frozen tag is identical across parent chain and forking subagent",
                  },
                  {
                    regression: "Block-aligned ISL (hard abort)",
                    asserts: "prompt token count = min(len(hash_ids), in // bs) * bs, or the build aborts",
                  },
                  {
                    regression: "async_launched exclusion",
                    asserts:
                      "a fire-and-forget child before a later out-of-subtree node produces no edge; nested subset rule keeps intra-subtree edges",
                  },
                ]}
              />
            </Stack>
          </Stack>
        </div>
      </div>
    </div>
  );
}
