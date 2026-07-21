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
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

// Ported from weka-trie-build.canvas.tsx (a real, hand-authored Cursor Canvas). Single-view
// canvas — no internal page tabs. Source: aiperf's graph/recorded/weka lowering
// (`_weka_trie_build.py`). Walks the four passes of `build_trie_graph`: flatten recorded
// leaves, resolve content lineage from a hash-id prefix trie, warp idle time, then emit
// LlmNodes with completed-before timing edges.

// --- pass 1/3/4 summary strip ---------------------------------------------

const PASSES: Array<{ n: string; title: string; sub: string }> = [
  {
    n: "1",
    title: "_flatten_requests",
    sub: "DFS every n/s leaf in recorded t order; recurse subagents; stamp spawner + timing-aware joined_causes + chain_prev.",
  },
  {
    n: "2",
    title: "_resolve_content_parents",
    sub: "incremental hash-id prefix trie: longest full prefix, else longest partial-LCP branch point.",
  },
  {
    n: "3",
    title: "_apply_idle_gap_warp",
    sub: "collapse true idle gaps to the cap on a shared warped clock; api_time never cut.",
  },
  {
    n: "4",
    title: "build node + edges",
    sub: "segments via memoized reconstructor; completed-before waits-for edges; AND-fan-in inputs.",
  },
];

function PassStep({ n, title, sub }: { n: string; title: string; sub: string }): React.JSX.Element {
  return (
    <Stack
      gap={3}
      className={`min-w-[150px] flex-1 rounded-lg border border-t-4 border-t-accent-primary px-3 py-2.5 shadow-sm ${strokeClassName("secondary")}`}
    >
      <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>pass {n}</span>
      <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</span>
      <span className={`text-sm ${inkClassName("secondary")}`}>{sub}</span>
    </Stack>
  );
}

function Passes(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        build_trie_graph — four passes
      </h2>
      <Row gap={12} wrap align="stretch">
        {PASSES.map((p) => (
          <PassStep key={p.n} n={p.n} title={p.title} sub={p.sub} />
        ))}
      </Row>
      <Callout tone="info" title="Emitted IR is intentionally tiny">
        One <span className={`font-semibold ${inkClassName("primary")}`}>LlmNode</span> per recorded
        n/s request + plain <span className={`font-semibold ${inkClassName("primary")}`}>StaticEdge</span>{" "}
        waits-for edges. No subgraph / spawn / await / reducer / channel topology — structure comes
        purely from recorded hash-id prefixes and timing facts.
      </Callout>
    </Stack>
  );
}

// --- pass 2: prefix trie graph ---------------------------------------------

const trieNodes: Node[] = [
  {
    id: "root",
    type: "card",
    position: { x: 220, y: 0 },
    data: { title: "root", detail: "empty prefix" },
  },
  {
    id: "A",
    type: "card",
    position: { x: 220, y: 110 },
    data: {
      title: "hash A",
      detail: "passer=r1 · terminal=r1",
      className: "border-l-4 border-l-category-blue",
    },
  },
  {
    id: "B",
    type: "card",
    position: { x: 80, y: 220 },
    data: { title: "hash B", detail: "terminal=r2", className: "border-l-4 border-l-category-blue" },
  },
  {
    id: "C",
    type: "card",
    position: { x: 360, y: 220 },
    data: { title: "hash C", detail: "terminal=r3", className: "border-l-4 border-l-category-blue" },
  },
  {
    id: "D",
    type: "card",
    position: { x: 80, y: 330 },
    data: { title: "hash D", detail: "terminal=r4", className: "border-l-4 border-l-category-blue" },
  },
];

const trieEdges: Edge[] = [
  { id: "e-root-a", source: "root", target: "A", type: "flow" },
  { id: "e-a-b", source: "A", target: "B", type: "flow" },
  { id: "e-a-c", source: "A", target: "C", type: "flow" },
  { id: "e-b-d", source: "B", target: "D", type: "flow" },
];

const RESOLUTION_PICKS: Array<{ req: string; hashes: string; parent: string; why: string }> = [
  { req: "r2", hashes: "[A, B]", parent: "r1", why: "longest full prefix ([A] terminates at r1)" },
  {
    req: "r3",
    hashes: "[A, C]",
    parent: "r1",
    why: "no full prefix -> branch point via passer at depth 1",
  },
  { req: "r4", hashes: "[A, B, D]", parent: "r2", why: "full prefix [A,B] beats [A]" },
];

function ResolutionChip({ accent, children }: { accent?: boolean; children: React.ReactNode }): React.JSX.Element {
  return (
    <span
      className={`rounded-md border px-2 py-1 text-xs font-semibold shadow-sm ${
        accent ? "border-accent-primary text-accent-primary" : `${strokeClassName("secondary")} ${inkClassName("secondary")}`
      }`}
    >
      {children}
    </span>
  );
}

function ContentParent(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Pass 2 — content-parent = hash-id prefix tree
      </h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        Each request&apos;s content-parent is the earlier request whose hash_ids is the longest{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>full prefix</span> (tie-broken
        toward the most recent), else the longest{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>partial-LCP branch point</span>.
        Built incrementally — no O(n²) pairwise scan.
      </p>
      <Grid columns="minmax(0, 280px) 1fr" gap={20} align="start">
        <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
          <div className={`flex items-center justify-between border-b px-3 py-2 ${strokeClassName("secondary")}`}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Prefix trie</span>
            <span className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}>
              accent = terminal
            </span>
          </div>
          <div style={{ height: 420 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={trieNodes}
              edges={trieEdges}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
            </ReactFlow>
          </div>
        </div>
        <Stack gap={8}>
          <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>Resolution picks</h3>
          {RESOLUTION_PICKS.map((p) => (
            <div key={p.req} className={`rounded-lg border px-3 py-2.5 shadow-sm ${strokeClassName("secondary")}`}>
              <Row gap={8} align="center" wrap>
                <ResolutionChip accent>
                  {p.req} {p.hashes}
                </ResolutionChip>
                <span className={inkClassName("tertiary")}>-&gt;</span>
                <ResolutionChip>content_parent = {p.parent}</ResolutionChip>
              </Row>
              <p className={`mt-1.5 text-sm ${inkClassName("secondary")}`}>{p.why}</p>
            </div>
          ))}
          <Callout tone="warning" title="content_parent is content-only">
            It selects which segment-pool prefix a turn materializes — it is{" "}
            <span className={`font-semibold ${inkClassName("primary")}`}>not</span> a timing cause. A
            branch point can be arbitrarily far back; anchoring the firing delay there would sum the
            warped distance (the aggregate-timestamp bug). Timing anchors on{" "}
            <span className={`font-semibold ${inkClassName("primary")}`}>chain_prev</span> instead.
          </Callout>
        </Stack>
      </Grid>
    </Stack>
  );
}

// --- pass 4: timing edges ---------------------------------------------------

function CauseCard({
  badge,
  title,
  children,
}: {
  badge: string;
  title: string;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className={`rounded-lg border shadow-sm ${strokeClassName("secondary")}`}>
      <div className={`flex items-center justify-between border-b px-3 py-2 ${strokeClassName("secondary")}`}>
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</span>
        <span className={`rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${strokeClassName("secondary")} ${inkClassName("secondary")}`}>
          {badge}
        </span>
      </div>
      <div className="px-3 py-3">
        <Stack gap={8}>{children}</Stack>
      </div>
    </div>
  );
}

function TimingEdges(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
        Pass 4 — timing edges = completed-before waits-for
      </h2>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        Candidate causes are <span className={`font-semibold ${inkClassName("primary")}`}>spawner</span>,
        joined blocking-subagent leaves,{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>content_parent</span>, and{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>chain_prev</span>. A cause is a real
        dependency only if it{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>completed before</span> this turn
        started on the raw clock (
        <span className={`font-semibold ${inkClassName("primary")}`}>cause.t + api_time &lt;= R.t</span>).
      </p>
      <Grid columns={3} gap={16}>
        <CauseCard badge="binding" title="Latest completed cause">
          <Row gap={8} align="center" wrap>
            <ResolutionChip accent>cause*</ResolutionChip>
            <span className={inkClassName("tertiary")}>--&gt;</span>
            <ResolutionChip>R</ResolutionChip>
          </Row>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            delay_after_predecessor_us = max(0, R.start − cause.end) on the warped clock. Whoever
            returns last governs at runtime.
          </p>
        </CauseCard>
        <CauseCard badge="AND-fan-in" title="Other completed causes">
          <Row gap={8} align="center" wrap>
            <ResolutionChip>c1</ResolutionChip>
            <ResolutionChip>c2</ResolutionChip>
            <span className={inkClassName("tertiary")}>==&gt;</span>
            <ResolutionChip>R</ResolutionChip>
          </Row>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Each carries delay 0 (already finished) and one count=1 input on {"{src}_out"} — the recorded
            AND-join wait.
          </p>
        </CauseCard>
        <CauseCard badge="concurrent root" title="Nothing completed">
          <Row gap={8} align="center" wrap>
            <ResolutionChip>START</ResolutionChip>
            <span className={inkClassName("tertiary")}>--&gt;</span>
            <ResolutionChip>R</ResolutionChip>
          </Row>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Roots at START with min_start_delay_us = R.start, firing concurrently at the instant it
            actually started instead of serializing behind an overlapped request.
          </p>
        </CauseCard>
      </Grid>
      <Callout tone="info" title="Two planes, cleanly separated">
        <span className={`font-semibold ${inkClassName("primary")}`}>Content ancestry</span> (prefix trie
        -&gt; prompt segments) is independent of the{" "}
        <span className={`font-semibold ${inkClassName("primary")}`}>timing dependency</span> graph
        (completed-before edges). This split is exactly the Step/Emit Plane-2 vs Plane-1 distinction, and
        the interval-order rule (
        <span className={`font-semibold ${inkClassName("primary")}`}>
          A-&gt;B iff end(A) &lt;= start(B) ∧ rank(A) &lt; rank(B)
        </span>
        ) is its transitive reduction — a DAG by construction.
      </Callout>
    </Stack>
  );
}

// --- header ------------------------------------------------------------------

function DeckHeader(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Row align="center" gap={10} wrap>
        <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>Inside build_trie_graph</h1>
        <span
          className={`rounded-md border px-2 py-0.5 text-xs font-semibold border-accent-primary text-accent-primary shadow-sm`}
        >
          _weka_trie_build.py
        </span>
      </Row>
      <p className={`text-sm ${inkClassName("secondary")}`}>
        How one WekaTrace becomes a dependency-only ParsedGraph: flatten recorded leaves, resolve content
        lineage from a hash-id prefix trie, warp idle time, then emit LlmNodes with completed-before
        timing edges.
      </p>
    </Stack>
  );
}

/**
 * Ports `weka-trie-build.canvas.tsx` (a real, hand-authored Cursor Canvas) onto aiperf-flow's
 * component vocabulary. Single-view canvas — walks the four passes of `build_trie_graph`
 * (`_weka_trie_build.py`): flatten recorded leaves depth-first, resolve content_parent from an
 * incremental hash-id prefix trie, warp idle gaps on a shared clock, then emit `LlmNode` +
 * `StaticEdge` completed-before timing edges.
 */
export function WekaTrieBuildDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Trie Build" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={28}>
            <DeckHeader />
            <div className={`border-t ${strokeClassName("secondary")}`} />
            <Passes />
            <ContentParent />
            <TimingEdges />
          </Stack>
        </div>
      </div>
    </div>
  );
}
