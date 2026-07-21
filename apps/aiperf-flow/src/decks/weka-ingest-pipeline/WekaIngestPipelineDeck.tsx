/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Stat } from "../../prose/Stat.js";
import { Legend } from "../../prose/Legend.js";
import type { CategoryRole } from "../../theme/tokens.js";

// Ported from
// /home/anthony/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-weka-ir-v1/canvases/weka-ingest-pipeline.canvas.tsx.
// One Weka trace becomes a ParsedGraph + SegmentPool; a build plane persists
// content + addressing to mmap stores, and a schedule plane reparses for
// topology, agreeing on node ordinals so a worker can rematerialize any node.

type Tone = "input" | "parse" | "build" | "store" | "schedule" | "worker";

// Static lookup table (never a template-string interpolation) so Tailwind's JIT scanner sees
// every literal class name — see the aiperf-flow-diagrams skill's "Tailwind JIT trap" section.
const TONE_BORDER_CLASSES: Record<Tone, string> = {
  input: "border-l-4 border-l-category-gray",
  parse: "border-l-4 border-l-category-blue",
  build: "border-l-4 border-l-category-purple",
  store: "border-l-4 border-l-category-purple",
  schedule: "border-l-4 border-l-category-blue",
  worker: "border-l-4 border-l-category-green",
};

const TONE_STROKE_ROLE: Record<Tone, "primary" | "secondary" | "tertiary"> = {
  input: "tertiary",
  parse: "secondary",
  build: "primary",
  store: "primary",
  schedule: "secondary",
  worker: "primary",
};

type PipelineNodeSpec = {
  id: string;
  label: string;
  sub: string;
  tone: Tone;
  x: number;
  y: number;
};

const NODES: PipelineNodeSpec[] = [
  { id: "file", label: "Local .json", sub: "single Weka trace", tone: "input", x: 0, y: 0 },
  { id: "dir", label: "Local dir", sub: "parallel dir parser", tone: "input", x: 240, y: 0 },
  { id: "hf", label: "HF org/name (weka)", sub: "streaming rows", tone: "input", x: 480, y: 0 },

  { id: "detect", label: "workload_detect", sub: "~4KiB signature sniff", tone: "parse", x: 240, y: 130 },
  { id: "parse", label: "parse_graph_workload", sub: "shared ingest seam", tone: "parse", x: 240, y: 260 },

  { id: "from", label: "from_weka_trace", sub: "seed · tokenizer · corpus · idle cap", tone: "build", x: 240, y: 390 },
  { id: "build", label: "build_trie_graph", sub: "-> ParsedGraph + SegmentPool", tone: "build", x: 240, y: 520 },
  { id: "dm", label: "DatasetManager build plane", sub: "_configure_graph_workload", tone: "build", x: 240, y: 650 },

  { id: "seg", label: "segment store", sub: "content plane (dedup)", tone: "store", x: 60, y: 780 },
  { id: "delta", label: "graph delta store", sub: "addressing plane (envelopes)", tone: "store", x: 420, y: 780 },

  { id: "sched", label: "TimingManager reparse", sub: "scheduling_only=True", tone: "schedule", x: 620, y: 260 },
  { id: "strategy", label: "GraphIRReplayStrategy", sub: "trie_node_ordinals", tone: "schedule", x: 620, y: 390 },

  { id: "worker", label: "worker materialize", sub: "(trace_id, ordinal, variant)", tone: "worker", x: 240, y: 910 },
];

type PipelineEdgeSpec = { id: string; source: string; target: string; dashed?: boolean };

const EDGE_SPECS: PipelineEdgeSpec[] = [
  { id: "e-file-detect", source: "file", target: "detect" },
  { id: "e-dir-detect", source: "dir", target: "detect" },
  { id: "e-hf-detect", source: "hf", target: "detect" },
  { id: "e-detect-parse", source: "detect", target: "parse" },
  { id: "e-parse-from", source: "parse", target: "from" },
  { id: "e-from-build", source: "from", target: "build" },
  { id: "e-build-dm", source: "build", target: "dm" },
  { id: "e-dm-seg", source: "dm", target: "seg" },
  { id: "e-dm-delta", source: "dm", target: "delta" },
  { id: "e-seg-worker", source: "seg", target: "worker" },
  { id: "e-delta-worker", source: "delta", target: "worker" },
  { id: "e-parse-sched", source: "parse", target: "sched" },
  { id: "e-sched-strategy", source: "sched", target: "strategy" },
  { id: "e-strategy-worker", source: "strategy", target: "worker" },
  { id: "e-build-strategy", source: "build", target: "strategy", dashed: true },
];

const nodes: Node[] = NODES.map((n) => ({
  id: n.id,
  type: "panel",
  position: { x: n.x, y: n.y },
  data: { title: n.label, detail: n.sub, strokeRole: TONE_STROKE_ROLE[n.tone], className: TONE_BORDER_CLASSES[n.tone] },
}));

const edges: Edge[] = EDGE_SPECS.map((e) => ({
  id: e.id,
  source: e.source,
  target: e.target,
  type: "flow",
  ...(e.dashed
    ? { style: { strokeDasharray: "4 4" }, data: { speed: "slow" as const } }
    : {}),
}));

const LEGEND_ENTRIES: { color: CategoryRole; label: string }[] = [
  { color: "purple", label: "build plane (content + addressing)" },
  { color: "blue", label: "parse / schedule plane" },
  { color: "gray", label: "input forms" },
];

/**
 * Weka ingest, build, and runtime pipeline explainer deck.
 *
 * Ports the single-view `WekaIngestPipeline` component from
 * `weka-ingest-pipeline.canvas.tsx` onto aiperf-flow's node/edge vocabulary:
 * the hand-drawn SVG DAG becomes real React Flow `panel` nodes with `flow`
 * edges, and the tone-colored border accents become a static per-tone
 * Tailwind class lookup table. Shows how one Weka trace input (local JSON,
 * local dir, or an HF org/name) becomes a `ParsedGraph + SegmentPool`
 * consumed by two coupled planes — build (persists content + addressing to
 * mmap stores) and schedule (reparses for topology) — that agree on dense
 * node ordinals so a worker can rematerialize any node statelessly.
 */
export function WekaIngestPipelineDeck(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-6 overflow-auto p-6">
      <Stack gap={10}>
        <Row gap={10} align="center" wrap>
          <h1 className="text-xl font-bold">Weka ingest, build, and runtime pipeline</h1>
          <span className="rounded-none border border-[var(--color-stroke-secondary)] px-2 py-0.5 text-xs font-semibold uppercase tracking-wide">
            segment-trie IR
          </span>
        </Row>
        <p className="max-w-3xl text-sm text-[var(--color-ink-secondary)]">
          One Weka trace becomes a <strong>ParsedGraph + SegmentPool</strong>. Two planes read the
          parse: the <strong>build plane</strong> persists content + addressing to mmap stores, and
          the <strong>schedule plane</strong> reparses for topology. They agree on node ordinals so
          a worker can rematerialize any node.
        </p>
        <Grid columns={4} gap={12}>
          <Stat value="3" label="Input forms" />
          <Stat value="2" label="Coupled contracts" tone="positive" />
          <Stat value="LlmNode-only" label="Emitted IR" />
          <Stat value="stateless" label="Worker materialize" tone="positive" />
        </Grid>
      </Stack>

      <Row gap={16} wrap align="center">
        <Legend entries={LEGEND_ENTRIES} />
        <Row gap={6} align="center">
          <div className="h-0 w-4 border-t border-dashed border-[var(--color-stroke-tertiary)]" />
          <span className="text-sm text-[var(--color-ink-secondary)]">ordinal agreement</span>
        </Row>
      </Row>

      <div className="rounded-none border border-[var(--color-stroke-secondary)]">
        <div className="border-b border-[var(--color-stroke-secondary)] px-4 py-2 text-sm font-semibold">
          Flow: input to worker
        </div>
        <div style={{ height: 780 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
      </div>

      <Grid columns={2} gap={16}>
        <Callout tone="info" title="Graph / topology contract">
          Build and schedule parse the same workload through <strong>parse_graph_workload</strong>{" "}
          with identical run-derived knobs, so they derive the same ParsedGraph and the same dense
          node ordinals (sorted by arrival_offset_us, tie-broken by node id).
        </Callout>
        <Callout tone="info" title="Payload contract">
          Build persists each node&apos;s request-body data into mmap stores; workers materialize
          later by <strong>(trace_id, node_ordinal, phase_variant)</strong>. Prompt bytes come from
          segment ids/handles, never from predecessor channel values.
        </Callout>
      </Grid>

      <Callout tone="warning" title="Trie caveat">
        On the trie path DatasetManager skips the <strong>graph_meta.msgpack</strong> structural
        sidecar (segment_pool is not None), so TimingManager reparses in scheduling-only mode
        instead of reading the sidecar. Store payloads still come from the build.
      </Callout>
    </div>
  );
}
