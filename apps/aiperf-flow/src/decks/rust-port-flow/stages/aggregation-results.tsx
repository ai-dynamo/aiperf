/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 8 — Aggregation → final results. Fills in the level-1 subgraph, the level-2 "exact-fold vs
//! t-digest-merge" leaf, verified source anchors, and a level-1 `FlowStep[]` fragment for the play
//! head to traverse this stage's internals.
//!
//! The narrative (spec §8, corrections baked in): a worker-local `NativeMetricsObserver` accumulates
//! records into a `metrics_core` NaN-sparse column store (NaN is a sparse-column sentinel scrubbed at
//! the reporting boundary). Two fold strategies then coexist — EXACT folds replay the column-store
//! partitions in global order so the FINAL report stays bit-exact from records, while the SKETCH path
//! (`cellular::sketch::TDigest`, NOT DDSketch) drives mergeable heartbeats / cellular merge where
//! percentiles + stddev become streaming estimates but counts/sums/extrema stay exact. A
//! deterministic boundary merge feeds `NativeReporter`, producing a `NativeReport` that the
//! `ExporterRegistry` (nine sinks, file writers before uploaders) emits, and the runner writes
//! `native-v2.json` and emits `RunTerminalV2` carrying its `report_path`.

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";
import { Diagram, NodeChip, DbNode, MiniArrow, MiniBars } from "../../../chalk/index.js";

/** Leaf id: the clickable node that drills into the exact-fold vs t-digest-merge comparison. */
const EXACT_VS_SKETCH_LEAF = "aggExactVsSketch";

const COL = 240;

/** Level-1 subgraph: the aggregation pipeline internals, worker-local accumulation → terminal report. */
function aggregationNodes(): Node[] {
  return [
    {
      id: "agg-observer",
      type: "card",
      position: { x: 0, y: 0 },
      data: {
        title: "NativeMetricsObserver",
        subtitle: "worker-local",
        detail: "Per-worker record accumulation on the run clock origin.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip accent>observe</NodeChip>
            <MiniArrow />
            <DbNode>store</DbNode>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-store",
      type: "card",
      position: { x: COL, y: 0 },
      data: {
        title: "NaN-sparse column store",
        subtitle: "metrics_core",
        detail: "MetricsAccumulator ragged columns; NaN = sparse sentinel.",
        className: roleClassName("storage"),
        diagram: (
          <Diagram>
            <MiniBars heights={[30, 100, 55, 80, 20]} />
            <MiniArrow />
            <DbNode accent>cols</DbNode>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-exact",
      type: "card",
      position: { x: 2 * COL, y: -110 },
      data: {
        title: "Exact record fold",
        subtitle: "authoritative",
        detail: "Partitions replayed in global order — exact from records.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip accent>exact</NodeChip>
            <MiniArrow />
            <NodeChip>global order</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-sketch",
      type: "card",
      position: { x: 2 * COL, y: 110 },
      data: {
        title: "t-digest sketch",
        subtitle: "cellular::sketch::TDigest",
        detail: "Mergeable p*/stddev estimate (NOT DDSketch); extrema exact.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip accent>p50·p90·p99</NodeChip>
            <MiniArrow />
            <MiniBars heights={[50, 90, 99]} />
          </Diagram>
        ),
      },
    },
    {
      id: EXACT_VS_SKETCH_LEAF,
      type: "card",
      position: { x: 3 * COL, y: 0 },
      data: {
        title: "Exact vs sketch merge",
        subtitle: "click to compare",
        detail: "Exact folds for reports; sketches for mergeable heartbeats.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip accent>exact</NodeChip>
            <NodeChip>sketch</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-boundary",
      type: "card",
      position: { x: 4 * COL, y: 0 },
      data: {
        title: "Deterministic boundary merge",
        subtitle: "order-independent",
        detail: "Column-store replay + associative TDigest::merge.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <DbNode>shard</DbNode>
            <MiniArrow />
            <NodeChip accent>merge</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-reporter",
      type: "card",
      position: { x: 5 * COL, y: 0 },
      data: {
        title: "NativeReporter → NativeReport",
        subtitle: "metrics_core::report",
        detail: "Builds NativeReport (version, summary, metric map), no IO.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip>summary</NodeChip>
            <MiniArrow />
            <NodeChip accent>NativeReport</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-registry",
      type: "card",
      position: { x: 6 * COL, y: 0 },
      data: {
        title: "ExporterRegistry",
        subtitle: "nine sinks",
        detail: "Ordered Exporters: file writers before network uploaders.",
        className: roleClassName("media"),
        diagram: (
          <Diagram>
            <NodeChip accent>JSON·CSV</NodeChip>
            <MiniArrow />
            <NodeChip>9 sinks</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "agg-terminal",
      type: "card",
      position: { x: 7 * COL, y: 0 },
      data: {
        title: "RunTerminalV2",
        subtitle: "report_path",
        detail: "Writes native-v2.json; emits terminal with report_path.",
        className: roleClassName("control"),
        diagram: (
          <Diagram>
            <DbNode accent>native-v2.json</DbNode>
            <MiniArrow />
            <NodeChip>path</NodeChip>
          </Diagram>
        ),
      },
    },
  ];
}

/** Level-1 edges: linear spine with an exact/sketch split converging on the comparison node. */
function aggregationEdges(): Edge[] {
  return [
    { id: "e-obs-store", source: "agg-observer", target: "agg-store", type: "flow" },
    { id: "e-store-exact", source: "agg-store", target: "agg-exact", type: "flow" },
    { id: "e-store-sketch", source: "agg-store", target: "agg-sketch", type: "flow" },
    { id: "e-exact-cmp", source: "agg-exact", target: EXACT_VS_SKETCH_LEAF, type: "flow" },
    { id: "e-sketch-cmp", source: "agg-sketch", target: EXACT_VS_SKETCH_LEAF, type: "flow" },
    { id: "e-cmp-boundary", source: EXACT_VS_SKETCH_LEAF, target: "agg-boundary", type: "flow" },
    { id: "e-boundary-reporter", source: "agg-boundary", target: "agg-reporter", type: "flow" },
    { id: "e-reporter-registry", source: "agg-reporter", target: "agg-registry", type: "flow" },
    { id: "e-registry-terminal", source: "agg-registry", target: "agg-terminal", type: "flow" },
  ];
}

/** Level-2 leaf: the exact-fold (authoritative) vs t-digest-merge (mergeable estimate) comparison. */
function exactVsSketchLeafNodes(): Node[] {
  return [
    {
      id: "leaf-exact-fold",
      type: "card",
      position: { x: 0, y: 0 },
      data: {
        title: "Exact: merge_records_in_global_order",
        subtitle: "column store",
        detail: "Shard partitions replayed in order — bit-exact from records.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <DbNode>shard</DbNode>
            <MiniArrow />
            <NodeChip accent>bit-exact</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "leaf-exact-report",
      type: "card",
      position: { x: 0, y: 170 },
      data: {
        title: "Final NativeReport = EXACT",
        subtitle: "from records",
        detail: "Metrics from retained records; no terminal estimation.",
        className: roleClassName("storage"),
        diagram: (
          <Diagram>
            <DbNode>records</DbNode>
            <MiniArrow />
            <NodeChip accent>report</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "leaf-sketch-merge",
      type: "card",
      position: { x: 340, y: 0 },
      data: {
        title: "t-digest: TDigest::merge",
        subtitle: "cellular::sketch",
        detail: "Concat centroids + compress; min/max exact; associative.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <MiniBars heights={[40, 85, 60, 95]} />
            <MiniArrow />
            <NodeChip accent>compress</NodeChip>
          </Diagram>
        ),
      },
    },
    {
      id: "leaf-sketch-heartbeat",
      type: "card",
      position: { x: 340, y: 170 },
      data: {
        title: "Mergeable heartbeats / cellular merge",
        subtitle: "streaming estimate",
        detail: "Percentiles + stddev become streaming estimates; counts/sums/extrema exact.",
        className: roleClassName("compute"),
        diagram: (
          <Diagram>
            <NodeChip>♥</NodeChip>
            <MiniArrow />
            <NodeChip accent>p* estimate</NodeChip>
          </Diagram>
        ),
      },
    },
  ];
}

/** Level-2 leaf edges: each column chains its type into its consequence. */
function exactVsSketchLeafEdges(): Edge[] {
  return [
    { id: "e-leaf-exact", source: "leaf-exact-fold", target: "leaf-exact-report", type: "flow" },
    { id: "e-leaf-sketch", source: "leaf-sketch-merge", target: "leaf-sketch-heartbeat", type: "flow" },
  ];
}

/**
 * Level-1 play fragment: the request's record traversing this stage's internals, active node id +
 * real caption per hop. Exposed so a stage-internal play head (or a future full-pipeline flatten)
 * can animate the aggregation stage using the real type names.
 */
export const aggregationResultsSteps: readonly FlowStep[] = [
  {
    nodeId: "agg-observer",
    caption: "The record lands in the worker-local NativeMetricsObserver, sharing the runtime clock origin.",
  },
  {
    nodeId: "agg-store",
    caption: "It is appended into the metrics_core NaN-sparse column store (MetricsAccumulator ragged columns).",
  },
  {
    nodeId: "agg-exact",
    caption: "Exact fold: the column store replays partitions in global order so the final report stays exact from records.",
  },
  {
    nodeId: "agg-sketch",
    caption: "Sketch path: a cellular::sketch::TDigest (not DDSketch) captures a mergeable percentile/stddev estimate.",
  },
  {
    nodeId: EXACT_VS_SKETCH_LEAF,
    caption: "Exact folds drive the final report; t-digest sketches drive mergeable heartbeats — counts/sums/extrema stay exact either way.",
  },
  {
    nodeId: "agg-boundary",
    caption: "Deterministic boundary merge: exact column-store replay + associative TDigest::merge, order-independent.",
  },
  {
    nodeId: "agg-reporter",
    caption: "NativeReporter builds the NativeReport (schema_version, summary, metric map) without IO.",
  },
  {
    nodeId: "agg-registry",
    caption: "ExporterRegistry fans the report out to nine sinks — local-file writers before network uploaders.",
  },
  {
    nodeId: "agg-terminal",
    caption: "The runner writes native-v2.json and emits RunTerminalV2 carrying its report_path.",
  },
];

/**
 * Stage 8 detail: aggregation → final results. Keeps the spine id/order/label/tone so it drops into
 * the `STAGES` registry in place of the stub; adds the real subgraph, the exact-vs-sketch leaf, and
 * verified `file:line` anchors.
 */
export const aggregationResultsStage: StageDef = {
  id: "aggregation",
  order: 8,
  label: "Aggregation → results",
  caption:
    "Worker-local NativeMetricsObserver → metrics_core NaN-sparse column store → EXACT folds (final report is exact from records) vs t-digest sketch (cellular::sketch::TDigest, mergeable heartbeats; percentiles+stddev streaming, counts/sums/extrema exact) → deterministic boundary merge → NativeReporter → NativeReport → ExporterRegistry (nine sinks) → RunTerminalV2 report_path.",
  tone: "green",
  // v2 timeline: the region sits in the Aggregate lane (measure, merge); the report/terminal events
  // hop into the Export lane — the request's journey ends at RunTerminalV2's report_path.
  lane: "aggregate",
  events: [
    { id: "ag-measure", label: "measure", laneId: "aggregate", atOrder: 12, realOffsetMs: 208 },
    { id: "ag-merge", label: "merge", laneId: "aggregate", atOrder: 13, realOffsetMs: 212 },
    { id: "ag-report", label: "report", laneId: "export", atOrder: 14, realOffsetMs: 216 },
    { id: "ag-terminal", label: "terminal", laneId: "export", atOrder: 15, realOffsetMs: 220 },
  ],
  subgraph: {
    nodes: aggregationNodes(),
    edges: aggregationEdges(),
    children: [EXACT_VS_SKETCH_LEAF],
  },
  leaves: {
    [EXACT_VS_SKETCH_LEAF]: {
      label: "Exact folds vs t-digest merge",
      nodes: exactVsSketchLeafNodes(),
      edges: exactVsSketchLeafEdges(),
    },
  },
  evidence: [
    { label: "struct NativeMetricsObserver", path: "runtime/src/metrics.rs:203" },
    { label: "struct MetricsAccumulator", path: "runtime/src/metrics_core/accumulator.rs:416" },
    { label: "MetricValue NaN sparse-column sentinel", path: "runtime/src/metrics_core/value.rs:6" },
    { label: "TDigest (cellular::sketch)", path: "runtime/src/cellular/mod.rs:33" },
    { label: "fn TDigest::merge", path: "runtime/src/cellular/sketch.rs:127" },
    { label: "fn merge_records_in_global_order", path: "runtime/src/cellular/shard.rs:106" },
    { label: "struct NativeReporter", path: "runtime/src/metrics_core/report.rs:1031" },
    { label: "struct NativeReport", path: "runtime/src/metrics_core/report.rs:1079" },
    { label: "trait Exporter", path: "runtime/src/export/mod.rs:208" },
    { label: "struct ExporterRegistry", path: "runtime/src/export/mod.rs:258" },
    { label: "fn register_builtins (nine sinks)", path: "runtime/src/export/mod.rs:295" },
    { label: "RunTerminalV2 report_path", path: "runtime/src/engine/coordinator.rs:334" },
  ],
};
