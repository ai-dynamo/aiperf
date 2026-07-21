/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 11 — a request becomes a row before it becomes a summary. Hot observer lane,
//! RunCapture retention ribbon (retain | exact-fold | sketch), per-record lane, hierarchical
//! merge, and the coordinator join/commit band. Ported from `MetricsDeepDive`.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  headerNode,
  cardNode,
  panelNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type Retention = "retain" | "exactFold" | "sketch";

const SELECTED: Record<Retention, { title: string; sub: string; micro: string }> = {
  retain: {
    title: "Retain",
    sub: "CapturedRecord rows survive to batch finalize",
    micro: "rows sorted before batch finalize",
  },
  exactFold: {
    title: "Exact-fold",
    sub: "dispatch ordinal + exact accumulator, clean row dropped",
    micro: "request_index overwritten with fold ordinal",
  },
  sketch: {
    title: "Sketch",
    sub: "TagSketch t-digest + exact count/sum/extrema, row dropped",
    micro: "request_index unused",
  },
};

function buildNodes(retention: Retention): Node[] {
  const sel = SELECTED[retention];
  return [
    headerNode("band-hot", 0, 0, "Hot observer lane"),
    cardNode("reduce", 20, 50, "reduce_parsed_response", "wire facts"),
    cardNode("observer", 250, 50, "NativeMetricsObserver", "uuid → PendingRequest"),
    cardNode("into-record", 480, 50, "PendingRequest::into_record", "terminal reduction"),
    cardNode("ingest", 710, 50, "RecordIngest", "metrics-plane DTO", undefined, "primary"),

    headerNode("band-ribbon", 0, 160, "RunCapture retention ribbon"),
    cardNode("begin", 20, 210, "RunCapture::begin", "context + optional ordinal"),
    cardNode("fold", 250, 210, "fold_record | finish", "record lane before drop"),
    cardNode("variant", 480, 210, sel.title, sel.sub, sel.micro, "primary"),

    panelNode("perrecord", 20, 320, "per-record lane writes at completion", "JSONL · raw · CSV · outputs · Parquet"),

    headerNode("band-merge", 0, 400, "Hierarchical merge"),
    cardNode("worker-acc", 20, 450, "worker accumulator", "MetricsAccumulator::merge"),
    cardNode("shard-absorb", 250, 450, "shard absorb", "ColumnStore::append_store"),
    cardNode("cell-merge", 480, 450, "cell partition merge", "sorted by cell_id"),
    cardNode("profiling-summary", 710, 450, "profiling summary", "catalog + derived", undefined, "primary"),

    headerNode("band-commit", 0, 560, "Coordinator join, hard commit, soft export"),
    cardNode("side-channels", 20, 610, "side channels", "network · GPU · server"),
    cardNode("summarize", 250, 610, "summarize_run_metrics", "attach run-level facts"),
    cardNode("native-report", 480, 610, "NativeReport", "finalize provenance"),
    cardNode("atomic-json", 710, 610, "atomic native-v2.json", "tmp · sync_all · rename", "existing target returns reporting error", "primary"),
  ];
}

const edges: Edge[] = [
  flowEdge("e-reduce-obs", "reduce", "observer"),
  flowEdge("e-obs-into", "observer", "into-record"),
  flowEdge("e-into-ingest", "into-record", "ingest"),
  flowEdge("e-ingest-begin", "ingest", "begin", { speed: "slow" }),
  flowEdge("e-begin-fold", "begin", "fold"),
  flowEdge("e-fold-variant", "fold", "variant"),
  flowEdge("e-fold-perrecord", "fold", "perrecord"),
  flowEdge("e-variant-worker", "variant", "worker-acc", { speed: "slow" }),
  flowEdge("e-worker-shard", "worker-acc", "shard-absorb"),
  flowEdge("e-shard-cell", "shard-absorb", "cell-merge"),
  flowEdge("e-cell-summary", "cell-merge", "profiling-summary"),
  flowEdge("e-summary-side", "profiling-summary", "side-channels", { speed: "slow" }),
  flowEdge("e-side-summarize", "side-channels", "summarize"),
  flowEdge("e-summarize-report", "summarize", "native-report"),
  flowEdge("e-report-atomic", "native-report", "atomic-json"),
];

/** Section 11 diagram: hot observer lane through retention ribbon, merge, and atomic commit. */
export function MetricsDeepDiveSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [retention, setRetention] = useState<Retention>("exactFold");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="11"
          title="A request becomes a row before it becomes a summary"
          subtitle="Observer callbacks build PendingRequest state; terminal reduction yields RecordIngest, and RunCapture selects retention, fold, merge, and artifact behavior."
        />
        <Segmented
          ariaLabel="Retention"
          value={retention}
          onChange={setRetention}
          options={[
            { id: "retain", label: "Retain" },
            { id: "exactFold", label: "Exact-fold" },
            { id: "sketch", label: "Sketch" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(retention)} edges={edges} height={640} />

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Retain">
          Supports batch ordering and complete per-record materialization.
        </Callout>
        <Callout tone="success" title="Exact-fold">
          Drops clean rows while preserving exact columns and percentiles; dispatch-slot ordinals retain row identity.
        </Callout>
        <Callout tone="warning" title="Sketch">
          Bounded storage keeps exact count/sum/extrema, computes approximate percentiles, and exports an empty
          timeslice set.
        </Callout>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          When enabled, streaming per-record artifacts write at fold completion before a clean row is dropped. Sketch
          plans reject per-record artifacts.
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "observer", path: "rust/runtime/src/metrics.rs" },
          { label: "RunCapture", path: "rust/runtime/src/engine/execute.rs" },
          { label: "accumulator", path: "rust/runtime/src/metrics_core/accumulator.rs" },
          { label: "store", path: "rust/runtime/src/metrics_core/store.rs" },
          { label: "record lane", path: "rust/runtime/src/engine/record_lane.rs" },
          { label: "export order", path: "rust/runtime/src/export/mod.rs" },
        ]}
      />
    </SectionShell>
  );
}
