/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the MetricsView page: measurement and exports.

const nodes: Node[] = [
  bandHeader("b-hot", "Hot-path observations", 0, 0),
  panel("sink", "RequestSink<R>", "transport completes request", 0, 60),
  card("callbacks", "RequestObserver callbacks", undefined, "arrival · admit · token · usage · terminal", 300, 60),
  panel("tee", "ObserverTee", "preserve event order", 640, 60),
  card("records", "records", undefined, "optional raw", 900, 60),

  bandHeader("b-accum", "Worker-local accumulation", 0, 200),
  card("collector", "CollectorObserver", undefined, "timing trace + request lifecycle", 0, 260),
  card("native", "NativeMetricsObserver", undefined, "catalog RecordIngest facts", 340, 260),
  card("storage", "storage policy", undefined, "exact retain or t-digest sketch", 680, 260),

  bandHeader("b-drain", "Post-drain reduction", 0, 400),
  panel("partitions", "worker partitions", "plain data after callbacks stop", 0, 460),
  card("accumulator", "MetricsAccumulator", undefined, "merge stores + derived metrics", 300, 460),
  card("sidechannels", "side channels", undefined, "GPU · server · network", 620, 460),
  card("report", "NativeReport", undefined, "typed schema v2", 900, 460),

  bandHeader("b-persist", "Persistence and fan-out", 0, 600),
  card("json", "native-v2.json", undefined, "durable report commit", 0, 660),
  panel("compat", "compat reports", "aiperf JSON + CSV + console", 280, 660),
  panel("columnar", "columnar artifacts", "records + server metrics", 560, 660),
  card("exporters", "network exporters", undefined, "OTLP · MLflow · W&B", 840, 660),
];

const edges: Edge[] = [
  flow("sink", "callbacks"),
  flow("callbacks", "tee"),
  dashed("tee", "records"),
  flow("tee", "collector"),
  flow("tee", "native"),
  flow("native", "storage"),
  flow("partitions", "accumulator"),
  flow("sidechannels", "report"),
  flow("accumulator", "report"),
  flow("report", "json"),
  flow("json", "compat"),
  flow("compat", "columnar"),
  dashed("columnar", "exporters"),
];

/** MetricsView: measurement as an event stream, finalized into a report exporters consume. */
export function MetricsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Measurement and exports">
        Measurement is an event stream, not transport-specific reporting. Request callbacks feed local collectors and
        native metrics; side channels join later, and exporters consume the finalized report.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={600} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Exact mode">
          Retained rows support raw records, timeslices, and byte-exact percentile reporting.
        </Callout>
        <Callout tone="info" title="Sketch mode">
          Rows are folded and dropped; counts and extrema stay exact while percentiles are approximate.
        </Callout>
        <Callout tone="warning" title="Separate artifact path">
          Per-record files are written at capture sites because those rows do not live in the finalized report.
        </Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "Observer adapter", path: "rust/aiperf/src/metrics.rs" },
          { label: "Metrics core", path: "rust/aiperf/src/metrics_core/accumulator.rs" },
          { label: "Report commit", path: "rust/aiperf/src/report.rs" },
          { label: "Export registry", path: "rust/aiperf/src/export/mod.rs" },
        ]}
      />
    </div>
  );
}
