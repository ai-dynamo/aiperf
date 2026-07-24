/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of AIPerf's measurement plane: request callbacks feed worker-local
// collectors, fold into a finalized report, and fan out to exporters. Each spoke is one beat.

/** MetricsView: measurement as an event stream, finalized into a report exporters consume. */
export function MetricsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Measurement and exports">
        Measurement is an event stream, not transport-specific reporting. Request callbacks feed local collectors and
        native metrics; side channels join later, and exporters consume the finalized report.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · MEASUREMENT",
          title: "How is each request measured?",
          body: "An event stream folded per worker, finalized once, then fanned out.",
        }}
        liveWire={0}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Hot-path observations",
            diagram: (
              <Diagram>
                <NodeChip>SINK</NodeChip>
                <MiniArrow />
                <NodeChip accent>OBSERVER</NodeChip>
              </Diagram>
            ),
            children: "RequestSink<R> completion drives RequestObserver: arrival · admit · token · usage · terminal.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "ObserverTee",
            diagram: (
              <Diagram>
                <NodeChip accent>TEE</NodeChip>
                <MiniArrow />
                <NodeChip>records</NodeChip>
              </Diagram>
            ),
            children: "The tee preserves event order and forks optional raw records off the hot path.",
          },
          {
            accent: "green",
            badge: 3,
            title: "CollectorObserver",
            diagram: (
              <Diagram>
                <NodeChip>token</NodeChip>
                <MiniArrow />
                <MiniBars heights={[30, 58, 88, 70]} />
              </Diagram>
            ),
            children: "Worker-local timing trace plus per-request lifecycle, accumulated without contention.",
          },
          {
            accent: "purple",
            badge: 4,
            title: "NativeMetricsObserver",
            diagram: (
              <Diagram>
                <NodeChip accent>INGEST</NodeChip>
                <MiniArrow />
                <NodeChip>exact · sketch</NodeChip>
              </Diagram>
            ),
            children: "Catalogs RecordIngest facts under the storage policy: exact retain or t-digest sketch.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "MetricsAccumulator",
            diagram: (
              <Diagram>
                <MiniBars heights={[44, 72, 100, 60]} />
                <MiniArrow />
                <NodeChip accent>derived</NodeChip>
              </Diagram>
            ),
            children: "After drain, worker partitions merge into derived metrics; GPU · server · network join here.",
          },
          {
            accent: "red",
            badge: 6,
            title: "NativeReport",
            diagram: (
              <Diagram>
                <NodeChip accent>REPORT</NodeChip>
                <MiniArrow />
                <NodeChip>native-v2.json</NodeChip>
              </Diagram>
            ),
            children: "One typed schema-v2 commit, plus compat aiperf JSON · CSV · console and columnar records.",
          },
          {
            accent: "orange",
            badge: 7,
            title: "Network exporters",
            diagram: (
              <Diagram>
                <NodeChip>OTLP</NodeChip>
                <MiniArrow />
                <NodeChip accent>W&B</NodeChip>
              </Diagram>
            ),
            children: "OTLP · MLflow · W&B sinks stream the finalized report to external systems.",
          },
        ]}
      />

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
