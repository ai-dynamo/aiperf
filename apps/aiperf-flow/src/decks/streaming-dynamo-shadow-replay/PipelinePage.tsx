/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow, RoundNode, DbNode } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

//! Pipeline + Results page: StreamingPipeline, placement, cellular transport, epoch/compactor/delivery.

/** Pipeline and result plane. */
export function PipelinePage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Deliver actions, compact results, export">
        The streaming pipeline is a fused event loop that drains settlement, action, stop, barrier, and admit events
        in priority order. It places each action on the correct worker thread through a set of five placement traits.
        In cellular mode a bounded multiplexed transfer channel ships actions to remote cell processes via the sticky
        session placement policy. Results flow through an epoch coordinator, a deterministic compactor, and an
        idempotent delivery layer that is safe to restart without duplicating exports.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "STREAMING · PIPELINE & RESULTS",
          title: "How do actions move from the session layer to exported results?",
          body: "Fused event loop → placement → sink → epoch rotate → compact → idempotent delivery.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Fused event loop (P3)",
            diagram: (
              <Diagram>
                <NodeChip>settle</NodeChip>
                <MiniArrow />
                <NodeChip accent>admit</NodeChip>
              </Diagram>
            ),
            children:
              "select_biased! over five event types: settlement (checkpoint), actions, stop signal, barrier advancement, and admission. Backpressure via a bounded capacity limit stalls acquisition when the sink falls behind.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Local placement (P3)",
            diagram: (
              <Diagram>
                <NodeChip accent>action</NodeChip>
                <MiniArrow />
                <RoundNode>W</RoundNode>
              </Diagram>
            ),
            children:
              "Five placement traits route each action to the right worker: sticky-session, round-robin, capacity-weighted, hash-based, and explicit-worker. LocalStreamingPlacement owns the routing table.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Cellular transport (C2)",
            diagram: (
              <Diagram>
                <NodeChip>cell 0</NodeChip>
                <MiniArrow />
                <NodeChip accent>cell 1</NodeChip>
              </Diagram>
            ),
            children:
              "CellularStreamingTransport ships actions to remote cell processes over a bounded multiplexed channel. Each channel slot is bounded; backpressure propagates to the pipeline's admission event.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Sticky placement (C3)",
            diagram: (
              <Diagram>
                <NodeChip>session</NodeChip>
                <MiniArrow />
                <NodeChip accent>same cell</NodeChip>
              </Diagram>
            ),
            children:
              "All actions for a session land on the same cell process. The ownership epoch (C4) makes cell migration crash-safe: staged ownership commits only after the previous owner acknowledges drain.",
          },
          {
            accent: "orange",
            badge: 5,
            title: "Epoch coordinator (6B)",
            diagram: (
              <Diagram>
                <DbNode>epoch N</DbNode>
                <MiniArrow />
                <DbNode accent>epoch N+1</DbNode>
              </Diagram>
            ),
            children:
              "Rotates result epochs on a configured cadence. In-flight sessions produce provisional holes in the current epoch; holes fill when the session finalizes or are sealed as durable holes on rotation.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Result compactor (6C1)",
            diagram: (
              <Diagram>
                <NodeChip>epoch</NodeChip>
                <MiniArrow />
                <NodeChip accent>receipts</NodeChip>
              </Diagram>
            ),
            children:
              "Deterministically finalizes a completed epoch into an ordered set of per-session scored receipts. Compaction is idempotent: re-running it on the same epoch produces the identical output.",
          },
          {
            accent: "yellow",
            badge: 7,
            title: "Idempotent delivery (6C2)",
            diagram: (
              <Diagram>
                <NodeChip>receipt</NodeChip>
                <MiniArrow />
                <NodeChip accent>export</NodeChip>
              </Diagram>
            ),
            children:
              "DeliveryRestart writes results to configured exporters (JSON, Parquet, metrics) with target-idempotent semantics. A restarted run re-delivers without duplicating output records.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Pipeline", path: "rust/runtime/src/streaming/pipeline.rs" },
          { label: "Placement", path: "rust/runtime/src/streaming/placement.rs" },
          { label: "Epoch coordinator", path: "rust/runtime/src/streaming/results/epoch.rs" },
          { label: "Compactor", path: "rust/runtime/src/streaming/results/compactor.rs" },
          { label: "Delivery", path: "rust/runtime/src/streaming/results/delivery.rs" },
        ]}
      />
    </div>
  );
}
