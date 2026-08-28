/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Node, Edge } from "@xyflow/react";
import { DeckDiagram, EvidenceRow, PageIntro } from "./shared.js";

//! Overview page: the complete end-to-end pipeline from S3/local source through to exported results.

const LAYOUT = { direction: "RIGHT" as const };

const NODES: Node[] = [
  // Source layer
  { id: "s3", type: "panel", position: { x: 0, y: 0 }, data: { title: "S3 / Local Source", detail: "paginate bucket listing or read local paths; one file = one partition", surfaceRole: "elevated" } },
  { id: "fmt", type: "panel", position: { x: 0, y: 100 }, data: { title: "Dynamo Format Decoder", detail: "decompress gzip, parse JSONL; emit request units with event timestamps", surfaceRole: "elevated" } },
  { id: "chkpt-backend", type: "card", position: { x: 0, y: 200 }, data: { title: "Checkpoint Backend", subtitle: "LOCAL / NONE / CAS", detail: "persists per-partition read cursor; survives restarts", strokeRole: "secondary" } },

  // Session layer
  { id: "conv", type: "panel", position: { x: 300, y: 0 }, data: { title: "Conversation Coordinator", detail: "joins fragments across chunk boundaries by session key; folds endpoint replies into transcript" } },
  { id: "clo", type: "card", position: { x: 300, y: 120 }, data: { title: "Session Closure Policy", subtitle: "P1B", detail: "finite seal closes sessions with no causal gaps; quarantine on failure", strokeRole: "secondary" } },
  { id: "host", type: "panel", position: { x: 300, y: 240 }, data: { title: "Action Host", detail: "holds per-session turn state; emits Request actions when a turn is ready to execute" } },

  // Closure seam
  { id: "tci", type: "card", position: { x: 600, y: 0 }, data: { title: "Turn Closure Intake", subtitle: "P2", detail: "Rc<RefCell<VecDeque>>; zero-copy worker-local closed-turn queue", strokeRole: "secondary" } },

  // Pipeline
  { id: "pl", type: "panel", position: { x: 900, y: 0 }, data: { title: "Streaming Pipeline", detail: "fused select_biased! loop: settlement → actions → stop → barrier → admit; bounded capacity backpressure" } },
  { id: "place", type: "card", position: { x: 900, y: 140 }, data: { title: "Local Placement", subtitle: "P3", detail: "routes each action to the correct worker thread via five placement traits", strokeRole: "secondary" } },
  { id: "xport", type: "card", position: { x: 900, y: 260 }, data: { title: "Cellular Transport", subtitle: "C2 / C3", detail: "bounded mux transfer channel + sticky session placement across cell processes", strokeRole: "secondary" } },

  // Shadow replay
  { id: "sink", type: "card", position: { x: 1200, y: 0 }, data: { title: "ScheduledRequestSink", subtitle: "P4", detail: "issues one real HTTP/gRPC request per action; respects original inter-request timing offsets" } },
  { id: "inv", type: "panel", position: { x: 1200, y: 130 }, data: { title: "Action Inventory Ledger", detail: "dense gap-closure: tracks in-flight, done, and failed actions for ordered delivery" } },

  // Results
  { id: "epoch", type: "panel", position: { x: 1500, y: 0 }, data: { title: "Epoch Coordinator", detail: "rotates result epochs for long-running streams; provisionally holds in-flight holes" } },
  { id: "compact", type: "card", position: { x: 1500, y: 120 }, data: { title: "Result Compactor", subtitle: "6C1", detail: "deterministic finalization of completed epochs → per-session scored receipts", strokeRole: "secondary" } },
  { id: "deliver", type: "card", position: { x: 1500, y: 240 }, data: { title: "Delivery Restart", subtitle: "6C2", detail: "target-idempotent delivery; safe to replay on restart without duplicating exports", strokeRole: "secondary" } },

  // Output
  { id: "out", type: "chip", position: { x: 1800, y: 80 }, data: { label: "JSON · Parquet · Metrics" } },
];

const EDGES: Edge[] = [
  { id: "e-s3-fmt", source: "s3", target: "fmt", type: "flow", label: "partitions" },
  { id: "e-fmt-conv", source: "fmt", target: "conv", type: "flow", label: "decoded units" },
  { id: "e-chkpt-conv", source: "chkpt-backend", target: "conv", type: "flow", data: { speed: "slow", color: "var(--color-stroke-tertiary)" } },
  { id: "e-conv-clo", source: "conv", target: "clo", type: "flow" },
  { id: "e-clo-host", source: "clo", target: "host", type: "flow" },
  { id: "e-host-tci", source: "host", target: "tci", type: "flow", label: "turn receipts" },
  { id: "e-tci-pl", source: "tci", target: "pl", type: "flow" },
  { id: "e-pl-place", source: "pl", target: "place", type: "flow" },
  { id: "e-pl-xport", source: "pl", target: "xport", type: "flow", data: { speed: "slow", color: "var(--color-stroke-tertiary)" } },
  { id: "e-place-sink", source: "place", target: "sink", type: "flow", label: "actions" },
  { id: "e-xport-sink", source: "xport", target: "sink", type: "flow", data: { speed: "slow", color: "var(--color-stroke-tertiary)" } },
  { id: "e-sink-inv", source: "sink", target: "inv", type: "flow" },
  { id: "e-inv-epoch", source: "inv", target: "epoch", type: "flow", label: "receipts" },
  { id: "e-epoch-compact", source: "epoch", target: "compact", type: "flow" },
  { id: "e-compact-deliver", source: "compact", target: "deliver", type: "flow" },
  { id: "e-deliver-out", source: "deliver", target: "out", type: "flow", label: "export" },
];

/** End-to-end pipeline overview. */
export function OverviewPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="End-to-end streaming shadow replay">
        A Dynamo request-trace file (local or on S3) is one streaming source. The decoder emits request units; the
        conversation coordinator joins them by session key; the action host fires each turn as a{" "}
        <code>Request</code> action through the pipeline; the scheduled-request sink issues it against a real
        endpoint; the result compactor finalizes per-session scored receipts and hands them to the delivery layer for
        idempotent export. A local checkpoint backend keeps the read cursor so any stage can resume after a restart.
      </PageIntro>
      <DeckDiagram nodes={NODES} edges={EDGES} height={460} layout={LAYOUT} />
      <EvidenceRow
        items={[
          { label: "Source seam", path: "rust/runtime/src/streaming/source.rs" },
          { label: "Session coordinator", path: "rust/runtime/src/streaming/session/conversation.rs" },
          { label: "Pipeline", path: "rust/runtime/src/streaming/pipeline.rs" },
          { label: "Shadow replay", path: "rust/runtime/src/engine/streaming_execution.rs" },
          { label: "Result compactor", path: "rust/runtime/src/streaming/results/compactor.rs" },
        ]}
      />
    </div>
  );
}
