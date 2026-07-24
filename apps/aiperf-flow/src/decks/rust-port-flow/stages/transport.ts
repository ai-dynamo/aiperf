/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 6 — The Transport seam. STUB: overview node + caption + verified evidence anchors. A
//! stage agent owns this file and fills in `subgraph` and optional `leaves` (e.g. Transport → HTTP
//! hyper/SSE-decode/reduce internals as a deeper zoom level).

import type { StageDef } from "../stage.js";

export const transportStage: StageDef = {
  id: "transport",
  order: 6,
  label: "Transport seam",
  caption:
    "A transport implements exactly two traits (WorkerSink + ExecutionSinkBuilder); everything else is shared. Four targets: HTTP (hyper, streaming), gRPC (Tonic, non-streaming), dry-run, dynosim (offline co-sim).",
  tone: "yellow",
  lane: "transport",
  events: [{ id: "tp-dispatch", label: "dispatch", laneId: "transport", atOrder: 8, realOffsetMs: 62 }],
  evidence: [
    { label: "trait WorkerSink", path: "runtime/src/engine/turn_execution.rs:74" },
    { label: "trait ExecutionSinkBuilder", path: "runtime/src/engine/turn_execution.rs:136" },
    { label: "struct TransportSink (HTTP)", path: "runtime/src/transport/http/sink.rs:164" },
    { label: "struct GrpcTransportSink", path: "runtime/src/transport/grpc/sink.rs:102" },
  ],
};
