/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 7 — The request hot-path. STUB: overview node + caption + verified evidence anchors. A
//! stage agent owns this file and fills in `subgraph` and optional `leaves`.

import type { StageDef } from "../stage.js";

export const hotPathStage: StageDef = {
  id: "hotpath",
  order: 7,
  label: "Request hot-path",
  caption:
    "ScheduledRuntime/Workload (RequestRateWorkload etc.) → SlotPool + StopChecker admission → Rc<dyn Dispatcher> → the chosen sink → shared reduce_parsed_response → shared measure. TTFT = first token observation.",
  tone: "red",
  lane: "server",
  events: [
    { id: "hp-send", label: "send", laneId: "server", atOrder: 9, realOffsetMs: 64 },
    { id: "hp-ttft", label: "TTFT", laneId: "server", atOrder: 10, realOffsetMs: 121 },
    { id: "hp-reduce", label: "reduce", laneId: "server", atOrder: 11, realOffsetMs: 205 },
  ],
  evidence: [
    { label: "struct RequestRateWorkload", path: "runtime/src/request_rate.rs:140" },
    { label: "struct SlotPool", path: "runtime/src/timing/slots.rs:105" },
    { label: "struct StopChecker", path: "runtime/src/timing/stop.rs:164" },
    { label: "trait Dispatcher", path: "runtime/src/transport/core/dispatch.rs:332" },
  ],
};
