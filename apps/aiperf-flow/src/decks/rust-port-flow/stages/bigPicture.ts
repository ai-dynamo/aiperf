/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 0 — Big Picture. STUB: overview node + caption only. A stage agent owns this file and
//! fills in `subgraph` (the level-1 diagram), optional `leaves` (a deeper zoom level), and real
//! `evidence` source anchors — without editing the deck shell.

import type { StageDef } from "../stage.js";

export const bigPictureStage: StageDef = {
  id: "big-picture",
  order: 0,
  label: "Big Picture",
  caption: "The whole request lifecycle as one connected map — start here, then drill into any stage.",
  tone: "gray",
  // v2 timeline: the pre-run origin marker in the Dataset lane — the run begins here.
  lane: "dataset",
  events: [{ id: "bp-run", label: "run", laneId: "dataset", atOrder: 0, realOffsetMs: 0 }],
};
