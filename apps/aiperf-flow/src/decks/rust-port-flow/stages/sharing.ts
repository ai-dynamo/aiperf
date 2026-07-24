/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 3 — Sharing the dataset with workers. STUB: overview node + caption only. A stage agent
//! owns this file and fills in `subgraph`, optional `leaves`, and real `evidence` anchors (the
//! frozen-SegmentStore / content_server sidecar paths are to be pinned during implementation).

import type { StageDef } from "../stage.js";

export const sharingStage: StageDef = {
  id: "sharing",
  order: 3,
  label: "Sharing the dataset",
  caption:
    "The frozen SegmentStore: bytes live exactly once; turns carry Handles not bytes for zero-copy sharing across worker threads (content_server is a separate media sidecar).",
  tone: "cyan",
  lane: "dataset",
  events: [{ id: "sh-handles", label: "Handles", laneId: "dataset", atOrder: 4, realOffsetMs: 44 }],
};
