/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 2 — Dataset loading. STUB: overview node + caption only. A stage agent owns this file and
//! fills in `subgraph`, optional `leaves`, and real `evidence` anchors (the SegmentStore/Handle/Turn
//! paths are to be pinned to exact file:line during that stage's implementation).

import type { StageDef } from "../stage.js";

export const datasetStage: StageDef = {
  id: "dataset",
  order: 2,
  label: "Dataset loading",
  caption:
    "Loaders → SegmentStore (six disjoint BLAKE3 content domains, prefix-folded hashing) → dense integer Handles → Turn/body freeze.",
  tone: "green",
};
