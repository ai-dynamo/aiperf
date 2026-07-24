/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 8 — Aggregation → final results. STUB: overview node + caption + verified evidence
//! anchors. A stage agent owns this file and fills in `subgraph` and optional `leaves` (e.g. the
//! exact-fold vs t-digest-sketch comparison as a deeper zoom level).

import type { StageDef } from "../stage.js";

export const aggregationStage: StageDef = {
  id: "aggregation",
  order: 8,
  label: "Aggregation → results",
  caption:
    "worker-local NativeMetricsObserver → metrics_core NaN-sparse column store → exact folds vs t-digest (cellular::sketch::TDigest) sketch → NativeReporter → NativeReport → ExporterRegistry (nine sinks) → RunTerminalV2.",
  tone: "green",
  evidence: [
    { label: "struct NativeMetricsObserver", path: "runtime/src/metrics.rs:203" },
    { label: "TDigest (cellular::sketch)", path: "runtime/src/cellular/mod.rs:33" },
    { label: "struct NativeReporter", path: "runtime/src/metrics_core/report.rs:1031" },
    { label: "struct ExporterRegistry", path: "runtime/src/export/mod.rs:258" },
  ],
};
