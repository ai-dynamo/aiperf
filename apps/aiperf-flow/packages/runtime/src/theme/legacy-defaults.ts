// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Visual literals retained for documents without an active theme.

/** Existing contribution colors used when no theme is selected. */
export const LEGACY_VISUAL_FALLBACKS = Object.freeze({
  queueLane: "#111827",
  queueWaiting: "#64748b",
  queueServing: "#22c55e",
  waterfallPoint: "#7dcfff",
  waterfallInterval: "#38bdf8",
  waterfallText: "#f8fafc",
  waterfallPlayhead: "#fbbf24",
  segmentFill: "#334155",
  segmentText: "#f8fafc",
  segmentContinuation: "#38bdf8",
  spanUncovered: "#ef4444",
  spanCovered: "#94a3b8",
  spanEdge: "#38bdf8",
  glyphFill: "#f8fafc",
  morphFill: "#38bdf8",
} as const);
