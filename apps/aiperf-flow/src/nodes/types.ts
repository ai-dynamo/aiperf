/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SurfaceRole, StrokeRole } from "../theme/tokens.js";
import type { TimelineBar, TimelineGap } from "./timelineLayout.js";

/** Every node data shape accepts an optional `className`, merged onto the component's own root classes. */
type BaseNodeData = {
  className?: string;
};

export type HeaderNodeData = BaseNodeData & {
  title: string;
  caption?: string;
};

export type PanelNodeData = BaseNodeData & {
  title: string;
  detail?: string;
  surfaceRole?: SurfaceRole;
  strokeRole?: StrokeRole;
  /** Optional in-card mini-diagram (compose `chalk` `Diagram`/`NodeChip`/… atoms). */
  diagram?: React.ReactNode;
};

/**
 * A time-scaled swimlane Gantt: overlap, idle gaps, and clock warp — claims a box-and-arrow
 * diagram cannot make. Plain data (no JSX), so declarative `.ts` decks can author one.
 */
export type TimelineNodeData = BaseNodeData & {
  /** Optional heading above the chart. Its presence changes the node's height. */
  title?: string;
  /** Swimlane order, top to bottom; also fixes each lane's hue. */
  lanes: string[];
  bars: TimelineBar[];
  /** Idle bands drawn over the raw block. */
  gaps?: TimelineGap[];
  /** Draw the second, warped-clock block. Defaults to true; false gives a single-clock chart. */
  showWarp?: boolean;
  rawLabel?: string;
  warpLabel?: string;
  surfaceRole?: SurfaceRole;
  /** Drawing width before the per-second scale is clamped; see `DEFAULT_TIMELINE_WIDTH`. */
  width?: number;
  ariaLabel?: string;
};

export type ChipNodeData = BaseNodeData & {
  label: string;
  strokeRole?: StrokeRole;
};

export type CardNodeData = BaseNodeData & {
  title: string;
  detail?: string;
  subtitle?: string;
  strokeRole?: StrokeRole;
  /** Optional in-card mini-diagram (compose `chalk` `Diagram`/`NodeChip`/… atoms). */
  diagram?: React.ReactNode;
};
