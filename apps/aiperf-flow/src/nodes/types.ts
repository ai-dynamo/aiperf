/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SurfaceRole, StrokeRole } from "../theme/tokens.js";

/** Every node data shape accepts an optional `className`, merged onto the component's own root classes. */
type BaseNodeData = {
  className?: string;
};

export type HeaderNodeData = BaseNodeData & {
  title: string;
  caption?: string;
  surfaceRole?: SurfaceRole;
};

export type PanelNodeData = BaseNodeData & {
  title: string;
  detail?: string;
  surfaceRole?: SurfaceRole;
  strokeRole?: StrokeRole;
  /** Optional in-card mini-diagram (compose `chalk` `Diagram`/`NodeChip`/… atoms). */
  diagram?: React.ReactNode;
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
