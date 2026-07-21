/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SurfaceRole, StrokeRole } from "../theme/tokens.js";

export type HeaderNodeData = {
  title: string;
  caption?: string;
  surfaceRole?: SurfaceRole;
};

export type PanelNodeData = {
  title: string;
  detail?: string;
  surfaceRole?: SurfaceRole;
  strokeRole?: StrokeRole;
};

export type ChipNodeData = {
  label: string;
  strokeRole?: StrokeRole;
};

export type CardNodeData = {
  title: string;
  detail?: string;
  subtitle?: string;
  strokeRole?: StrokeRole;
};
