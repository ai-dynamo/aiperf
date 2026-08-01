/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cell metrics and box sizing for the `blocks` node type, shared by the component and the
//! authoring helper so a `blocks` node declares its size before React Flow measures it.

import type { CategoryRole } from "../theme/tokens.js";

/** One labelled row of per-block tags. */
export type BlockStrip = {
  /** Row caption, e.g. "parent chain". Also the React key, so it must be unique per node. */
  label: string;
  cells: CategoryRole[];
};

export const CELL_W = 13;
export const CELL_H = 16;
export const CELL_GAP = 2;

const PAD_X = 16;
const PAD_Y = 14;
/** Title `<div>`: `text-sm` line-height (20px) plus its `mb-2` gap (8px). */
const TITLE_H = 28;
const STRIP_LABEL_H = 16;
const STRIP_GAP = 10;
const LABEL_GAP = 4;
/** The 1px `border` on each side of the node chrome. */
const BORDER = 2;
/** Roughly two lines of the 12px detail text, plus its top margin. */
const DETAIL_H = 40;

export type BlocksLayoutInput = {
  strips: readonly BlockStrip[];
  hasTitle: boolean;
  hasDetail: boolean;
};

/**
 * The box a `blocks` node will occupy.
 *
 * Width comes from the longest strip; a `detail` line is allotted a fixed two-line block rather
 * than measured, which is why `BlocksNode` caps its detail at `max-w-[420px]`.
 */
export function blocksNodeSize({
  strips,
  hasTitle,
  hasDetail,
}: BlocksLayoutInput): { width: number; height: number } {
  const widest = Math.max(0, ...strips.map((s) => s.cells.length));
  const stripWidth = Math.max(0, widest * (CELL_W + CELL_GAP) - CELL_GAP);
  const stripHeight = STRIP_LABEL_H + LABEL_GAP + CELL_H;
  const stackHeight = Math.max(
    0,
    strips.length * stripHeight + Math.max(0, strips.length - 1) * STRIP_GAP,
  );

  return {
    width: Math.max(stripWidth, hasDetail ? 420 : 0) + 2 * PAD_X + BORDER,
    height:
      stackHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0) + (hasDetail ? DETAIL_H : 0),
  };
}
