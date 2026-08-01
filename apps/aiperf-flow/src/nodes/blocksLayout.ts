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
/** Title `<div>`: its pinned `leading-[24px]` line box plus the `mb-2 (8px)` gap below it.
 * The component pins that leading so this stays a contract rather than a font-metric guess. */
const TITLE_H = 32;
/** Strip caption: pinned `leading-[16px]`. */
const STRIP_LABEL_H = 16;
const STRIP_GAP = 10;
const LABEL_GAP = 4;
/** The 1px `border` on each side of the node chrome. */
const BORDER = 2;
/** Three pinned 16px lines, which `BlocksNode` clamps the detail text to. */
export const DETAIL_TEXT_H = 48;
/** The clamped text box plus its `mt-2` (8px). */
const DETAIL_H = DETAIL_TEXT_H + 8;

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
