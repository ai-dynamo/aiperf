/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Flattening and sizing for the `ragged` node type: variable-length per-record lists packed into
//! one flat value array plus an offsets table and an owner index — the CSR layout that lets a
//! whole dataset be filtered with a mask instead of a per-record loop.

import type { CategoryRole } from "../theme/tokens.js";

/** Cycled per-record hues, so a flat cell's colour names its owner. */
export const RAGGED_ROLES: readonly CategoryRole[] = ["blue", "orange", "green", "purple", "yellow"];

export function recordRole(record: number): CategoryRole {
  return RAGGED_ROLES[record % RAGGED_ROLES.length]!;
}

export type FlattenedRagged = {
  /** Every value, in record order. */
  values: number[];
  /** `recordIndices[i]` owns `values[i]` — the column a boolean mask is applied to. */
  recordIndices: number[];
  /** Start index of each record's run, or -1 for a record that contributed nothing. */
  offsets: number[];
};

/**
 * Pack per-record lists into the flat triple.
 *
 * An empty record gets offset -1 rather than the next start: a real start would be
 * indistinguishable from the following record's, and "absent" is a distinct state from "empty run
 * that happens to begin here".
 */
export function flattenRagged(lists: ReadonlyArray<readonly number[]>): FlattenedRagged {
  const values: number[] = [];
  const recordIndices: number[] = [];
  const offsets: number[] = [];
  lists.forEach((list, record) => {
    if (list.length === 0) {
      offsets.push(-1);
      return;
    }
    offsets.push(values.length);
    for (const v of list) {
      values.push(v);
      recordIndices.push(record);
    }
  });
  return { values, recordIndices, offsets };
}

/** Running prefix sum, which resets at each record boundary in the flat layout. */
export function cumulative(list: readonly number[]): number[] {
  let acc = 0;
  return list.map((v) => (acc += v));
}

const CELL_W = 34;
const CELL_H = 26;
const CELL_GAP = 3;
const ROW_LABEL_W = 96;
const ROW_GAP = 4;
const SECTION_GAP = 14;
const SECTION_LABEL_H = 20;
const PAD_X = 16;
const PAD_Y = 14;
/** Title `<div>`: its pinned `leading-[24px]` line box plus the `mb-1.5 (6px)` gap below it.
 * The component pins that leading so this stays a contract rather than a font-metric guess. */
const TITLE_H = 30;
/** The 1px `border` on each side of the node chrome. */
const BORDER = 2;

export type RaggedLayoutInput = {
  lists: ReadonlyArray<readonly number[]>;
  hasTitle: boolean;
  /** Also draw the flat value array and its owner-index row. */
  showFlat: boolean;
};

/**
 * The box a `ragged` node will occupy.
 *
 * Width is driven by the flat row when shown, since it is by construction at least as long as the
 * longest per-record row.
 */
export function raggedNodeSize({
  lists,
  hasTitle,
  showFlat,
}: RaggedLayoutInput): { width: number; height: number } {
  const total = lists.reduce((n, l) => n + l.length, 0);
  const longest = Math.max(0, ...lists.map((l) => l.length));
  const cells = showFlat ? Math.max(total, longest) : longest;
  const gridWidth = Math.max(0, cells * (CELL_W + CELL_GAP) - CELL_GAP);

  // One gap per row, not per row-pair: the section label is itself a flex child, so N rows sit
  // below it behind N gaps.
  const perRecordHeight = SECTION_LABEL_H + lists.length * (CELL_H + ROW_GAP);
  // Flat values, owner indices, and the offsets table.
  const flatHeight = showFlat ? SECTION_GAP + SECTION_LABEL_H + 3 * (CELL_H + ROW_GAP) : 0;

  return {
    width: ROW_LABEL_W + gridWidth + 2 * PAD_X + BORDER,
    height: perRecordHeight + flatHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0),
  };
}

/** Shared with `RaggedNode` so the rendered flex gaps cannot drift from the sizing formula. */
export const RAGGED_CELL = {
  width: CELL_W,
  height: CELL_H,
  gap: CELL_GAP,
  rowGap: ROW_GAP,
  labelWidth: ROW_LABEL_W,
};
