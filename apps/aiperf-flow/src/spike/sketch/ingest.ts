/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Step-wise ingestion and traced summarization, for watching both algorithms work.
//!
//! Built *on top of* the pinned port in `sketchSim.ts` rather than inside it: that module is
//! verified byte-for-byte against the Rust through a golden fixture, and instrumenting it for a
//! visual would put presentation concerns inside the thing under test. Everything here calls the
//! same `clustered`/`merge`/`quantileFrom` the fixture pins.

import {
  clustered,
  createDigest,
  exactPercentile,
  kScale,
  merge,
  type Centroid,
  type TDigest,
} from "./sketchSim.js";

/** One arrival: a latency value and the cell that owns it. */
export type Arrival = { index: number; value: number; cell: number };

export type IngestState = {
  /** Every value that has arrived, in arrival order. The exact path keeps all of them. */
  arrived: number[];
  /** Values sorted — what the exact percentile actually reads. */
  sorted: number[];
  /** One digest per cell. Each summarizes only its own slice. */
  cells: TDigest[];
  /** Which cell took the most recent value, for highlighting. */
  lastCell: number | null;
  /** The most recent value, for highlighting. */
  lastValue: number | null;
  /** Index in `sorted` the last value landed at. */
  lastSortedIndex: number | null;
  /** Cells whose centroid count fell on the most recent step — a compression fired there. */
  compressedCells: number[];
};

export function createIngest(cells: number, compression: number): IngestState {
  return {
    arrived: [],
    sorted: [],
    cells: Array.from({ length: cells }, () => createDigest(compression)),
    lastCell: null,
    lastValue: null,
    lastSortedIndex: null,
    compressedCells: [],
  };
}

/** Insert into a sorted array, returning the index it landed at. */
function insertSorted(sorted: number[], value: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid]! < value) lo = mid + 1;
    else hi = mid;
  }
  sorted.splice(lo, 0, value);
  return lo;
}

/**
 * Take one arrival into both structures.
 *
 * The digest is compressed eagerly on every step rather than at the runtime's threshold. The
 * runtime batches — it lets weight-1 centroids accumulate to `compress_threshold` and clusters in
 * one pass, which is faster and reaches the same place because clustering sorts first. Compressing
 * per value shows the clustering happening instead of hiding it inside one bulk step; the
 * resulting digest is the same either way.
 */
export function ingestOne(state: IngestState, arrival: Arrival): IngestState {
  const cells = state.cells.map((c) => ({ ...c, centroids: c.centroids.map((x) => ({ ...x })) }));
  const target = cells[arrival.cell]!;
  const before = target.centroids.length;

  target.centroids.push({ mean: arrival.value, weight: 1 });
  target.totalWeight += 1;
  if (arrival.value < target.min) target.min = arrival.value;
  if (arrival.value > target.max) target.max = arrival.value;
  target.centroids = clustered(target);

  const sorted = [...state.sorted];
  const at = insertSorted(sorted, arrival.value);

  return {
    arrived: [...state.arrived, arrival.value],
    sorted,
    cells,
    lastCell: arrival.cell,
    lastValue: arrival.value,
    lastSortedIndex: at,
    // A cluster absorbed the new centroid when the count did not grow.
    compressedCells: target.centroids.length <= before ? [arrival.cell] : [],
  };
}

/** How the exact percentile is computed — every intermediate the formula uses. */
export type ExactTrace = {
  count: number;
  percentile: number;
  virtualIndex: number;
  lo: number;
  hi: number;
  frac: number;
  loValue: number;
  hiValue: number;
  result: number;
};

/**
 * Trace `exactPercentile`, exposing the two values it interpolates between.
 *
 * Type-7: a virtual index into the sorted array, then a linear blend of the two neighbours it
 * falls between. The whole computation reads exactly two of the retained values — which is both
 * why it is exact and why it needs all of them kept.
 */
export function traceExact(sorted: readonly number[], percentile: number): ExactTrace | null {
  if (sorted.length === 0) return null;
  const virtualIndex = (percentile / 100) * (sorted.length - 1);
  const lo = Math.floor(virtualIndex);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return {
    count: sorted.length,
    percentile,
    virtualIndex,
    lo,
    hi,
    frac: virtualIndex - lo,
    loValue: sorted[lo]!,
    hiValue: sorted[hi]!,
    result: exactPercentile(sorted, percentile),
  };
}

/** One centroid visited while walking to a quantile. */
export type WalkStep = {
  centroid: Centroid;
  /** Weight accumulated before this centroid. */
  cumulativeBefore: number;
  /** The quantile at this centroid's centre — what the walk compares against. */
  centerQ: number;
  /** True for the centroid the walk stopped on. */
  stopped: boolean;
};

/** How the sketch answers the same question, and from what. */
export type SketchTrace = {
  quantile: number;
  totalWeight: number;
  steps: WalkStep[];
  /** The bracketing pair the result interpolates between, in (quantile, value) space. */
  fromQ: number;
  fromValue: number;
  toQ: number;
  toValue: number;
  result: number;
  /** True when the answer came straight off an exact anchor rather than an interpolation. */
  anchored: boolean;
};

/**
 * Trace a quantile walk over a digest.
 *
 * Mirrors `quantileFrom`: accumulate weight, compare the running centre quantile against the
 * target, and interpolate between the previous anchor and the centroid that overshoots. Anchors
 * are `(0, min)` and `(1, max)`, which is why q0 and q1 are exact rather than estimated.
 */
export function traceSketch(digest: TDigest, q: number): SketchTrace | null {
  if (digest.totalWeight === 0) return null;
  const target = Math.min(1, Math.max(0, q));
  const centroids = clustered(digest);
  const total = digest.totalWeight;

  if (target <= 0 || target >= 1) {
    const value = target <= 0 ? digest.min : digest.max;
    return {
      quantile: target,
      totalWeight: total,
      steps: [],
      fromQ: target,
      fromValue: value,
      toQ: target,
      toValue: value,
      result: value,
      anchored: true,
    };
  }

  const steps: WalkStep[] = [];
  let cumulative = 0;
  let prevQ = 0;
  let prevValue = digest.min;
  for (const centroid of centroids) {
    const centerQ = (cumulative + centroid.weight / 2) / total;
    const stopped = target < centerQ;
    steps.push({ centroid, cumulativeBefore: cumulative, centerQ, stopped });
    if (stopped) {
      const result =
        prevValue + ((centroid.mean - prevValue) * (target - prevQ)) / (centerQ - prevQ);
      return {
        quantile: target,
        totalWeight: total,
        steps,
        fromQ: prevQ,
        fromValue: prevValue,
        toQ: centerQ,
        toValue: centroid.mean,
        result,
        anchored: false,
      };
    }
    prevQ = centerQ;
    prevValue = centroid.mean;
    cumulative += centroid.weight;
  }
  const result = prevValue + ((digest.max - prevValue) * (target - prevQ)) / (1 - prevQ);
  return {
    quantile: target,
    totalWeight: total,
    steps,
    fromQ: prevQ,
    fromValue: prevValue,
    toQ: 1,
    toValue: digest.max,
    result,
    anchored: false,
  };
}

/** The folded digest, plus the per-cell centroids that went into it. */
export type FoldTrace = {
  /** Centroids contributed by each cell, before compression. */
  contributed: { cell: number; centroids: Centroid[] }[];
  /** All of them concatenated, in cell order — the literal intermediate `merge` builds. */
  concatenated: Centroid[];
  /** After clustering. */
  folded: TDigest;
};

/**
 * Fold the cells, keeping the intermediate visible.
 *
 * `merge` is concatenate-then-compress. Showing the concatenated pile before the compression is
 * the only way to see that the fold does no arithmetic on the values themselves — it just pools
 * summaries and re-clusters them.
 */
export function traceFold(cells: readonly TDigest[], compression: number): FoldTrace {
  const contributed = cells.map((cell, index) => ({
    cell: index,
    centroids: clustered(cell),
  }));
  const concatenated = contributed.flatMap((c) => c.centroids);
  const folded = createDigest(compression);
  for (const cell of cells) merge(folded, cell);
  return { contributed, concatenated, folded };
}

/**
 * The quantile band each cluster is allowed to span, as the K1 rule sees it.
 *
 * Exposed so the page can show *why* a merge was accepted or refused at the moment it happens,
 * rather than only its outcome.
 */
export function clusterBudget(q: number, compression: number): number {
  return kScale(q, compression);
}

/** Deterministic arrivals: latency values dealt round-robin across cells. */
export function arrivals(values: readonly number[], cells: number): Arrival[] {
  return values.map((value, index) => ({ index, value, cell: index % cells }));
}
