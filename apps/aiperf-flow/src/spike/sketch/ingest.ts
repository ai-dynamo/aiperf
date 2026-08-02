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
  /**
   * Every value that has arrived, in arrival order and unsorted.
   *
   * This is what exact mode actually holds: an append-only `Vec`. Nothing keeps it ordered during
   * the run — `kernel.rs` sorts once, at summarize time, when a percentile is finally asked for.
   */
  arrived: number[];
  /** One digest per cell. Each summarizes only its own slice. */
  cells: TDigest[];
  /**
   * Per cell, how many leading centroids are settled — sorted and clustered by the last compress.
   *
   * Everything after this index is the raw appended tail: weight-1 centroids in arrival order,
   * not yet sorted against anything. The Rust says so directly — centroids are "sorted by mean
   * after every compress" and "may hold transient unsorted weight-1 centroids between
   * compressions".
   */
  settled: number[];
  /** Which cell took the most recent value, for highlighting. */
  lastCell: number | null;
  /** The most recent value, for highlighting. */
  lastValue: number | null;
  /** Cells that crossed their threshold on this step and ran a bulk compress. */
  compressedCells: number[];
  /** Centroid count immediately before the most recent compress, for showing what it collapsed. */
  collapsedFrom: number | null;
};

/**
 * Build the ingest state.
 *
 * `threshold` overrides the digest's own `compress_threshold`, which the Rust derives as
 * `max(64, δ × 10)` — 1000 at production's δ=100. A thousand raw centroids accumulating before
 * anything happens is the real behaviour and completely unwatchable, so the page turns the same
 * knob down. Nothing else about the rule changes: the same append happens, and the same bulk
 * cluster fires when the count is exceeded.
 */
export function createIngest(cells: number, compression: number, threshold: number): IngestState {
  return {
    arrived: [],
    cells: Array.from({ length: cells }, () => {
      const digest = createDigest(compression);
      digest.compressThreshold = threshold;
      return digest;
    }),
    settled: Array.from({ length: cells }, () => 0),
    lastCell: null,
    lastValue: null,
    compressedCells: [],
    collapsedFrom: null,
  };
}

/**
 * Take one arrival into both structures.
 *
 * Both are appends. The exact path pushes onto an unsorted vector; the digest pushes a weight-1
 * centroid onto the end of its own. Neither sorts, neither compares. Only when a digest exceeds
 * its threshold does anything expensive happen, and then it happens to the whole buffer at once:
 * sort by mean, then greedily cluster. That rhythm — nothing, nothing, nothing, collapse — is the
 * real cost profile, and an earlier version of this page hid it by compressing on every value.
 */
export function ingestOne(state: IngestState, arrival: Arrival): IngestState {
  // `arrived` is pushed in place and handed back by reference. Copying it per value would make
  // ingestion quadratic, which matters once the run is unbounded — and it is the array whose
  // unbounded growth the page exists to show. Re-render is driven by the new state object, not by
  // this array's identity.
  state.arrived.push(arrival.value);
  const cells = state.cells.map((c) => ({ ...c, centroids: c.centroids.map((x) => ({ ...x })) }));
  const settled = [...state.settled];
  const target = cells[arrival.cell]!;

  target.centroids.push({ mean: arrival.value, weight: 1 });
  target.totalWeight += 1;
  if (arrival.value < target.min) target.min = arrival.value;
  if (arrival.value > target.max) target.max = arrival.value;

  let compressed = false;
  let collapsedFrom: number | null = null;
  if (target.centroids.length > target.compressThreshold) {
    collapsedFrom = target.centroids.length;
    target.centroids = clustered(target);
    settled[arrival.cell] = target.centroids.length;
    compressed = true;
  }

  return {
    arrived: state.arrived,
    cells,
    settled,
    lastCell: arrival.cell,
    lastValue: arrival.value,
    compressedCells: compressed ? [arrival.cell] : [],
    collapsedFrom,
  };
}

/**
 * The sorted values, produced the way the runtime produces them: once, on demand.
 *
 * `kernel.rs:117` sorts the retained vector at summarize time. Calling this is the moment the
 * exact path does its only expensive piece of work.
 */
export function sortedAtSummarize(state: IngestState): number[] {
  return [...state.arrived].sort((a, b) => a - b);
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

/**
 * The latency value at an arrival index, computed rather than stored.
 *
 * An unbounded run cannot pre-generate its values, and it must stay reproducible: reset and replay
 * has to give the same stream. A hash of the index does both — a full avalanche so consecutive
 * indices are uncorrelated, then Box-Muller into the same lognormal shape `latencySamples` draws.
 */
export function latencyAt(index: number): number {
  const hash = (n: number): number => {
    let h = (n + 1) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 2246822507) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
    return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
  };
  const u1 = Math.max(Number.EPSILON, hash(index * 2));
  const u2 = hash(index * 2 + 1);
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return Math.exp(Math.log(100) + 0.45 * z);
}

/** The arrival at an index, for a run with no end. */
export function arrivalAt(index: number, cells: number): Arrival {
  return { index, value: latencyAt(index), cell: index % cells };
}
