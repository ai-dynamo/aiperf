/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — what a mergeable sketch keeps exact, and what it only estimates.
//!
//! A faithful port of `rust/runtime/src/cellular/sketch.rs`, so the numbers on the page are the
//! numbers the runtime produces rather than an illustration of them:
//!
//! - The merging t-digest (Dunning). A value enters as a weight-1 centroid; `clustered` sorts by
//!   mean and greedily merges adjacent centroids while each cluster spans at most one unit of the
//!   K1 scale `k(q) = δ·asin(2q−1)/2π`.
//! - `merge` is concatenate-then-compress, which is why it is associative and order-independent
//!   up to floating point — the property that lets cells fold without a central pass.
//! - `min` and `max` are tracked separately and survive every merge exactly. Quantile 0 and 1
//!   return them verbatim; interior quantiles interpolate centroid means.
//!
//! The shape of the K1 scale is the whole reason to draw this. `asin` steepens towards ±1, so one
//! scale unit spans a *narrow* band of quantiles at the tails and a *wide* one through the body:
//! the digest spends its centroids where the tail is, and is coarsest in the middle. That is the
//! opposite of the intuition that an approximation degrades at the extremes.

/** Default compression δ, from `DEFAULT_COMPRESSION`. */
export const DEFAULT_COMPRESSION = 100;

/** A cluster of ingested values summarized by their mean and total weight. */
export type Centroid = { mean: number; weight: number };

export type TDigest = {
  centroids: Centroid[];
  totalWeight: number;
  /** Exact minimum ingested value; `+Infinity` when empty. */
  min: number;
  /** Exact maximum ingested value; `-Infinity` when empty. */
  max: number;
  compression: number;
  compressThreshold: number;
};

export function createDigest(compression: number = DEFAULT_COMPRESSION): TDigest {
  const c = Math.max(1, compression);
  return {
    centroids: [],
    totalWeight: 0,
    min: Number.POSITIVE_INFINITY,
    max: Number.NEGATIVE_INFINITY,
    compression: c,
    // Bound transient centroids to a small multiple of the compressed size.
    compressThreshold: Math.max(64, Math.trunc(c) * 10),
  };
}

/**
 * The K1 scale function.
 *
 * `asin` is near-vertical approaching ±1, so a fixed one-unit budget buys a narrow quantile span
 * at the tails and a wide one in the body — fine resolution exactly where the percentiles people
 * care about live.
 */
export function kScale(q: number, compression: number): number {
  const clamped = Math.min(1, Math.max(-1, 2 * q - 1));
  return (compression * Math.asin(clamped)) / (2 * Math.PI);
}

/** Ingest one finite value. Non-finite values are ignored — a sketch never stores sentinels. */
export function add(digest: TDigest, value: number): void {
  if (!Number.isFinite(value)) return;
  digest.centroids.push({ mean: value, weight: 1 });
  digest.totalWeight += 1;
  if (value < digest.min) digest.min = value;
  if (value > digest.max) digest.max = value;
  if (digest.centroids.length > digest.compressThreshold) compress(digest);
}

export function extendFrom(digest: TDigest, values: readonly number[]): void {
  for (const v of values) add(digest, v);
}

export function isEmpty(digest: TDigest): boolean {
  return digest.totalWeight === 0;
}

/** Ingested value count. Exact — it is a sum of weights, not an estimate. */
export function count(digest: TDigest): number {
  return digest.totalWeight;
}

/**
 * Sort by mean, then greedily cluster adjacent centroids while each cluster spans at most one
 * K1-scale unit.
 *
 * Sorting first is what makes the result independent of arrival order, and therefore what makes
 * merging cells safe in any order.
 */
export function clustered(digest: TDigest): Centroid[] {
  if (digest.centroids.length <= 1) return digest.centroids.map((c) => ({ ...c }));
  const sorted = digest.centroids.map((c) => ({ ...c })).sort((a, b) => a.mean - b.mean);

  const total = digest.totalWeight;
  const out: Centroid[] = [];
  let cumulativeBefore = 0;
  let current = { ...sorted[0]! };
  for (const centroid of sorted.slice(1)) {
    const qStart = cumulativeBefore / total;
    const proposedWeight = current.weight + centroid.weight;
    const qEnd = (cumulativeBefore + proposedWeight) / total;
    if (kScale(qEnd, digest.compression) - kScale(qStart, digest.compression) <= 1) {
      const combined = current.weight + centroid.weight;
      current.mean = (current.mean * current.weight + centroid.mean * centroid.weight) / combined;
      current.weight = combined;
    } else {
      out.push(current);
      cumulativeBefore += current.weight;
      current = { ...centroid };
    }
  }
  out.push(current);
  return out;
}

export function compress(digest: TDigest): void {
  digest.centroids = clustered(digest);
}

/**
 * Concatenate centroids, then compress.
 *
 * That is the entire merge. No central pass over the values, no coordination — which is what lets
 * N cells each summarize their own slice and have the union be meaningful.
 */
export function merge(into: TDigest, other: TDigest): void {
  if (isEmpty(other)) return;
  into.centroids.push(...other.centroids.map((c) => ({ ...c })));
  into.totalWeight += other.totalWeight;
  if (other.min < into.min) into.min = other.min;
  if (other.max > into.max) into.max = other.max;
  compress(into);
}

function interpolate(q0: number, v0: number, q1: number, v1: number, q: number): number {
  if (q1 <= q0) return v0;
  return v0 + ((v1 - v0) * (q - q0)) / (q1 - q0);
}

/**
 * Estimate the value at quantile `q`.
 *
 * `q = 0` and `q = 1` return the exact min and max — those anchors are never estimated. Everything
 * between interpolates centroid means by cumulative quantile.
 */
export function quantile(digest: TDigest, q: number): number | null {
  if (isEmpty(digest)) return null;
  return quantileFrom(digest, clustered(digest), q);
}

export function quantileFrom(digest: TDigest, centroids: readonly Centroid[], q: number): number {
  const clamped = Math.min(1, Math.max(0, q));
  if (clamped <= 0) return digest.min;
  if (clamped >= 1) return digest.max;
  const total = digest.totalWeight;
  let cumulative = 0;
  let prevQ = 0;
  let prevValue = digest.min;
  for (const centroid of centroids) {
    const centerQ = (cumulative + centroid.weight / 2) / total;
    if (clamped < centerQ) {
      return interpolate(prevQ, prevValue, centerQ, centroid.mean, clamped);
    }
    prevQ = centerQ;
    prevValue = centroid.mean;
    cumulative += centroid.weight;
  }
  return interpolate(prevQ, prevValue, 1, digest.max, clamped);
}

/**
 * The report's exact percentile: type-7 linear interpolation over the retained values.
 *
 * This is what the sketch is compared against, and what exact mode actually computes — ported from
 * the reference implementation in `sketch.rs`'s own tests.
 */
export function exactPercentile(sorted: readonly number[], percentile: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const virtualIdx = (percentile / 100) * (n - 1);
  const lo = Math.floor(virtualIdx);
  const hi = Math.min(lo + 1, n - 1);
  const frac = virtualIdx - lo;
  return sorted[lo]! + frac * (sorted[hi]! - sorted[lo]!);
}

/** Deterministic pseudo-random values from a small LCG — reproducible, no clock, no rand. */
export function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

/** Which shape to draw the sample from. */
export type Shape = "lognormal" | "bimodal";

/**
 * A latency-shaped sample.
 *
 * `lognormal` is the realistic default — smooth, right-skewed, the shape request latency actually
 * takes. `bimodal` is the digest's worst case and is offered deliberately: a hard split between a
 * fast mode and a slow one puts a near-vertical cliff in the CDF, and since quantiles interpolate
 * linearly between centroid means, a step is the one thing this representation cannot express.
 * Measured on the page — error at the cliff is an order of magnitude worse than anywhere else.
 */
export function latencySamples(n: number, seed: number, shape: Shape = "lognormal"): number[] {
  const rand = lcg(seed);
  if (shape === "bimodal") {
    return Array.from({ length: n }, () => {
      const u = rand();
      const body = 80 + 40 * rand();
      return u > 0.9 ? body + 300 * Math.pow(rand(), 0.35) : body;
    });
  }
  // Box-Muller on the same LCG, so the sample stays reproducible without a rand dependency.
  return Array.from({ length: n }, () => {
    const u1 = Math.max(Number.EPSILON, rand());
    const u2 = rand();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.exp(Math.log(100) + 0.45 * z);
  });
}

export type CellSlice = { id: number; values: number[]; digest: TDigest };

/** Split a sample across cells the way a cellular run does: each cell owns a contiguous slice. */
export function splitAcrossCells(values: readonly number[], cells: number): CellSlice[] {
  const out: CellSlice[] = [];
  const per = Math.ceil(values.length / cells);
  for (let id = 0; id < cells; id++) {
    const slice = values.slice(id * per, (id + 1) * per);
    const digest = createDigest();
    extendFrom(digest, slice);
    compress(digest);
    out.push({ id, values: slice, digest });
  }
  return out;
}

/** Fold every cell's digest into one, exactly as the controller does. */
export function foldCells(cellSlices: readonly CellSlice[]): TDigest {
  const folded = createDigest();
  for (const cell of cellSlices) merge(folded, cell.digest);
  return folded;
}

export type Comparison = {
  label: string;
  exact: number;
  sketch: number;
  /** Relative error as a percentage of the exact value. */
  errorPct: number;
  /** Whether this figure is exact by construction rather than estimated. */
  guaranteed: boolean;
};

/** The percentile band a report emits. */
export const PERCENTILES = [50, 90, 95, 99] as const;

/**
 * Everything the two paths report, side by side.
 *
 * Counts, sums, extrema and averages are marked guaranteed: they are not approximations that
 * happen to agree, they are computed from running totals the sketch keeps exactly.
 */
export function compare(values: readonly number[], folded: TDigest): Comparison[] {
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  const rows: Comparison[] = [
    { label: "count", exact: values.length, sketch: count(folded), errorPct: 0, guaranteed: true },
    { label: "min", exact: sorted[0]!, sketch: folded.min, errorPct: 0, guaranteed: true },
    { label: "max", exact: sorted.at(-1)!, sketch: folded.max, errorPct: 0, guaranteed: true },
    {
      label: "mean",
      exact: sum / values.length,
      sketch: sum / count(folded),
      errorPct: 0,
      guaranteed: true,
    },
  ];
  for (const p of PERCENTILES) {
    const exact = exactPercentile(sorted, p);
    const sketch = quantile(folded, p / 100) ?? Number.NaN;
    rows.push({
      label: `p${p}`,
      exact,
      sketch,
      errorPct: exact === 0 ? 0 : ((sketch - exact) / exact) * 100,
      guaranteed: false,
    });
  }
  return rows;
}

/** Quantile span covered by each centroid, for drawing where the digest spends its resolution. */
export function centroidSpans(digest: TDigest): { q0: number; q1: number; centroid: Centroid }[] {
  const centroids = clustered(digest);
  const total = digest.totalWeight;
  const out: { q0: number; q1: number; centroid: Centroid }[] = [];
  let cumulative = 0;
  for (const centroid of centroids) {
    const q0 = cumulative / total;
    cumulative += centroid.weight;
    out.push({ q0, q1: cumulative / total, centroid });
  }
  return out;
}
