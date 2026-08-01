/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the advanced steady-state detectors, ported from the `ajc/new-config-kube` worktree.
//!
//! The Rust tree on this branch detects steady state one way: the concurrency curve crossing a
//! fraction of the configured target (`metrics_core/steady_state.rs`). That needs a target to
//! exist. The Python suite in `new-config-kube` is purely data-driven and does not, which is why
//! it carries four independent estimators and then reconciles them.
//!
//! Ported from:
//! - `src/aiperf/analysis/ramp_detection.py:24`  — `cusum_steady_state_window`
//! - `src/aiperf/analysis/ramp_detection.py:106` — `mser5_truncation_point`
//! - `src/aiperf/analysis/ramp_detection.py:248` — `detect_steady_state_window` (consensus)
//! - `src/aiperf/analysis/stationarity.py:178`   — `batch_means_trend_test`

import type { StepFn } from "./sweepAlgo.js";

export type Boundary = { startNs: number; endNs: number };

/** Default floor below which a detected window is rejected, as a percent of total duration. */
export const MIN_WINDOW_PCT = 10.0;

/**
 * Time-weighted p95 of a step function's held values.
 *
 * Weighting by how *long* each level was held rather than counting events is the point: a curve
 * that spikes for a microsecond and sits at 40 for a minute has a p95 of 40, not of the spike.
 */
export function timeWeightedP95(curve: StepFn): number {
  const n = curve.timestampsNs.length;
  if (n === 0) return 0;
  const durations = curve.timestampsNs.map((t, i) =>
    i + 1 < n ? curve.timestampsNs[i + 1]! - t : 0,
  );
  const total = durations.reduce((a, b) => a + b, 0);
  if (total <= 0) return curve.values[n - 1] ?? 0;

  const order = curve.values.map((v, i) => ({ v, d: durations[i]! })).sort((a, b) => a.v - b.v);
  let cumulative = 0;
  for (const { v, d } of order) {
    cumulative += d;
    if (cumulative / total >= 0.95) return v;
  }
  return order[order.length - 1]?.v ?? 0;
}

export type CusumTrace = {
  target: number;
  /** Time-weighted deviation from target at each event. */
  deviations: number[];
  forward: number[];
  /** Backward cumulative sum, already re-reversed onto the forward index. */
  backward: number[];
  rampUpIndex: number;
  rampDownIndex: number;
  window: Boundary | null;
  method: string;
};

/**
 * Retrospective time-weighted CUSUM.
 *
 * Ported from `cusum_steady_state_window`. Deviations are `(value - target) * duration`. The
 * forward cumulative sum is most negative exactly where the run stops being *below* target — that
 * argmin is the end of ramp-up. Running the same sum backwards finds the start of ramp-down.
 *
 * It is retrospective, not online: nothing here is a control chart with a decision threshold, it
 * is a single pass over a finished curve looking for two turning points.
 */
export function cusumWindow(curve: StepFn, minWindowPct = 0): CusumTrace {
  const n = curve.timestampsNs.length;
  const empty: CusumTrace = {
    target: 0,
    deviations: [],
    forward: [],
    backward: [],
    rampUpIndex: 0,
    rampDownIndex: 0,
    window: null,
    method: "empty",
  };
  if (n === 0) return empty;

  const durations = curve.timestampsNs.map((t, i) =>
    i + 1 < n ? curve.timestampsNs[i + 1]! - t : 0,
  );
  const target = timeWeightedP95(curve);
  const deviations = curve.values.map((v, i) => (v - target) * durations[i]!);

  const forward: number[] = [];
  let running = 0;
  for (const d of deviations) {
    running += d;
    forward.push(running);
  }

  const reversed: number[] = [];
  running = 0;
  for (let i = n - 1; i >= 0; i--) {
    running += deviations[i]!;
    reversed.push(running);
  }
  // Re-reversed so index i lines up with the forward series.
  const backward = reversed.slice().reverse();

  const argmin = (xs: readonly number[]) =>
    xs.reduce((best, v, i) => (v < xs[best]! ? i : best), 0);

  const rampUpIndex = argmin(forward);
  const rampDownOffset = argmin(reversed);
  const rampDownIndex = n - 1 - rampDownOffset;

  const first = curve.timestampsNs[0]!;
  const last = curve.timestampsNs[n - 1]!;
  const full: Boundary = { startNs: first, endNs: last };

  if (rampUpIndex >= rampDownIndex) {
    return { target, deviations, forward, backward, rampUpIndex, rampDownIndex, window: full, method: "cusum_inverted" };
  }
  const window: Boundary = {
    startNs: curve.timestampsNs[rampUpIndex]!,
    endNs: curve.timestampsNs[rampDownIndex]!,
  };
  const totalDuration = last - first;
  if (totalDuration > 0 && (window.endNs - window.startNs) / totalDuration * 100 < minWindowPct) {
    return { target, deviations, forward, backward, rampUpIndex, rampDownIndex, window: full, method: "fallback_min_window" };
  }
  return { target, deviations, forward, backward, rampUpIndex, rampDownIndex, window, method: "cusum" };
}

export type Mser5Trace = {
  /** Means of each non-overlapping batch of five. */
  batches: number[];
  /** MSER statistic per candidate truncation point, `variance / count`. */
  mser: number[];
  /** Chosen truncation, in batches. */
  dStar: number;
  /** Chosen truncation, in samples. */
  truncation: number;
  /** Never delete more than half. */
  maxD: number;
};

/**
 * MSER-5: Marginal Standard Error Rule with batch size five.
 *
 * Ported from `mser5_truncation_point`. Batch the series into non-overlapping groups of five and
 * take their means. For every candidate truncation `d`, compute the variance of the *retained*
 * batches divided by how many remain — that ratio is the squared standard error of the mean you
 * would report after deleting `d`. Pick the `d` that minimizes it.
 *
 * The guard that matters: `d` is searched only over the first half, so the rule can never
 * "converge" by deleting almost everything and reporting the standard error of two points.
 */
export function mser5(series: readonly number[], batchSize = 5): Mser5Trace {
  const none: Mser5Trace = { batches: [], mser: [], dStar: 0, truncation: 0, maxD: 0 };
  if (series.length < 10) return none;

  const batches: number[] = [];
  for (let i = 0; i + batchSize <= series.length; i += batchSize) {
    let sum = 0;
    for (let k = 0; k < batchSize; k++) sum += series[i + k]!;
    batches.push(sum / batchSize);
  }
  const m = batches.length;
  if (m < 4) return none;

  const maxD = Math.floor(m / 2);
  const mser: number[] = [];
  for (let d = 0; d <= maxD; d++) {
    const retained = batches.slice(d);
    const count = retained.length;
    const mean = retained.reduce((a, b) => a + b, 0) / count;
    const variance = Math.max(
      retained.reduce((a, b) => a + (b - mean) * (b - mean), 0) / count,
      0,
    );
    mser.push(variance / count);
  }
  const dStar = mser.reduce((best, v, i) => (v < mser[best]! ? i : best), 0);
  return { batches, mser, dStar, truncation: dStar * batchSize, maxD };
}

export type ConsensusSignal = { name: string; window: Boundary | null };

export type Consensus = {
  signals: ConsensusSignal[];
  window: Boundary;
  method: string;
};

/**
 * Reconcile independent boundary estimates.
 *
 * Ported from `detect_steady_state_window`: take the **latest** start any signal proposes and the
 * **earliest** end. Deliberately the most conservative reading — the window is only steady if
 * every detector agrees it is, so one late-settling signal shortens it rather than being outvoted.
 */
export function consensusWindow(
  signals: readonly ConsensusSignal[],
  full: Boundary,
  minWindowPct = MIN_WINDOW_PCT,
): Consensus {
  const usable = signals.filter((s) => s.window !== null);
  if (usable.length === 0) return { signals: [...signals], window: full, method: "empty" };

  const startNs = Math.max(...usable.map((s) => s.window!.startNs));
  const endNs = Math.min(...usable.map((s) => s.window!.endNs));
  const method = usable.map((s) => s.name).join("_");

  if (startNs >= endNs) {
    return { signals: [...signals], window: full, method: "fallback_no_overlap" };
  }
  const total = full.endNs - full.startNs;
  if (total > 0 && ((endNs - startNs) / total) * 100 < minWindowPct) {
    return { signals: [...signals], window: full, method: "fallback_min_window" };
  }
  return { signals: [...signals], window: { startNs, endNs }, method };
}

export type Stationarity = { rho: number; batches: number[]; warning: boolean };

/**
 * Batch-means trend test.
 *
 * Ported from `batch_means_trend_test`: split the window's values into ten equal batches, take
 * their means, and Spearman-rank-correlate them against batch index. A strong correlation means
 * the "steady" window is still trending, so the detection was wrong. The analyzer flags it at
 * `|rho| > 0.65`.
 *
 * The p-value in the Python version comes from a regularized incomplete beta; the correlation
 * alone is what the visual needs, so only rho is computed here — stated rather than implied.
 */
export function batchMeansTrend(values: readonly number[], k = 10): Stationarity {
  if (values.length < k) return { rho: 0, batches: [], warning: false };
  const size = Math.floor(values.length / k);
  const batches: number[] = [];
  for (let i = 0; i < k; i++) {
    const slice = values.slice(i * size, (i + 1) * size);
    batches.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }

  const rank = (xs: readonly number[]): number[] => {
    const order = xs.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const out = new Array<number>(xs.length);
    let i = 0;
    while (i < order.length) {
      let j = i;
      while (j + 1 < order.length && order[j + 1]!.v === order[i]!.v) j++;
      const averaged = (i + j) / 2 + 1;
      for (let k2 = i; k2 <= j; k2++) out[order[k2]!.i] = averaged;
      i = j + 1;
    }
    return out;
  };

  const a = rank(batches);
  const b = rank(batches.map((_, i) => i));
  const mean = (xs: number[]) => xs.reduce((p, q) => p + q, 0) / xs.length;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i]! - ma) * (b[i]! - mb);
    da += (a[i]! - ma) ** 2;
    db += (b[i]! - mb) ** 2;
  }
  const rho = da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
  return { rho, batches, warning: Math.abs(rho) > 0.65 };
}
