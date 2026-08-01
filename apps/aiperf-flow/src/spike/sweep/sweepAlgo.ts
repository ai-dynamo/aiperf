/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the metrics plane's sweep-line algorithms, ported faithfully enough to watch run.
//!
//! Every rule here is taken from `rust/runtime/src/metrics_core/`:
//!
//! - Columnar storage: `store.rs` keeps NaN-sparse numeric columns aligned by *absolute request
//!   index*, so an absent value is a NaN sentinel rather than a shorter column.
//! - Ragged/CSR: list metrics (inter-chunk latency) use a flat values array plus per-record
//!   offsets and lengths — `RaggedSeries` / `IclSeries`.
//! - Cumsum: `sweepline::sweep_line_cumsum` sorts events, sums deltas, then snaps residuals
//!   below `1e-9 * max_abs` to zero.
//! - Collision avoidance: events sort by `(timestamp asc, delta asc)`, which puts end deltas
//!   *before* start deltas at an equal timestamp so touching intervals never double-count.
//! - ICL awareness: `sweepline::kv_cache` places each chunk's tokens at
//!   `generation_start + cumsum(icl[0..=i])` instead of dumping every output token at
//!   generation start.
//! - Steady state: `steady_state::detect_steady_window` opens at the first event reaching
//!   `ceil(fraction * target)` and closes at the *last* descent back below it.

/** Absent value sentinel. The real columns store NaN in an f64 column, not a shorter column. */
export const ABSENT = Number.NaN;

/** One inference record, as the accumulator ingests it. Times are ns. */
export type Record = {
  index: number;
  startNs: number;
  generationStartNs: number;
  endNs: number;
  inputTokens: number;
  outputTokens: number;
  /** Inter-chunk latencies, one per generated chunk after the first. Ragged by nature. */
  iclNs: number[];
};

/** A NaN-sparse numeric column, addressed by absolute request index. */
export type NumericColumn = { name: string; values: number[] };

/** CSR-style ragged series: flat values plus per-record slices. */
export type RaggedSeries = {
  valuesNs: number[];
  offsets: number[];
  lengths: number[];
  /** Records in flat-value append order, as `IclSeries::append_order`. */
  appendOrder: number[];
};

export type ColumnStore = {
  rows: number;
  columns: NumericColumn[];
  icl: RaggedSeries;
};

/** One timestamped change applied by the cumulative sum. */
export type SweepEvent = {
  timestampNs: number;
  delta: number;
  /** Which record produced it, for tracing an event back to its bar. */
  record: number;
  kind: "start" | "end" | "chunk";
};

/** A right-continuous step function stored at its event boundaries. */
export type StepFn = { timestampsNs: number[]; values: number[] };

// ---------------------------------------------------------------------------
// Columnar storage
// ---------------------------------------------------------------------------

/**
 * Ingest records into columns addressed by absolute request index.
 *
 * A record missing a value writes the NaN sentinel rather than shifting later rows, which is what
 * keeps every column index-aligned with every other and with the ragged series.
 */
export function buildColumnStore(records: readonly Record[]): ColumnStore {
  const rows = records.length;
  const col = (name: string, pick: (r: Record) => number): NumericColumn => ({
    name,
    values: records.map(pick),
  });

  const valuesNs: number[] = [];
  const offsets: number[] = [];
  const lengths: number[] = [];
  const appendOrder: number[] = [];
  for (const record of records) {
    // Offsets are absolute into the flat array; an empty list still records its offset so the
    // row stays index-aligned with the numeric columns.
    offsets.push(valuesNs.length);
    lengths.push(record.iclNs.length);
    if (record.iclNs.length > 0) appendOrder.push(record.index);
    for (const value of record.iclNs) valuesNs.push(value);
  }

  return {
    rows,
    columns: [
      col("start_ns", (r) => r.startNs),
      col("generation_start_ns", (r) => r.generationStartNs),
      col("end_ns", (r) => r.endNs),
      col("input_tokens", (r) => r.inputTokens),
      col("output_tokens", (r) => r.outputTokens),
    ],
    icl: { valuesNs, offsets, lengths, appendOrder },
  };
}

// ---------------------------------------------------------------------------
// Event construction
// ---------------------------------------------------------------------------

/** Concurrency events: `+1` at each start, `-1` at each end. */
export function concurrencyEvents(records: readonly Record[]): SweepEvent[] {
  const events: SweepEvent[] = [];
  for (const r of records) {
    if (Number.isNaN(r.startNs) || Number.isNaN(r.endNs)) continue;
    events.push({ timestampNs: r.startNs, delta: 1, record: r.index, kind: "start" });
    events.push({ timestampNs: r.endNs, delta: -1, record: r.index, kind: "end" });
  }
  return events;
}

/**
 * Coarse tokens-in-flight: input tokens live for the whole request, and every output token
 * arrives together at generation start.
 */
export function coarseTokenEvents(records: readonly Record[]): SweepEvent[] {
  const events: SweepEvent[] = [];
  for (const r of records) {
    events.push({ timestampNs: r.startNs, delta: r.inputTokens, record: r.index, kind: "start" });
    events.push({
      timestampNs: r.generationStartNs,
      delta: r.outputTokens,
      record: r.index,
      kind: "chunk",
    });
    events.push({
      timestampNs: r.endNs,
      delta: -(r.inputTokens + r.outputTokens),
      record: r.index,
      kind: "end",
    });
  }
  return events;
}

/**
 * ICL-aware tokens-in-flight.
 *
 * Instead of one lump at generation start, each chunk's tokens enter at
 * `generation_start + cumsum(icl[0..=i])`. A chunk landing at or after the record's end is
 * clamped back to the end, matching the guard in `kv_cache.rs`.
 */
export function iclTokenEvents(records: readonly Record[], store: ColumnStore): SweepEvent[] {
  const events: SweepEvent[] = [];
  for (const r of records) {
    events.push({ timestampNs: r.startNs, delta: r.inputTokens, record: r.index, kind: "start" });

    const offset = store.icl.offsets[r.index]!;
    const length = store.icl.lengths[r.index]!;
    const chunks = length + 1;
    const perChunk = chunks > 0 ? r.outputTokens / chunks : 0;

    // The first chunk lands at generation start; the rest walk the cumulative ICL.
    events.push({
      timestampNs: r.generationStartNs,
      delta: perChunk,
      record: r.index,
      kind: "chunk",
    });
    let cumulative = 0;
    for (let i = 0; i < length; i++) {
      cumulative += store.icl.valuesNs[offset + i]!;
      let timestamp = r.generationStartNs + cumulative;
      if (!Number.isNaN(r.endNs) && timestamp >= r.endNs) timestamp = r.endNs;
      events.push({ timestampNs: timestamp, delta: perChunk, record: r.index, kind: "chunk" });
    }

    events.push({
      timestampNs: r.endNs,
      delta: -(r.inputTokens + r.outputTokens),
      record: r.index,
      kind: "end",
    });
  }
  return events;
}

// ---------------------------------------------------------------------------
// Collision avoidance and the cumulative sum
// ---------------------------------------------------------------------------

/**
 * Sort by `(timestamp asc, delta asc)`.
 *
 * The delta tie-break is the collision avoidance: at an equal timestamp a negative delta sorts
 * before a positive one, so a request ending exactly as another starts never shows a phantom
 * extra unit of concurrency. Sorting by timestamp alone would leave that ordering to chance.
 */
export function sortSweepEvents(events: SweepEvent[]): SweepEvent[] {
  return [...events].sort((a, b) => a.timestampNs - b.timestampNs || a.delta - b.delta);
}

/** True when two adjacent events share a timestamp, i.e. they collide. */
export function collisionsIn(sorted: readonly SweepEvent[]): number[] {
  const hits: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i]!.timestampNs === sorted[i - 1]!.timestampNs) hits.push(i);
  }
  return hits;
}

/**
 * Snap residuals below `1e-9 * max_abs` to zero.
 *
 * Summing thousands of `+w` then `-w` pairs in floating point leaves a curve that should be
 * exactly zero sitting at 1e-13. Snapping is what makes "the curve returns to zero" true rather
 * than nearly true — and it is why the steady-state comparison uses a half-unit margin.
 */
export function snapSmallResiduals(values: number[], epsilon: number): number[] {
  return values.map((v) => (Math.abs(v) < epsilon ? 0 : v));
}

/** Each step of the cumulative sum, retained so the running total can be watched. */
export type CumsumStep = {
  event: SweepEvent;
  /** Running total *after* applying this event. */
  running: number;
  /** True when this event shares its timestamp with the previous one. */
  collided: boolean;
};

/** Run the sweep-line cumulative sum, keeping every intermediate step. */
export function sweepLineCumsum(events: readonly SweepEvent[]): {
  steps: CumsumStep[];
  curve: StepFn;
  snapped: number;
} {
  const sorted = sortSweepEvents([...events]);
  const steps: CumsumStep[] = [];
  let running = 0;
  for (let i = 0; i < sorted.length; i++) {
    const event = sorted[i]!;
    running += event.delta;
    steps.push({
      event,
      running,
      collided: i > 0 && sorted[i - 1]!.timestampNs === event.timestampNs,
    });
  }

  const timestampsNs = steps.map((s) => s.event.timestampNs);
  const raw = steps.map((s) => s.running);
  const maxAbs = raw.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const values = maxAbs > 0 ? snapSmallResiduals(raw, 1e-9 * maxAbs) : raw;
  const snapped = values.reduce((n, v, i) => n + (v !== raw[i] ? 1 : 0), 0);

  return { steps, curve: { timestampsNs, values }, snapped };
}

// ---------------------------------------------------------------------------
// Steady state
// ---------------------------------------------------------------------------

/** Default occupancy fraction, matching `DEFAULT_STEADY_STATE_FRACTION`. */
export const DEFAULT_STEADY_FRACTION = 0.8;

export type SteadyWindow = {
  startNs: number;
  endNs: number;
  threshold: number;
  peakConcurrency: number;
};

/**
 * Detect the steady-state window from a concurrency curve.
 *
 * Ported from `detect_steady_window`: the threshold is `max(1, ceil(fraction * target))`, compared
 * at a half-unit margin so residual snapping can never flip a boundary. The window opens at the
 * first event reaching it and closes at the *last* descent back below — keep overwriting, so a
 * mid-run dip does not end the window early.
 */
export function detectSteadyWindow(
  curve: StepFn,
  targetConcurrency: number,
  fraction = DEFAULT_STEADY_FRACTION,
): SteadyWindow | null {
  if (curve.timestampsNs.length === 0 || targetConcurrency <= 0) return null;
  const f = Number.isFinite(fraction) && fraction > 0 && fraction <= 1
    ? fraction
    : DEFAULT_STEADY_FRACTION;
  const threshold = Math.max(1, Math.ceil(f * targetConcurrency));
  const level = threshold - 0.5;

  let peak = 0;
  let start: number | null = null;
  let end: number | null = null;
  let prevSaturated = false;

  for (let i = 0; i < curve.timestampsNs.length; i++) {
    const held = curve.values[i]!;
    const timestamp = curve.timestampsNs[i]!;
    peak = Math.max(peak, held);
    const saturated = held >= level;
    if (saturated && start === null) start = timestamp;
    if (start !== null && prevSaturated && !saturated) end = timestamp;
    prevSaturated = saturated;
  }

  if (start === null) return null;
  // Still saturated when the run ended: close at the last curve event.
  const endNs = end ?? curve.timestampsNs[curve.timestampsNs.length - 1]!;
  return { startNs: start, endNs, threshold, peakConcurrency: Math.round(peak) };
}

/** Value the step function holds at `t`, i.e. the last event at or before it. */
export function valueAt(curve: StepFn, t: number): number {
  let value = 0;
  for (let i = 0; i < curve.timestampsNs.length; i++) {
    if (curve.timestampsNs[i]! > t) break;
    value = curve.values[i]!;
  }
  return value;
}
