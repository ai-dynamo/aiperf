/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — one dataset, carried through every stage of the metrics plane.
//!
//! The deck's whole claim is that these are stages of a single pipeline rather than separate
//! tricks, so the records built here are the only records any panel ever draws.

import {
  buildColumnStore,
  coarseTokenEvents,
  concurrencyEvents,
  detectSteadyWindow,
  iclTokenEvents,
  sortSweepEvents,
  sweepLineCumsum,
  type ColumnStore,
  type CumsumStep,
  type Record,
  type StepFn,
  type SweepEvent,
} from "./sweepAlgo.js";
import {
  batchMeansTrend,
  consensusWindow,
  cusumWindow,
  mser5,
  type Consensus,
  type CusumTrace,
  type Mser5Trace,
  type Stationarity,
} from "./detectors.js";

/** Concurrency the load generator is aiming for. */
export const TARGET_CONCURRENCY = 12;
/** Records in the run. Small enough that every event stays individually visible. */
const RECORD_COUNT = 96;
/** Nanoseconds, scaled so the whole run is a readable number of "seconds". */
const SEC = 1_000_000_000;

function rand(seed: number, a: number, b: number): number {
  const x = Math.sin(a * 127.1 + b * 311.7 + seed * 51.17) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * A concurrency-target run: a ramp while the generator fills its slots, a saturated middle, and a
 * drain as the last requests finish after admission stops.
 *
 * Deliberately shaped this way because it is the shape steady-state detection exists for — a
 * summary over the whole thing blends both transients into the steady interval.
 */
export function buildRecords(seed = 1): Record[] {
  const records: Record[] = [];
  // Closed loop, as a concurrency-target generator actually works: a slot is refilled the moment
  // it frees. An open loop at a fixed arrival rate settles at rate x service time instead, which
  // is a different run shape and would never hold the target long enough to have a steady window.
  const freeAt: number[] = [];
  let issued = 0;

  const shape = (index: number, startNs: number): Record => {
    const ttft = (0.45 + rand(seed, index, 2) * 0.45) * SEC;
    const chunks = 3 + Math.floor(rand(seed, index, 3) * 4);
    const iclNs: number[] = [];
    for (let c = 0; c < chunks; c++) iclNs.push((0.2 + rand(seed, index * 13 + c, 4) * 0.28) * SEC);
    const decode = iclNs.reduce((a, b) => a + b, 0);
    return {
      index,
      startNs,
      generationStartNs: startNs + ttft,
      endNs: startNs + ttft + decode,
      inputTokens: Math.round(120 + rand(seed, index, 5) * 260),
      outputTokens: Math.round(40 + rand(seed, index, 6) * 90),
      iclNs,
    };
  };

  // Ramp: the generator opens its slots over a short window rather than all at one instant.
  for (let slot = 0; slot < TARGET_CONCURRENCY && issued < RECORD_COUNT; slot++) {
    const startNs = slot * 0.09 * SEC;
    const record = shape(issued++, startNs);
    records.push(record);
    freeAt.push(record.endNs);
  }

  // Plateau: refill whichever slot frees first, so exactly TARGET stay in flight.
  while (issued < RECORD_COUNT) {
    let slot = 0;
    for (let i = 1; i < freeAt.length; i++) if (freeAt[i]! < freeAt[slot]!) slot = i;
    // A little client-side think time, so arrivals are not exactly coincident with completions.
    const startNs = freeAt[slot]! + rand(seed, issued, 7) * 0.05 * SEC;
    const record = shape(issued++, startNs);
    records.push(record);
    freeAt[slot] = record.endNs;
  }

  records.sort((a, b) => a.startNs - b.startNs);
  return records.map((r, i) => ({ ...r, index: i }));
}

/**
 * A deliberate collision: two records where one ends at exactly the instant another starts.
 *
 * Real runs produce these constantly at nanosecond resolution; the scenario forces one so the
 * tie-break has something to demonstrate on.
 */
export function withForcedCollision(records: Record[]): Record[] {
  const out = records.map((r) => ({ ...r }));
  const donor = out[6];
  const receiver = out[7];
  if (donor !== undefined && receiver !== undefined) {
    receiver.startNs = donor.endNs;
    receiver.generationStartNs = receiver.startNs + (receiver.generationStartNs - donor.endNs > 0
      ? receiver.generationStartNs - receiver.startNs
      : 0.6 * SEC);
    receiver.endNs = Math.max(receiver.endNs, receiver.generationStartNs + 0.9 * SEC);
  }
  return out;
}

export type Scenario = {
  records: Record[];
  store: ColumnStore;
  /** Unsorted concurrency events, in record order. */
  rawEvents: SweepEvent[];
  /** After the `(timestamp, delta)` sort. */
  sortedEvents: SweepEvent[];
  /** Indices into `sortedEvents` that share a timestamp with their predecessor. */
  collisions: number[];
  steps: CumsumStep[];
  concurrency: StepFn;
  snapped: number;
  coarseTokens: StepFn;
  iclTokens: StepFn;
  /** Threshold-crossing window, the Rust detector. */
  thresholdWindow: ReturnType<typeof detectSteadyWindow>;
  cusum: CusumTrace;
  mser5Latency: Mser5Trace;
  mser5Ttft: Mser5Trace;
  consensus: Consensus;
  stationarity: Stationarity;
  runStartNs: number;
  runEndNs: number;
};

/** Run the whole pipeline once. Pure and deterministic, so every panel agrees. */
export function buildScenario(seed = 1): Scenario {
  const records = withForcedCollision(buildRecords(seed));
  const store = buildColumnStore(records);

  const rawEvents = concurrencyEvents(records);
  const sortedEvents = sortSweepEvents(rawEvents);
  const collisions: number[] = [];
  for (let i = 1; i < sortedEvents.length; i++) {
    if (sortedEvents[i]!.timestampNs === sortedEvents[i - 1]!.timestampNs) collisions.push(i);
  }

  const { steps, curve: concurrency, snapped } = sweepLineCumsum(rawEvents);
  const coarseTokens = sweepLineCumsum(coarseTokenEvents(records)).curve;
  const iclTokens = sweepLineCumsum(iclTokenEvents(records, store)).curve;

  const runStartNs = concurrency.timestampsNs[0] ?? 0;
  const runEndNs = concurrency.timestampsNs[concurrency.timestampsNs.length - 1] ?? 0;
  const full = { startNs: runStartNs, endNs: runEndNs };

  const thresholdWindow = detectSteadyWindow(concurrency, TARGET_CONCURRENCY);
  const cusum = cusumWindow(concurrency);

  // MSER-5 runs on per-record series in start order, as `mser5_boundary_ns` does.
  const byStart = [...records].sort((a, b) => a.startNs - b.startNs);
  const latency = byStart.map((r) => (r.endNs - r.startNs) / SEC);
  const ttft = byStart.map((r) => (r.generationStartNs - r.startNs) / SEC);
  const mser5Latency = mser5(latency);
  const mser5Ttft = mser5(ttft);

  const boundaryFrom = (trace: Mser5Trace) =>
    trace.truncation > 0 && trace.truncation < byStart.length
      ? { startNs: byStart[trace.truncation]!.startNs, endNs: runEndNs }
      : null;

  const consensus = consensusWindow(
    [
      { name: "cusum", window: cusum.window },
      { name: "mser5_latency", window: boundaryFrom(mser5Latency) },
      { name: "mser5_ttft", window: boundaryFrom(mser5Ttft) },
    ],
    full,
  );

  const inWindow = byStart
    .filter((r) => r.startNs >= consensus.window.startNs && r.endNs <= consensus.window.endNs)
    .map((r) => (r.endNs - r.startNs) / SEC);
  const stationarity = batchMeansTrend(inWindow);

  return {
    records,
    store,
    rawEvents,
    sortedEvents,
    collisions,
    steps,
    concurrency,
    snapped,
    coarseTokens,
    iclTokens,
    thresholdWindow,
    cusum,
    mser5Latency,
    mser5Ttft,
    consensus,
    stationarity,
    runStartNs,
    runEndNs,
  };
}

/** Seconds since run start, for axis labels. */
export function toSeconds(ns: number, runStartNs: number): number {
  return (ns - runStartNs) / SEC;
}
