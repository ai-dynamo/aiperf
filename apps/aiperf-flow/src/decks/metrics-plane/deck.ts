/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the metrics plane's three data-shape decisions: the sweep line that
//! turns intervals into curves, the ragged flat layout that holds list metrics, and the uniform
//! slice grid that buckets a run over time.
//!
//! Requests and `inter_chunk_latency` lists are the same fixtures the interactive
//! `/aiperf-metrics-accumulator` workbook uses, so the two views agree.

import type { DeckDefinition, SlideDefinition } from "../../deck/types.js";
import { band, card, chip, fanIn, link, note, ragged, slices, sweep } from "../shared/diagram.js";
import type { SweepRequest } from "../../nodes/sweepMath.js";

/** Six overlapping requests; `gen` is `start + TTFT`, so `gen -> end` is the decode window. */
const REQUESTS: SweepRequest[] = [
  { id: "A", start: 0, gen: 6, end: 20, tokens: 120 },
  { id: "B", start: 3, gen: 10, end: 30, tokens: 200 },
  { id: "C", start: 8, gen: 12, end: 24, tokens: 90 },
  { id: "D", start: 14, gen: 22, end: 40, tokens: 260 },
  { id: "E", start: 18, gen: 25, end: 35, tokens: 150 },
  { id: "F", start: 28, gen: 34, end: 50, tokens: 180 },
];

const SLIDES: readonly SlideDefinition[] = [
  {
    id: "sweep",
    eyebrow: "01 · SWEEP LINE",
    title: "Intervals become curves without scanning time",
    lede:
      "Each request contributes +weight at its start and −weight at its end. Sort the events, take a running cumsum, and the step function is exact — O(E log E), never point by point.",
    narration:
      "Every in-flight curve the metrics plane reports comes from one trick. A request is an interval, and an interval contributes a plus-weight event when it starts and a minus-weight event when it ends. Sort those events by time, take a running cumulative sum, and what falls out is the exact step function — not a sample of it, the real thing. The cost is sorting the events, not walking the timeline, so a million requests is a million-log-million sort rather than a scan of every nanosecond. Follow any bar's left edge down to the green tick and you will see the curve step up; follow its right edge to the orange tick and it steps back down.",
    caption:
      "Weight selects the curve: 1 for concurrency, output_tokens for tokens in flight, tokens-per-decode-second for throughput. Ends sort before starts on ties so touching intervals never double-count.",
    nodes: [
      sweep(
        "sw",
        {
          title: "Concurrency — weight 1 per interval",
          requests: REQUESTS,
          curve: "concurrency",
          tMax: 52,
          axisLabel: "time (relative ns) · green = +delta, orange = −delta",
          valueLabel: "requests",
        },
        { col: 0, row: 0 },
      ),
      note(
        "weight",
        "One machine, three curves",
        "Concurrency weighs 1. Tokens-in-flight weighs output_tokens. Decode throughput weighs tokens per decode second and sweeps gen→end instead of start→end. Nothing else changes.",
        { col: 0, row: 2.6 },
      ),
      note(
        "ties",
        "Why ends sort first",
        "On a tie, a −delta is applied before a +delta. Two touching intervals therefore never show a phantom extra unit of concurrency at the instant one hands off to the next.",
        { col: 1.1, row: 2.6 },
      ),
    ],
    edges: [],
    revealOrder: ["sw", "weight", "ties"],
  },
  {
    id: "ragged",
    eyebrow: "02 · LIST METRICS",
    title: "Ragged lists, flat arrays",
    lede:
      "inter_chunk_latency is one variable-length list per record. Packing them into values + record_indices + offsets makes a per-record question answerable with a mask.",
    narration:
      "Not every metric is a scalar. Inter-chunk latency is a list, and each request's list is a different length — one for every gap between streamed chunks. Storing that as a list of Python lists means a loop per record for every question you ask. So it is packed instead: one flat values array with every element back to back, a record-indices column saying who owns each element, and an offsets table saying where each record's run begins. Now a per-record question is a boolean mask over one column. Note record one — it streamed as a single chunk, so it has no gaps at all, and its offset is minus one rather than a real index. Absent is a different state from empty, and the layout keeps them apart.",
    caption:
      "The prefix sum resets at each record boundary via offsets, turning per-chunk gaps into absolute chunk end-times — the input to the ICL-aware throughput sweep.",
    nodes: [
      ragged(
        "rg",
        {
          title: "inter_chunk_latency — five records",
          lists: [[12, 9, 11], [], [8, 10, 9, 13, 7], [10, 12], [9]],
          highlight: 2,
          raggedLabel: "per-record lists (each a different length)",
          flatLabel: "flat arrays (one allocation, masked in bulk)",
        },
        { col: 0, row: 0 },
      ),
      note(
        "absent",
        "−1 is not zero",
        "Record 1 emitted a single chunk, so it has no inter-chunk gaps. A real offset there would be indistinguishable from record 2's start, so absence gets its own value.",
        { col: 0, row: 2.4 },
      ),
      note(
        "mask",
        "Why flatten at all",
        "One allocation instead of N, and a per-record query becomes a comparison over record_indices — no Python-level loop over requests to answer a dataset-wide question.",
        { col: 1.1, row: 2.4 },
      ),
    ],
    edges: [],
    revealOrder: ["rg", "absent", "mask"],
  },
  {
    id: "slices",
    eyebrow: "03 · TIME SLICES",
    title: "The last bucket is almost never full",
    lede:
      "A uniform grid bins each record by its start. The grid outruns real activity, so the trailing slice is clipped and flagged rather than silently diluting a rate.",
    narration:
      "Time-slice exports cut a run into uniform buckets, and two things about that are easy to get wrong. First, a record is binned by its start — the dot on each bar — not by every bucket it overlaps. A request spanning three slices still counts once, in the slice it began in. Second, the grid almost never lands on the end of activity. Here the last request ends at fifty but a fifteen-unit grid runs to sixty. If a rate for that bucket divided by fifteen it would be diluted by ten units of padding that contains nothing. So the trailing slice is clipped to real activity, marked with a star, and flagged is-complete false, and any rate over it divides by the clipped width.",
    caption:
      "The orange band is grid overrun past real activity. Slice colour names the bin; the dot marks the binning key.",
    nodes: [
      slices(
        "sl",
        {
          title: "slice_duration = 15 — grid runs to 60, activity ends at 50",
          requests: REQUESTS.map((r) => ({ id: r.id, start: r.start, end: r.end })),
          duration: 15,
          axisLabel: "time (relative ns) · dot = record start (binning key)",
        },
        { col: 0, row: 0 },
      ),
      note(
        "bin",
        "Binned by start, counted once",
        "B runs 3 to 30 across three slices and still belongs to slice 0 alone. Overlap-based binning would count it three times and inflate every per-slice total.",
        { col: 0, row: 2.3 },
      ),
      note(
        "trail",
        "Flag it, do not drop it",
        "Dropping the partial bucket loses real records; keeping it at full width dilutes the rate. Clipping plus is_complete=false keeps the records and tells the consumer what happened.",
        { col: 1.1, row: 2.3 },
      ),
    ],
    edges: [],
    revealOrder: ["sl", "bin", "trail"],
  },
  {
    id: "shapes",
    eyebrow: "04 · WHY THESE SHAPES",
    title: "Three layouts, one constraint",
    lede:
      "Every choice here is the same trade: pay once at ingest to make the whole-dataset question answerable without a per-record loop.",
    narration:
      "Step back and the three decisions are one decision. A column store with lazily created columns and running sums answers a total in constant time. The flat ragged layout answers a per-record question with a mask. The sweep line answers an in-flight question by sorting events instead of scanning time. In each case the cost is paid once, at ingest, in exchange for never looping per record at query time. That is what lets the accumulator stay ahead of a run that is producing records faster than any Python loop could consume them.",
    caption:
      "Missing scalars are stored as NaN rather than reshaping the array; the row index is append-only and never reused across phases.",
    nodes: [
      band("b-in", "PAY ONCE · AT INGEST", { col: 0, row: 0.6 }),
      card("col", "column store", "lazy columns, running sums", "data", { col: 0, row: 0.6 }),
      card("flat", "flat ragged arrays", "values + indices + offsets", "data", { col: 1.05, row: 0.6 }),
      card("evt", "signed event stream", "sorted once", "time", { col: 2.1, row: 0.6 }),

      band("b-q", "NEVER · PER RECORD", { col: 0, row: 2.1 }),
      card("tot", "O(1) totals", "no rescan", "done", { col: 0, row: 2.1 }),
      card("msk", "boolean mask", "no per-record loop", "done", { col: 1.05, row: 2.1 }),
      card("cur", "exact step curve", "no timeline scan", "done", { col: 2.1, row: 2.1 }),

      chip("nan", "NaN, not reshape", { col: 3.2, row: 0.6 }),
      note(
        "why",
        "The constraint behind all three",
        "Records arrive faster than a per-record Python loop can consume them. Every layout here exists so the query side never has to run one.",
        { col: 3.2, row: 1.6 },
      ),
    ],
    edges: [
      link("col", "tot", "done"),
      link("flat", "msk", "done"),
      link("evt", "cur", "done"),
      ...fanIn(["tot", "msk", "cur"], "why", "muted", "slow"),
    ],
    revealOrder: [
      "b-in", "col", "flat", "evt", "nan", "b-q", "tot", "msk", "cur", "why",
    ],
  },
];

/** Registered in `App.tsx`; served by the generic `DeckRoute` at `/metrics-plane`. */
export const METRICS_PLANE_DECK: DeckDefinition = {
  id: "metrics-plane",
  title: "The Metrics Plane: sweep lines, ragged arrays, and time slices",
  slides: SLIDES,
};
