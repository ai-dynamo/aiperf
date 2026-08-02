/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  add,
  centroidSpans,
  clustered,
  compare,
  count,
  createDigest,
  exactPercentile,
  extendFrom,
  foldCells,
  kScale,
  latencySamples,
  merge,
  quantile,
  splitAcrossCells,
  DEFAULT_COMPRESSION,
} from "./sketchSim.js";

const digestOf = (values: readonly number[]) => {
  const d = createDigest();
  extendFrom(d, values);
  return d;
};

describe("what the sketch keeps exact", () => {
  it("returns the true minimum and maximum, never an estimate", () => {
    // min/max are tracked as scalars beside the centroids, so no clustering can move them.
    const values = latencySamples(5_000, 7);
    const digest = digestOf(values);
    expect(digest.min).toBe(Math.min(...values));
    expect(digest.max).toBe(Math.max(...values));
  });

  it("anchors quantile 0 and 1 on those exact values", () => {
    const values = latencySamples(2_000, 11);
    const digest = digestOf(values);
    expect(quantile(digest, 0)).toBe(Math.min(...values));
    expect(quantile(digest, 1)).toBe(Math.max(...values));
  });

  it("counts every ingested value", () => {
    expect(count(digestOf(latencySamples(3_333, 3)))).toBe(3_333);
  });

  it("keeps min and max exact through a fold across cells", () => {
    const values = latencySamples(6_000, 5);
    const folded = foldCells(splitAcrossCells(values, 4));
    expect(folded.min).toBe(Math.min(...values));
    expect(folded.max).toBe(Math.max(...values));
    expect(count(folded)).toBe(values.length);
  });

  it("ignores non-finite values rather than storing sentinels", () => {
    const digest = createDigest();
    add(digest, 1);
    add(digest, Number.NaN);
    add(digest, Number.POSITIVE_INFINITY);
    add(digest, 3);
    expect(count(digest)).toBe(2);
    expect(digest.max).toBe(3);
  });
});

describe("what it only estimates", () => {
  it("tracks the exact percentile closely but not identically", () => {
    // The claim in the runtime's own docs: well under a percent on a broad distribution. Asserting
    // "close but not equal" is the honest form — asserting equality would be asserting it is not a
    // sketch at all.
    const values = latencySamples(20_000, 13);
    const sorted = [...values].sort((a, b) => a - b);
    const digest = digestOf(values);
    for (const p of [50, 90, 99]) {
      const exact = exactPercentile(sorted, p);
      const sketch = quantile(digest, p / 100)!;
      const errorPct = Math.abs((sketch - exact) / exact) * 100;
      expect(errorPct, `p${p} error ${errorPct.toFixed(3)}%`).toBeLessThan(1);
    }
  });

  it("stores far fewer centroids than values", () => {
    // The point of the sketch: bounded memory. ~δ/2 centroids after compression.
    const digest = digestOf(latencySamples(50_000, 17));
    expect(clustered(digest).length).toBeLessThan(DEFAULT_COMPRESSION);
    expect(count(digest)).toBe(50_000);
  });
});

describe("the K1 scale — where the resolution goes", () => {
  it("spends narrow quantile bands at the tails and wide ones in the body", () => {
    // The counter-intuitive part. asin steepens towards ±1, so one scale unit covers less
    // quantile space at the extremes: the digest is FINEST at the tail, coarsest at the median.
    const digest = digestOf(latencySamples(20_000, 19));
    const spans = centroidSpans(digest);
    const width = (s: { q0: number; q1: number }) => s.q1 - s.q0;
    const tail = spans.filter((s) => s.q1 > 0.98);
    const body = spans.filter((s) => s.q0 > 0.4 && s.q1 < 0.6);
    const meanWidth = (xs: typeof spans) => xs.reduce((a, s) => a + width(s), 0) / xs.length;
    expect(tail.length).toBeGreaterThan(0);
    expect(body.length).toBeGreaterThan(0);
    expect(meanWidth(tail)).toBeLessThan(meanWidth(body));
  });

  it("is symmetric about the median", () => {
    const width = (q: number) => kScale(q + 0.01, 100) - kScale(q, 100);
    expect(width(0.01)).toBeCloseTo(width(0.98), 6);
  });
});

describe("where it does badly", () => {
  it("is an order of magnitude worse across a cliff in the distribution", () => {
    // Quantiles interpolate linearly between centroid means, so a step is the one shape this
    // representation cannot express. A bimodal sample puts a near-vertical jump at p90 and the
    // error there dwarfs everything the smooth case produces.
    const err = (shape: "lognormal" | "bimodal", p: number) => {
      const values = latencySamples(20_000, 13, shape);
      const sorted = [...values].sort((a, b) => a - b);
      const exact = exactPercentile(sorted, p);
      return Math.abs((quantile(digestOf(values), p / 100)! - exact) / exact) * 100;
    };
    expect(err("bimodal", 90)).toBeGreaterThan(err("lognormal", 90) * 10);
    // And away from the cliff the same sample is fine, so it is the shape and not the sample.
    expect(err("bimodal", 99)).toBeLessThan(1);
  });
});

describe("merging", () => {
  it("is stable under fold order, but not bit-identical", () => {
    // The runtime's wording is precise: deterministic at a FIXED topology, order-independent up to
    // floating point. Reversing the fold changes the topology, so asserting equality would assert
    // something the implementation never claimed.
    //
    // Measured here: p50 and p90 move by well under 0.1%, but p99 moves ~1%. The extreme tail is
    // the most sensitive because its centroids carry the least weight each, so re-clustering them
    // in a different order shifts their means furthest. Worth knowing before treating a folded p99
    // as reproducible to the last digit.
    const values = latencySamples(8_000, 23);
    const cells = splitAcrossCells(values, 4);
    const forward = foldCells(cells);
    const backward = foldCells([...cells].reverse());
    for (const p of [0.5, 0.9, 0.99]) {
      const a = quantile(forward, p)!;
      const b = quantile(backward, p)!;
      const deltaPct = Math.abs((a - b) / a) * 100;
      expect(deltaPct, `p${p * 100} moved ${deltaPct.toFixed(4)}% on fold order`).toBeLessThan(1.5);
    }
  });

  it("is concatenate-then-compress, so a merged digest stays bounded", () => {
    const a = digestOf(latencySamples(10_000, 29));
    const b = digestOf(latencySamples(10_000, 31));
    merge(a, b);
    expect(count(a)).toBe(20_000);
    expect(a.centroids.length).toBeLessThan(DEFAULT_COMPRESSION);
  });

  it("folds cells to within a percent of the whole-run exact percentiles", () => {
    const values = latencySamples(24_000, 37);
    const folded = foldCells(splitAcrossCells(values, 6));
    for (const row of compare(values, folded).filter((r) => !r.guaranteed)) {
      expect(Math.abs(row.errorPct), `${row.label} off by ${row.errorPct.toFixed(3)}%`).toBeLessThan(1);
    }
  });

  it("reports every guaranteed figure identically", () => {
    const values = latencySamples(12_000, 41);
    const folded = foldCells(splitAcrossCells(values, 3));
    for (const row of compare(values, folded).filter((r) => r.guaranteed)) {
      expect(row.sketch, row.label).toBeCloseTo(row.exact, 9);
    }
  });
});
