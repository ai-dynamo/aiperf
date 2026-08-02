/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-language parity: the TypeScript t-digest against the Rust one.
//!
//! `sketchSim.ts` is a hand port of `rust/runtime/src/cellular/sketch.rs`, and the explainer page
//! presents its output as the runtime's real numbers. A port with no pin against the original can
//! drift silently and then teach something false with complete confidence, so both sides are
//! pinned to one committed fixture:
//!
//! - the Rust test `tdigest_golden_fixture_matches_this_implementation` replays the fixture's
//!   inputs and asserts the Rust still produces the recorded outputs;
//! - this test replays the same inputs and asserts the port produces them too.
//!
//! Regenerate with `UPDATE_SKETCH_GOLDEN=1 cargo test -p aiperf-runtime --lib tdigest_golden`.
//! If the Rust changes, that test fails first and this one follows — which is the point.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { clustered, createDigest, extendFrom, merge, quantile, type TDigest } from "./sketchSim.js";

type CaseDigest = {
  count: number;
  min: number;
  max: number;
  centroid_count: number;
  centroid_means: number[];
  centroid_weights: number[];
  quantiles: (number | null)[];
};

// Vitest runs from the package root, so the fixture resolves relative to that rather than to this
// file's URL — the app is bundled for the browser and `import.meta.url` is not a file URL there.
const FIXTURE = JSON.parse(
  readFileSync(resolve(process.cwd(), "../../tools/parity/sketch_golden/tdigest.json"), "utf8"),
) as {
  compression: number;
  quantile_band: number[];
  cases: {
    broad: { input: number[]; digest: CaseDigest };
    folded: { cells: number[][]; digest: CaseDigest };
    tiny: { input: number[]; digest: CaseDigest };
  };
};

/** Same bound the Rust side uses — far tighter than any behaviour change worth catching. */
const REL_TOLERANCE = 1e-9;

function expectClose(actual: number, expected: number, what: string): void {
  // Finiteness is checked before the tolerance, because a relative bound against a non-finite
  // value degenerates: 1e-9 * Infinity is Infinity, and Infinity <= Infinity passes. Mutation
  // testing found exactly that hole — dropping min-preservation from `merge` leaves min at
  // +Infinity and the comparison waved it through.
  expect(Number.isFinite(actual), `${what}: expected a finite number, got ${actual}`).toBe(true);
  expect(Number.isFinite(expected), `${what}: fixture value not finite (${expected})`).toBe(true);
  const bound = REL_TOLERANCE * Math.max(Math.abs(actual), Math.abs(expected), 1);
  expect(Math.abs(actual - expected), `${what}: ${actual} vs ${expected}`).toBeLessThanOrEqual(bound);
}

function digestOf(values: readonly number[]): TDigest {
  const digest = createDigest(FIXTURE.compression);
  extendFrom(digest, values);
  digest.centroids = clustered(digest);
  return digest;
}

function checkAgainst(digest: TDigest, expected: CaseDigest, label: string): void {
  const centroids = clustered(digest);
  expect(digest.totalWeight, `${label}.count`).toBe(expected.count);
  expectClose(digest.min, expected.min, `${label}.min`);
  expectClose(digest.max, expected.max, `${label}.max`);
  // Centroid count is the sharpest signal that the clustering rule itself diverged: a wrong K1
  // scale or a wrong merge condition changes how many clusters survive before it changes any
  // quantile enough to notice.
  expect(centroids.length, `${label}.centroid_count`).toBe(expected.centroid_count);
  centroids.forEach((c, i) => {
    expectClose(c.mean, expected.centroid_means[i]!, `${label}.centroid_means[${i}]`);
    expectClose(c.weight, expected.centroid_weights[i]!, `${label}.centroid_weights[${i}]`);
  });
  FIXTURE.quantile_band.forEach((q, i) => {
    const want = expected.quantiles[i];
    const got = quantile(digest, q);
    if (want === null) {
      expect(got, `${label}.quantiles[${i}]`).toBeNull();
      return;
    }
    expectClose(got!, want!, `${label}.quantile(${q})`);
  });
}

describe("t-digest parity with the Rust implementation", () => {
  it("matches on a broad sample", () => {
    checkAgainst(digestOf(FIXTURE.cases.broad.input), FIXTURE.cases.broad.digest, "broad");
  });

  it("matches after folding three cells", () => {
    // Exercises `merge`, which is the operation the whole cellular story rests on.
    const folded = createDigest(FIXTURE.compression);
    for (const cell of FIXTURE.cases.folded.cells) merge(folded, digestOf(cell));
    checkAgainst(folded, FIXTURE.cases.folded.digest, "folded");
  });

  it("matches on a tiny input where every centroid stays weight-1", () => {
    // The least forgiving case: no clustering to hide behind, so quantile interpolation and the
    // min/max anchors are compared directly.
    checkAgainst(digestOf(FIXTURE.cases.tiny.input), FIXTURE.cases.tiny.digest, "tiny");
  });

  it("uses the same compression constant", () => {
    expect(createDigest().compression).toBe(FIXTURE.compression);
  });
});
