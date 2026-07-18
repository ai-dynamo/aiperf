/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import {
  FOUNDATION_CAPABILITIES,
  P0_CAPABILITIES,
  P0_CAPABILITY_IDS,
} from "../src/index.js";

describe("P0 capabilities", () => {
  it("registers hybrid components and leaf kernels", () => {
    expect([...P0_CAPABILITY_IDS].sort()).toEqual([
      "core.glyph-run",
      "core.segment-strip",
      "core.segment-strip.layout",
      "core.semantic-morph",
      "core.span-map",
      "leaf.glyph-measure",
      "leaf.span-interval",
      "viz.queue",
      "viz.queue.policy",
      "viz.waterfall",
      "viz.waterfall.nest-layout",
    ]);
  });

  it("includes foundation capabilities in the P0 manifest", () => {
    const ids = P0_CAPABILITIES.capabilities.map(({ id }) => id);
    for (const { id } of FOUNDATION_CAPABILITIES.capabilities) {
      expect(ids).toContain(id);
    }
  });

  it("keeps the combined manifest sorted and unique", () => {
    const ids = P0_CAPABILITIES.capabilities.map(({ id }) => id);
    expect(ids).toEqual([...ids].sort((left, right) => left.localeCompare(right)));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("marks every P0 leaf and hybrid as deterministic", () => {
    for (const id of P0_CAPABILITY_IDS) {
      const descriptor = P0_CAPABILITIES.capabilities.find(
        (candidate) => candidate.id === id,
      );
      expect(descriptor?.deterministic).toBe(true);
    }
  });
});
