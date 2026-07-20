/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { Bounds } from "../display-list.js";
import {
  applyQualityPolicy,
  qualityPolicyProfile,
  type QualityAnnotatedCommand,
  type QualityDisplayList,
} from "./quality-policy.js";

const BOUNDS: Bounds = { x: 0, y: 0, width: 10, height: 10 };

function pathCommand(
  id: string,
  order: number,
  extras: Partial<Omit<Extract<QualityAnnotatedCommand, { kind: "path" }>, "kind">> = {},
): Extract<QualityAnnotatedCommand, { kind: "path" }> {
  return {
    id,
    kind: "path",
    order,
    paintBounds: BOUNDS,
    damageBounds: BOUNDS,
    path: "M 0 0 L 10 10",
    ...extras,
  };
}

function displayList(
  commands: readonly QualityAnnotatedCommand[],
): QualityDisplayList {
  return {
    commands,
    hitRegions: [],
    paintBounds: BOUNDS,
    damageBounds: BOUNDS,
  };
}

describe("applyQualityPolicy suppressedCommandIndices", () => {
  it("reports budget suppressions in the original command-tree index space", () => {
    // Original pre-order indices:
    //   0 required, 1 blur, 2 glow, 3 particles, 4 shadow
    // Filter (reference + supportedFamilies excluding blur) drops index 1.
    // Filtered tree pre-order: required, glow, particles, shadow
    // Budget maxDecorativeCommands=1 keeps glow, drops particles+shadow.
    // Those must map back to original indices 3 and 4 — not filtered 2 and 3.
    const list = displayList([
      pathCommand("required", 0, { qualityClass: "required-semantic" }),
      pathCommand("blur", 1, {
        qualityClass: "decorative",
        decorativeFamily: "blur",
      }),
      pathCommand("glow", 2, {
        qualityClass: "decorative",
        decorativeFamily: "glow",
      }),
      pathCommand("particles", 3, {
        qualityClass: "decorative",
        decorativeFamily: "particles",
      }),
      pathCommand("shadow", 4, {
        qualityClass: "decorative",
        decorativeFamily: "shadow",
      }),
    ]);

    const result = applyQualityPolicy(
      list,
      qualityPolicyProfile("reference"),
      {
        supportedDecorativeFamilies: ["glow", "particles", "shadow"],
        maxDecorativeCommands: 1,
      },
    );

    // Original indices: filter drops blur(1); budget drops particles(3)+shadow(4).
    // Bug concatenates filtered-tree budget indices [2,3] → wrong [1,2,3].
    expect(result.report.suppressedCommandIndices).toEqual([1, 3, 4]);
    expect(result.list.commands.map(({ id }) => id)).toEqual([
      "required",
      "glow",
    ]);
  });

  it("keeps filter-only suppressions in original index space when no budget runs", () => {
    const list = displayList([
      pathCommand("required", 0, { qualityClass: "required-semantic" }),
      pathCommand("blur", 1, {
        qualityClass: "decorative",
        decorativeFamily: "blur",
      }),
      pathCommand("glow", 2, {
        qualityClass: "decorative",
        decorativeFamily: "glow",
      }),
    ]);

    const result = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(result.list.commands.map(({ id }) => id)).toEqual(["required"]);
    expect(result.report.suppressedCommandIndices).toEqual([1, 2]);
  });
});
