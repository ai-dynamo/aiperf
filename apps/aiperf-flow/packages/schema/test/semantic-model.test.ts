// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  parseSemanticModel,
  safeParseSemanticModel,
  semanticEntityIds,
} from "../src/semantic-model.js";

describe("semantic model", () => {
  test("parses entities, relations, and morph correspondences", () => {
    const model = parseSemanticModel({
      entities: [{ id: "t0", label: "151643", kind: "id" }],
      relations: [{ id: "r0", from: "g0", to: "t0", kind: "maps-to" }],
      morphs: [
        {
          id: "e0",
          sourceIds: ["g0", "g1", "g2"],
          targetIds: ["t0"],
          kind: "many-to-one",
        },
      ],
    });

    expect(semanticEntityIds(model)).toEqual(["t0"]);
    expect(model.morphs[0]?.kind).toBe("many-to-one");
  });

  test("rejects unknown fields", () => {
    const result = safeParseSemanticModel({
      entities: [{ id: "t0", label: "151643", extra: true }],
      relations: [],
      morphs: [],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics[0]?.code).toBe("SEMANTIC_INVALID");
    }
  });
});
