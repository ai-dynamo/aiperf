// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { parseFlowIr, safeParseFlowIr } from "../src/ir.js";

const sourceMap = {
  source: "p0.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

describe("Flow IR", () => {
  test("parses component nodes with semantic and layout attachments", () => {
    const flow = parseFlowIr({
      irVersion: 2,
      id: "token-span-morph",
      title: "Token span morph",
      capabilities: [{ id: "core.span-map", range: "1.0.0" }],
      tokens: {},
      themes: [],
      scenes: [
        {
          id: "main",
          title: "Main",
          summary: "Morph demo",
          roots: [
            {
              kind: "component",
              id: "tok-map",
              capabilityId: "core.span-map",
              props: { requireCover: "source" },
              semanticModel: {
                entities: [{ id: "t0", label: "151643" }],
                relations: [],
                morphs: [
                  {
                    id: "e0",
                    sourceIds: ["g0"],
                    targetIds: ["t0"],
                    kind: "one-to-one",
                  },
                ],
              },
              layoutPlan: {
                version: 1,
                nodes: [
                  {
                    nodeId: "tok-map",
                    bounds: { x: 8, y: 8, width: 400, height: 80 },
                  },
                ],
                routes: [],
              },
              children: [],
              geometry: { x: 0, y: 0, width: 416, height: 96 },
              style: {},
              accessibility: { label: "Span map" },
              fallback: "Span map unavailable",
              sourceMap,
            },
          ],
          camera: [],
          timeline: [],
          narration: "",
          interactions: [],
          responsive: [],
          accessibility: { label: "Main scene", readingOrder: ["tok-map"] },
          fallback: "Scene unavailable",
          sourceMap,
        },
      ],
      sourceMap,
    });

    const component = flow.scenes[0]?.roots[0];
    expect(component?.kind).toBe("component");
    if (component?.kind === "component") {
      expect(component.capabilityId).toBe("core.span-map");
      expect(component.semanticModel?.entities[0]?.id).toBe("t0");
    }
  });

  test("rejects unknown component props fields at the node level", () => {
    const result = safeParseFlowIr({
      irVersion: 2,
      id: "bad",
      title: "Bad",
      capabilities: [],
      tokens: {},
      themes: [],
      scenes: [
        {
          id: "main",
          title: "Main",
          summary: "Bad scene",
          roots: [
            {
              kind: "component",
              id: "tok-map",
              capabilityId: "core.span-map",
              props: { requireCover: "source" },
              children: [],
              geometry: { x: 0, y: 0, width: 10, height: 10 },
              style: {},
              accessibility: { label: "Span map" },
              fallback: "Fallback",
              sourceMap,
              unknown: true,
            },
          ],
          camera: [],
          timeline: [],
          narration: "",
          interactions: [],
          responsive: [],
          accessibility: { label: "Main", readingOrder: [] },
          fallback: "Fallback",
          sourceMap,
        },
      ],
      sourceMap,
    });

    expect(result.ok).toBe(false);
  });
});
