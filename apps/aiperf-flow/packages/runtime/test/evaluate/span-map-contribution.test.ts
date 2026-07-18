// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test, vi } from "vitest";

import { contributeSpanMap } from "../../src/evaluate/contributions/span-map.js";

const componentSource: SourceRange = {
  source: "token-span.flow",
  start: { offset: 40, line: 3, column: 1 },
  end: { offset: 90, line: 6, column: 2 },
};

const edgeSource: SourceRange = {
  source: "token-span.flow",
  start: { offset: 72, line: 5, column: 3 },
  end: { offset: 84, line: 5, column: 15 },
};

describe("contributeSpanMap", () => {
  test("projects authored span and relation identities into backend-neutral products", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });

    const contribution = contributeSpanMap({
      id: "token-map",
      source: componentSource,
      requireCover: "none",
      spans: [
        {
          id: "prompt:hello",
          label: "Hello",
          start: 0,
          end: 5,
          bounds: { x: 0, y: 0, width: 50, height: 20 },
        },
        {
          id: "token:1",
          label: "Hello",
          start: 0,
          end: 5,
          bounds: { x: 80, y: 40, width: 50, height: 20 },
        },
      ],
      edges: [
        {
          id: "maps-to",
          sourceSpanIds: ["prompt:hello"],
          targetSpanIds: ["token:1"],
          kind: "tokenization",
          source: edgeSource,
        },
      ],
    });

    expect(contribution.semanticEntities.map(({ id }) => id)).toEqual([
      "prompt:hello",
      "token:1",
    ]);
    expect(contribution.semanticRelations).toEqual([
      {
        id: "maps-to",
        fromId: "prompt:hello",
        toId: "token:1",
        label: "tokenization",
        role: "tokenization",
        source: {
          source: "token-span.flow",
          startOffset: 72,
          endOffset: 84,
        },
      },
    ]);
    expect(contribution.commands).toHaveLength(3);
    expect(contribution.commands.every(({ kind }) => kind === "path")).toBe(true);
    expect(contribution.hitRegions.map(({ semanticId }) => semanticId)).toEqual([
      "prompt:hello",
      "token:1",
      "maps-to",
    ]);
    expect(contribution.diagnostics).toEqual([]);
    expect(dateNow).not.toHaveBeenCalled();
  });

  test("reports required coverage gaps at the component source and marks the span", () => {
    const contribution = contributeSpanMap({
      id: "token-map",
      source: componentSource,
      requireCover: "source",
      spans: [
        {
          id: "prompt:gap",
          label: " gap",
          start: 5,
          end: 9,
          bounds: { x: 50, y: 0, width: 40, height: 20 },
        },
      ],
      edges: [],
    });

    expect(contribution.diagnostics).toEqual([
      {
        code: "SPAN_MAP_COVERAGE_GAP",
        severity: "warning",
        message:
          'Span "prompt:gap" does not satisfy required "source" edge coverage in "token-map".',
        range: componentSource,
        repair:
          'Add an edge covering "prompt:gap" or relax the requireCover policy.',
      },
    ]);
    expect(contribution.semanticEntities[0]).toMatchObject({
      id: "prompt:gap",
      kind: "span-uncovered",
      description: 'Missing required "source" mapping coverage.',
    });
    expect(contribution.commands[0]).toMatchObject({
      id: "token-map:span:prompt:gap",
      stroke: "#ef4444",
    });
  });

  test("keeps relation identity stable across authored geometry overrides", () => {
    const input = {
      id: "token-map",
      source: componentSource,
      requireCover: "none" as const,
      spans: [
        {
          id: "source",
          label: "source",
          start: 0,
          end: 2,
          bounds: { x: 0, y: 0, width: 20, height: 20 },
        },
        {
          id: "target",
          label: "target",
          start: 0,
          end: 2,
          bounds: { x: 40, y: 40, width: 20, height: 20 },
        },
      ],
      edges: [
        {
          id: "edge",
          sourceSpanIds: ["source"],
          targetSpanIds: ["target"],
          kind: "maps",
        },
      ],
    };
    const moved = {
      ...input,
      spans: input.spans.map((span) => ({
        ...span,
        bounds: { ...span.bounds, x: span.bounds.x + 100 },
      })),
    };

    expect(contributeSpanMap(input).semanticRelations).toEqual(
      contributeSpanMap(moved).semanticRelations,
    );
  });

  test("returns deeply immutable contribution data", () => {
    const contribution = contributeSpanMap({
      id: "token-map",
      source: componentSource,
      requireCover: "none",
      spans: [
        {
          id: "span",
          label: "span",
          start: 0,
          end: 1,
          bounds: { x: 0, y: 0, width: 10, height: 10 },
        },
      ],
      edges: [],
    });

    expect(Object.isFrozen(contribution)).toBe(true);
    expect(Object.isFrozen(contribution.commands)).toBe(true);
    expect(Object.isFrozen(contribution.commands[0])).toBe(true);
    expect(Object.isFrozen(contribution.commands[0]?.paintBounds)).toBe(true);
    expect(Object.isFrozen(contribution.semanticEntities[0])).toBe(true);
  });
});
