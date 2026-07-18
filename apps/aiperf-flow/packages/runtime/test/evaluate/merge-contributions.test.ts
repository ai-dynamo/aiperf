// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  mergeContributions,
  type EvaluatorContribution,
} from "../../src/evaluate/merge-contributions.js";

const zeroBounds = { x: 0, y: 0, width: 0, height: 0 } as const;

function contribution(
  id: string,
  sourceOrder: number,
  overrides: Partial<EvaluatorContribution> = {},
): EvaluatorContribution {
  return {
    id,
    sourceOrder,
    commands: [],
    hitRegions: [],
    semanticEntities: [],
    semanticRelations: [],
    readingOrder: [],
    captions: [],
    diagnostics: [],
    ...overrides,
  };
}

describe("mergeContributions", () => {
  test("merges every contribution product in stable source order", () => {
    const later = contribution("alpha", 2, {
      commands: [
        {
          id: "command-a",
          kind: "path",
          order: 0,
          path: "M 20 10 H 30 V 20 H 20 Z",
          paintBounds: { x: 20, y: 10, width: 10, height: 10 },
          damageBounds: { x: 19, y: 9, width: 12, height: 12 },
        },
      ],
      hitRegions: [
        {
          id: "hit-a",
          semanticId: "entity-b",
          order: 0,
          bounds: { x: 20, y: 10, width: 10, height: 10 },
        },
      ],
      paintBounds: { x: 20, y: 10, width: 10, height: 10 },
      damageBounds: { x: 19, y: 9, width: 12, height: 12 },
      semanticEntities: [{ id: "entity-b", label: "B" }],
      semanticRelations: [
        { id: "relation-b", fromId: "entity-a", toId: "entity-b" },
      ],
      readingOrder: ["entity-b", "relation-b"],
      captions: ["second"],
      diagnostics: [
        {
          id: "diagnostic-b",
          severity: "warning",
          message: "Second contribution",
        },
      ],
    });
    const earlier = contribution("zeta", 1, {
      commands: [
        {
          id: "command-z",
          kind: "path",
          order: 0,
          path: "M -5 0 H 5 V 5 H -5 Z",
          paintBounds: { x: -5, y: 0, width: 10, height: 5 },
          damageBounds: { x: -6, y: -1, width: 12, height: 7 },
        },
      ],
      hitRegions: [
        {
          id: "hit-z",
          semanticId: "entity-a",
          order: 0,
          bounds: { x: -5, y: 0, width: 10, height: 5 },
        },
      ],
      paintBounds: { x: -5, y: 0, width: 10, height: 5 },
      damageBounds: { x: -6, y: -1, width: 12, height: 7 },
      semanticEntities: [{ id: "entity-a", label: "A" }],
      readingOrder: ["entity-a"],
      captions: ["first"],
      diagnostics: [
        {
          id: "diagnostic-a",
          severity: "info",
          message: "First contribution",
        },
      ],
    });

    const merged = mergeContributions("scene", [later, earlier]);

    expect(merged.displayList.commands.map(({ id }) => id)).toEqual([
      "command-z",
      "command-a",
    ]);
    expect(merged.displayList.commands.map(({ order }) => order)).toEqual([0, 1]);
    expect(merged.displayList.hitRegions.map(({ id }) => id)).toEqual([
      "hit-z",
      "hit-a",
    ]);
    expect(merged.displayList.paintBounds).toEqual({
      x: -5,
      y: 0,
      width: 35,
      height: 20,
    });
    expect(merged.displayList.damageBounds).toEqual({
      x: -6,
      y: -1,
      width: 37,
      height: 22,
    });
    expect(merged.semantic).toEqual({
      sceneId: "scene",
      entities: [
        { id: "entity-a", label: "A" },
        { id: "entity-b", label: "B" },
      ],
      relations: [
        { id: "relation-b", fromId: "entity-a", toId: "entity-b" },
      ],
      readingOrder: ["entity-a", "entity-b", "relation-b"],
      captions: ["first", "second"],
    });
    expect(merged.diagnostics.map(({ id }) => id)).toEqual([
      "diagnostic-a",
      "diagnostic-b",
    ]);
    expect(Object.isFrozen(merged)).toBe(true);
    expect(Object.isFrozen(merged.semantic.entities)).toBe(true);
    expect(Object.isFrozen(merged.diagnostics[0])).toBe(true);
  });

  test("uses fragment id to break equal source-order ties", () => {
    const merged = mergeContributions("scene", [
      contribution("zeta", 4, { captions: ["zeta"] }),
      contribution("alpha", 4, { captions: ["alpha"] }),
    ]);

    expect(merged.semantic.captions).toEqual(["alpha", "zeta"]);
  });

  test("returns zero bounds and no captions for empty contributions", () => {
    const merged = mergeContributions("scene", []);

    expect(merged.displayList.paintBounds).toEqual(zeroBounds);
    expect(merged.displayList.damageBounds).toEqual(zeroBounds);
    expect(merged.semantic).toEqual({
      sceneId: "scene",
      entities: [],
      relations: [],
      readingOrder: [],
    });
    expect(merged.diagnostics).toEqual([]);
  });

  test.each([
    [
      "contribution",
      [
        contribution("duplicate", 0),
        contribution("duplicate", 1),
      ] as const,
    ],
    [
      "command",
      [
        contribution("a", 0, {
          commands: [
            {
              id: "duplicate",
              kind: "path",
              order: 0,
              path: "",
              paintBounds: zeroBounds,
              damageBounds: zeroBounds,
            },
          ],
        }),
        contribution("b", 1, {
          commands: [
            {
              id: "duplicate",
              kind: "path",
              order: 1,
              path: "",
              paintBounds: zeroBounds,
              damageBounds: zeroBounds,
            },
          ],
        }),
      ] as const,
    ],
    [
      "hit region",
      [
        contribution("a", 0, {
          hitRegions: [
            {
              id: "duplicate",
              semanticId: "a",
              order: 0,
              bounds: zeroBounds,
            },
          ],
        }),
        contribution("b", 1, {
          hitRegions: [
            {
              id: "duplicate",
              semanticId: "b",
              order: 1,
              bounds: zeroBounds,
            },
          ],
        }),
      ] as const,
    ],
    [
      "semantic entity",
      [
        contribution("a", 0, {
          semanticEntities: [{ id: "duplicate", label: "A" }],
        }),
        contribution("b", 1, {
          semanticEntities: [{ id: "duplicate", label: "B" }],
        }),
      ] as const,
    ],
    [
      "semantic relation",
      [
        contribution("a", 0, {
          semanticRelations: [
            { id: "duplicate", fromId: "a", toId: "b" },
          ],
        }),
        contribution("b", 1, {
          semanticRelations: [
            { id: "duplicate", fromId: "b", toId: "a" },
          ],
        }),
      ] as const,
    ],
    [
      "diagnostic",
      [
        contribution("a", 0, {
          diagnostics: [
            { id: "duplicate", severity: "info", message: "A" },
          ],
        }),
        contribution("b", 1, {
          diagnostics: [
            { id: "duplicate", severity: "warning", message: "B" },
          ],
        }),
      ] as const,
    ],
  ])("fails closed on duplicate %s ids", (kind, contributions) => {
    expect(() => mergeContributions("scene", contributions)).toThrow(
      `Duplicate ${kind} id "duplicate".`,
    );
  });

  test("fails closed on repeated reading-order ids", () => {
    expect(() =>
      mergeContributions("scene", [
        contribution("a", 0, {
          semanticEntities: [{ id: "entity", label: "Entity" }],
          readingOrder: ["entity"],
        }),
        contribution("b", 1, { readingOrder: ["entity"] }),
      ]),
    ).toThrow('Duplicate reading-order id "entity".');
  });

  test.each([
    [
      "hit region",
      contribution("a", 0, {
        hitRegions: [
          {
            id: "hit",
            semanticId: "missing",
            order: 0,
            bounds: zeroBounds,
          },
        ],
      }),
      'Hit region "hit" references unknown semantic id "missing".',
    ],
    [
      "relation",
      contribution("a", 0, {
        semanticEntities: [{ id: "known", label: "Known" }],
        semanticRelations: [
          { id: "relation", fromId: "known", toId: "missing" },
        ],
      }),
      'Semantic relation "relation" references unknown entity id "missing".',
    ],
    [
      "reading order",
      contribution("a", 0, { readingOrder: ["missing"] }),
      'Reading order references unknown semantic id "missing".',
    ],
  ])("fails closed on dangling %s references", (_kind, item, message) => {
    expect(() => mergeContributions("scene", [item])).toThrow(message);
  });
});
