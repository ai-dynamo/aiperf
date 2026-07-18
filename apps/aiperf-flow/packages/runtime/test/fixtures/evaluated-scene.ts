// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildDisplayList, type DisplayList } from "../../src/display-list.js";
import type { EvaluatedScene } from "../../src/evaluate/types.js";

export const CONFORMANCE_ENTITY = {
  id: "request-a",
  label: "Request A",
  focusTarget: "request-a",
} as const;

export const CONFORMANCE_SELECTION = {
  focusedEntityId: CONFORMANCE_ENTITY.id,
  selectedEntityIds: [CONFORMANCE_ENTITY.id],
} as const;

export const evaluatedSceneFixture: EvaluatedScene = {
  sceneId: "backend-conformance",
  atMs: 0,
  displayList: buildDisplayList({
    commands: [
      {
        kind: "path",
        id: "request-shape",
        order: 0,
        path: "M 16 16 H 112 V 48 H 16 Z",
        fill: "#76b900",
        paintBounds: { x: 16, y: 16, width: 96, height: 32 },
        damageBounds: { x: 16, y: 16, width: 96, height: 32 },
      },
    ],
    hitRegions: [
      {
        id: "request-hit",
        semanticId: CONFORMANCE_ENTITY.id,
        order: 0,
        bounds: { x: 16, y: 16, width: 96, height: 32 },
        label: CONFORMANCE_ENTITY.label,
        focusTarget: CONFORMANCE_ENTITY.focusTarget,
        selected: true,
        focusable: true,
      } as DisplayList["hitRegions"][number] & {
        label: string;
        focusTarget: string;
        selected: boolean;
        focusable: boolean;
      },
    ],
    paintBounds: { x: 16, y: 16, width: 96, height: 32 },
    damageBounds: { x: 16, y: 16, width: 96, height: 32 },
  }),
  semantic: {
    sceneId: "backend-conformance",
    readingOrder: [CONFORMANCE_ENTITY.id],
    entities: [
      {
        id: CONFORMANCE_ENTITY.id,
        label: CONFORMANCE_ENTITY.label,
        focusTarget: CONFORMANCE_ENTITY.focusTarget,
        selected: true,
      },
    ],
    relations: [],
  },
};

export const displayListFixture: DisplayList = evaluatedSceneFixture.displayList;
