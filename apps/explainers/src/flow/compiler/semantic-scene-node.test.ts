/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { sceneNodeSchema } from "../schema/ir.js";
import { lowerSemanticSceneNode } from "./semantic-scene-node.js";

const SOURCE_MAP = {
  source: "semantic.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

describe("semantic Scene node lowering", () => {
  it("retains panel payload without generating visual children", () => {
    const panel = lowerSemanticSceneNode(
      {
        id: "panel",
        capability: "core.panel",
        geometry: { x: 10, y: 20, width: 160, height: 64 },
        title: "Profile",
        detail: "source",
      },
      {
        id: "panel",
        capability: "core.panel",
        children: [],
        label: "Profile",
        fallback: "Profile",
        sourceMap: SOURCE_MAP,
      },
    );

    expect(panel).toMatchObject({
      id: "panel",
      kind: "group",
      capabilityId: "core.panel",
      props: { title: "Profile", detail: "source" },
      children: [],
    });
    expect(sceneNodeSchema.parse(panel)).toEqual(panel);
  });

  it("retains semantic stepper labels and linkage", () => {
    const stepper = lowerSemanticSceneNode(
      {
        id: "steps",
        capability: "core.stepper",
        geometry: { x: 0, y: 0, width: 160, height: 90 },
        steps: ["layout", "slots", "timeline"],
        linked: true,
        gap: 16,
      },
      {
        id: "steps",
        capability: "core.stepper",
        children: [],
        label: "stepper",
        fallback: "stepper",
        sourceMap: SOURCE_MAP,
      },
    );

    expect(stepper).toMatchObject({
      capabilityId: "core.stepper",
      props: {
        steps: ["layout", "slots", "timeline"],
        linked: true,
        gap: 16,
      },
      children: [],
    });
    expect(sceneNodeSchema.parse(stepper)).toEqual(stepper);
  });
});

