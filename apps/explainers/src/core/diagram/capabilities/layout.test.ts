/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { SceneNodeLike } from "../SceneRenderer.js";
import {
  createCapabilityRegistry,
  resolveCapabilityLayout,
} from "./registry.js";
import type { NativeSceneCapability } from "./types.js";

function node(
  id: string,
  capabilityId: string,
  width: number,
  height: number,
  extras: Partial<SceneNodeLike> = {},
): SceneNodeLike {
  return {
    id,
    kind: "group",
    capabilityId,
    geometry: { x: 0, y: 0, width, height },
    style: {},
    children: [],
    ...extras,
  };
}

describe("native Scene capability layout", () => {
  it("expands a semantic stepper to fit numbered labels", () => {
    const stepper = node("steps", "core.stepper", 160, 90, {
      props: { steps: ["layout", "slots", "timeline"], linked: true },
      style: { gap: 16 },
    });

    const layout = resolveCapabilityLayout(stepper, []);

    expect(layout.bounds).toEqual({ x: 0, y: 0, width: 279, height: 90 });
    expect(layout.childGeometries).toEqual([]);
  });

  it("expands a lane around its title band and children", () => {
    const lane = node("lane", "core.lane", 220, 120, {
      style: { gap: 8 },
    });
    const children = [
      node("a", "core.panel", 160, 64),
      node("b", "core.panel", 160, 64),
    ];

    const layout = resolveCapabilityLayout(lane, children);

    expect(layout.bounds.height).toBe(174);
    expect(layout.childGeometries).toEqual([
      { x: 10, y: 28, width: 160, height: 64 },
      { x: 10, y: 100, width: 160, height: 64 },
    ]);
  });

  it("expands a row rail to fit authored child widths and heights", () => {
    const rail = node("rail", "layout.rail", 160, 22, {
      style: { direction: "row", gap: 8 },
    });
    const children = [
      node("a", "core.chip", 84, 26),
      node("b", "core.chip", 84, 26),
      node("c", "core.chip", 84, 26),
    ];

    const layout = resolveCapabilityLayout(rail, children);

    expect(layout.bounds).toEqual({ x: 0, y: 0, width: 268, height: 26 });
    expect(layout.childGeometries.map((geometry) => geometry.x)).toEqual([
      0, 92, 184,
    ]);
  });

  it("rejects duplicate capability registrations", () => {
    const identity: NativeSceneCapability = {
      capabilityId: "core.group",
      resolveLayout: (value, children) => ({
        bounds: value.geometry!,
        childGeometries: children.map((child) => child.geometry!),
      }),
    };

    expect(() => createCapabilityRegistry([identity, identity])).toThrow(
      /duplicate native Scene capability "core\.group"/i,
    );
  });

  it("resolves semantic circle center and radius into bounds", () => {
    const circle = node("glow", "core.circle", 0, 0, {
      props: { center: { x: 420, y: 165 }, r: 36 },
      style: { r: 36 },
    });

    expect(resolveCapabilityLayout(circle, []).bounds).toEqual({
      x: 384,
      y: 129,
      width: 72,
      height: 72,
    });
  });
});

