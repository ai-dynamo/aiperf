// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { resolveCapabilityId } from "../src/capability-id.js";
import type { ComponentNodeIr, RectNodeIr } from "../src/ir.js";

const sourceMap = {
  source: "test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

describe("resolveCapabilityId", () => {
  test("defaults foundation rect nodes to core.rect", () => {
    const node: RectNodeIr = {
      kind: "rect",
      id: "box",
      geometry: { x: 0, y: 0, width: 10, height: 10 },
      style: {},
      accessibility: { label: "Box" },
      fallback: "Box",
      sourceMap,
    };

    expect(resolveCapabilityId(node)).toBe("core.rect");
  });

  test("uses explicit capabilityId when authored", () => {
    const node: RectNodeIr = {
      kind: "rect",
      id: "box",
      capabilityId: "core.span-map",
      geometry: { x: 0, y: 0, width: 10, height: 10 },
      style: {},
      accessibility: { label: "Span map" },
      fallback: "Span map",
      sourceMap,
    };

    expect(resolveCapabilityId(node)).toBe("core.span-map");
  });

  test("falls back to authored capability alias", () => {
    const node: RectNodeIr = {
      kind: "rect",
      id: "box",
      capability: "core.rect",
      geometry: { x: 0, y: 0, width: 10, height: 10 },
      style: {},
      accessibility: { label: "Box" },
      fallback: "Box",
      sourceMap,
    };

    expect(resolveCapabilityId(node)).toBe("core.rect");
  });

  test("component nodes resolve to their capability id", () => {
    const node: ComponentNodeIr = {
      kind: "component",
      id: "tok-map",
      capabilityId: "core.span-map",
      props: { requireCover: "source" },
      children: [],
      geometry: { x: 0, y: 0, width: 100, height: 40 },
      style: {},
      accessibility: { label: "Token span map" },
      fallback: "Token span map",
      sourceMap,
    };

    expect(resolveCapabilityId(node)).toBe("core.span-map");
  });
});
