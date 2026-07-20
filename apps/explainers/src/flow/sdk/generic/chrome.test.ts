/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { expandSdkInvocation } from "../expand.js";
import { createSdkRegistry } from "../registry.js";
import type { SdkExpansionContext } from "../types.js";

const SOURCE_MAP = {
  source: "chrome.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function context(instanceId: string): SdkExpansionContext {
  return {
    instanceId,
    sourceMap: SOURCE_MAP,
    themeTokens: new Map(),
  };
}

describe("sdk.note chrome factory", () => {
  it("accepts strokeWidth through expand validation (STRICT_UNKNOWN_PROP)", () => {
    const definition = createSdkRegistry().lookup("sdk.note")!;
    const result = expandSdkInvocation(
      definition,
      { id: "note", text: "Hello", strokeWidth: 2 },
      {},
      context("note"),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root?.style?.strokeWidth).toBe(2);
    }
  });

  it("applies the declared inkRole default to caption text on the semantic root", () => {
    const definition = createSdkRegistry().lookup("sdk.note")!;
    const result = definition.factory(
      { id: "note", text: "Hello" },
      {},
      context("note"),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0];
      expect(root).toMatchObject({
        kind: "group",
        props: {
          text: "Hello",
          inkRole: "@theme.ink.secondary",
        },
      });
    }
  });

  it("honors an authored inkRole on the semantic root", () => {
    const definition = createSdkRegistry().lookup("sdk.note")!;
    const result = definition.factory(
      { id: "note", text: "Hello", inkRole: "@theme.ink.primary" },
      {},
      context("note"),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]).toMatchObject({
        props: { text: "Hello", inkRole: "@theme.ink.primary" },
      });
    }
  });
});

describe("sdk.bracket chrome factory", () => {
  it.each([
    { side: "left", start: "ne", end: "se" },
    { side: "right", start: "nw", end: "sw" },
    { side: "top", start: "sw", end: "se" },
    { side: "bottom", start: "nw", end: "ne" },
  ] as const)(
    "exposes start/end port anchors matching bracePath endpoints for side=$side",
    ({ side, start, end }) => {
      const definition = createSdkRegistry().lookup("sdk.bracket")!;
      const result = definition.factory(
        { id: "bracket", side },
        {},
        context("bracket"),
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.value.ports.start).toEqual({
          nodeId: "bracket",
          anchor: start,
        });
        expect(result.value.ports.end).toEqual({
          nodeId: "bracket",
          anchor: end,
        });
      }
    },
  );
});

describe("sdk chrome factory default geometry floors", () => {
  const registry = createSdkRegistry();

  it.each([
    ["sdk.header", { id: "hdr", title: "Title" }, { height: 66 }],
    ["sdk.panel", { id: "pnl", title: "Title", detail: "Detail" }, { height: 70 }],
    ["sdk.card", { id: "crd", title: "Title", detail: "Detail", subtitle: "Sub" }, { height: 88 }],
    ["sdk.note", { id: "nt", text: "Note" }, { height: 48 }],
    ["sdk.label", { id: "lbl", text: "Label" }, { height: 22 }],
    ["sdk.callout", { id: "co", text: "Callout" }, { height: 48 }],
  ] as const)(
    "%s uses raised default geometry when height is omitted",
    (componentId, props, expected) => {
      const result = registry.lookup(componentId)!.factory(props, {}, context(props.id));

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.value.roots[0]?.geometry).toMatchObject(expected);
      }
    },
  );

  it.each([
    ["compact", 88],
    ["standard", 88],
    ["wide", 88],
  ] as const)("sdk.card size preset %s defaults to height %i", (size, height) => {
    const result = registry.lookup("sdk.card")!.factory(
      { id: "card", title: "Title", detail: "Detail", subtitle: "Sub", size },
      {},
      context("card"),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.roots[0]?.geometry.height).toBe(height);
    }
  });
});
