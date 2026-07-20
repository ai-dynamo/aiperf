/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { SceneNodeLike } from "../scene-types.js";
import {
  estimateTextWidth,
  stepperChipWidth,
} from "../text-metrics.js";
import {
  createCapabilityRegistry,
  resolveCapabilityLayout,
} from "./registry.js";
import {
  hasNativeSemanticChrome,
  resolveSemanticChrome,
} from "./chrome.js";
import { managedLayoutOptions } from "./layout.js";
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
  it("normalizes shared managed layout defaults and compatibility input", () => {
    const defaults = managedLayoutOptions(node("stack", "layout.stack", 0, 0));
    const compatible = managedLayoutOptions(
      node("stack", "layout.stack", 0, 0, {
        style: {
          padding: -12,
          align: "center",
          justify: "space-between",
          fixedWidth: true,
          fixedHeight: true,
        },
      }),
    );

    expect(defaults).toEqual({
      padding: 0,
      align: "start",
      justify: "start",
      fixedWidth: false,
      fixedHeight: false,
    });
    expect(compatible).toEqual({
      padding: 0,
      align: "center",
      justify: "space-between",
      fixedWidth: true,
      fixedHeight: true,
    });
  });

  it("places stack children inside padded stretched content bounds", () => {
    const stack = node("stack", "layout.stack", 140, 96, {
      geometry: { x: 20, y: 30, width: 140, height: 96 },
      style: {
        direction: "column",
        gap: 8,
        padding: 12,
        align: "stretch",
      },
    });
    const children = [
      node("one", "core.panel", 100, 30),
      node("two", "core.panel", 100, 30),
    ];

    expect(resolveCapabilityLayout(stack, children)).toMatchObject({
      bounds: { x: 20, y: 30, width: 140, height: 96 },
      contentBounds: { x: 12, y: 12, width: 116, height: 72 },
      childGeometries: [
        { x: 12, y: 12, width: 116, height: 30 },
        { x: 12, y: 50, width: 116, height: 30 },
      ],
    });
  });

  it("keeps absolute managed children out of normal flow and reports overlap", () => {
    const stack = node("stack", "layout.stack", 120, 80, {
      style: { direction: "column", gap: 8, padding: 8 },
    });
    const children = [
      node("flow", "core.panel", 80, 30),
      node("absolute", "core.panel", 80, 30, {
        geometry: { x: 8, y: 8, width: 80, height: 30 },
        style: { position: "absolute" },
      }),
    ];

    const layout = resolveCapabilityLayout(stack, children);

    expect(layout.childGeometries).toEqual([
      { x: 8, y: 8, width: 80, height: 30 },
      { x: 8, y: 8, width: 80, height: 30 },
    ]);
    expect(layout.diagnostics).toContainEqual(
      expect.objectContaining({ code: "SCENE_MANAGED_CHILD_OVERLAP" }),
    );
  });

  it("reports fixed managed content overflow", () => {
    const stack = node("stack", "layout.stack", 60, 60, {
      style: {
        direction: "column",
        padding: 10,
        fixedWidth: true,
      },
    });

    const layout = resolveCapabilityLayout(stack, [
      node("wide", "core.panel", 80, 20),
    ]);

    expect(layout.bounds.width).toBe(60);
    expect(layout.diagnostics).toContainEqual(
      expect.objectContaining({ code: "SCENE_MANAGED_CONTENT_OVERFLOW" }),
    );
  });

  it("intentionally overlays aligned children without overlap diagnostics", () => {
    const overlay = node("overlay", "layout.overlay", 80, 40, {
      style: { align: "stretch" },
    });
    const children = [
      node("a", "core.panel", 20, 10),
      node("b", "core.panel", 30, 20),
    ];

    const layout = resolveCapabilityLayout(overlay, children);

    expect(layout.childGeometries).toEqual([
      { x: 0, y: 0, width: 80, height: 40 },
      { x: 0, y: 0, width: 80, height: 40 },
    ]);
    expect(layout.diagnostics).not.toContainEqual(
      expect.objectContaining({ code: "SCENE_MANAGED_CHILD_OVERLAP" }),
    );
  });

  it("reserves a title-safe frame content band", () => {
    const frame = node("frame", "layout.frame", 180, 100, {
      props: { title: "Worker" },
      style: { padding: 6, gap: 8 },
    });

    const layout = resolveCapabilityLayout(frame, [
      node("content", "core.panel", 100, 30),
    ]);

    expect(layout.contentBounds.y).toBe(34);
    expect(layout.childGeometries[0]?.y).toBeGreaterThanOrEqual(34);
  });

  it("resolves one semantic frame box with title and detail parts", () => {
    const frame = node("frame", "layout.frame", 180, 100, {
      props: { title: "Worker", detail: "One event loop" },
    });

    expect(hasNativeSemanticChrome(frame)).toBe(true);
    expect(
      resolveSemanticChrome(frame, frame.geometry!),
    ).toMatchObject({
      rootBox: { id: "frame__chrome", geometry: frame.geometry },
      texts: [
        { id: "frame__title", text: "Worker" },
        { id: "frame__detail", text: "One event loop" },
      ],
    });
  });

  it("expands a semantic stepper using scale-aware chip widths", () => {
    const stepper = node("steps", "core.stepper", 160, 90, {
      props: { steps: ["layout", "slots", "timeline"], linked: true },
      style: { gap: 16 },
    });

    const layout = resolveCapabilityLayout(stepper, []);
    const expected =
      stepperChipWidth("layout", 0) +
      stepperChipWidth("slots", 1) +
      stepperChipWidth("timeline", 2) +
      16 * 2;

    expect(layout.bounds).toEqual({ x: 0, y: 0, width: expected, height: 90 });
    expect(layout.childGeometries).toEqual([]);
  });

  it("grows a chip to fit its label while treating authored size as a minimum", () => {
    const short = node("chip", "core.chip", 84, 26, {
      props: { label: "ok" },
    });
    const long = node("chip", "core.chip", 84, 26, {
      props: { label: "authoritative" },
    });

    expect(resolveCapabilityLayout(short, []).bounds.width).toBe(84);
    expect(resolveCapabilityLayout(long, []).bounds.width).toBeGreaterThan(84);
    expect(resolveCapabilityLayout(long, []).bounds.width).toBe(
      Math.max(84, estimateTextWidth("authoritative", 11, "bold") + 24),
    );
  });

  it("grows panel and note chrome to fit title and detail bands", () => {
    const panel = node("panel", "core.panel", 100, 40, {
      props: { title: "Profile source panel", detail: "authoritative metrics" },
    });
    const note = node("note", "core.note", 100, 40, {
      props: { text: "Remember that authoritative metrics come from the endpoint" },
    });

    const panelLayout = resolveCapabilityLayout(panel, []);
    const noteLayout = resolveCapabilityLayout(note, []);

    expect(panelLayout.bounds.width).toBeGreaterThan(100);
    expect(panelLayout.bounds.height).toBeGreaterThanOrEqual(40);
    expect(noteLayout.bounds.width).toBeGreaterThan(100);
    expect(noteLayout.bounds.height).toBeGreaterThanOrEqual(40);
  });

  it("grows header chrome to fit its title and caption", () => {
    const header = node("header", "core.header", 100, 20, {
      props: {
        title: "Authoritative benchmark results",
        caption: "Generated from endpoint metrics",
      },
    });

    const layout = resolveCapabilityLayout(header, []);

    expect(layout.bounds.width).toBeGreaterThan(100);
    expect(layout.bounds.height).toBeGreaterThan(20);
  });

  it("grows text to fit node text at the authored font size", () => {
    const text = node("label", "core.text", 80, 10, {
      kind: "text",
      text: "An authoritative long label",
      style: { fontSize: 20 },
    });

    const layout = resolveCapabilityLayout(text, []);

    expect(layout.bounds.width).toBe(
      Math.max(80, estimateTextWidth(text.text!, 20)),
    );
    expect(layout.bounds.height).toBeGreaterThan(10);
  });

  it("resolves presentation chrome for code, quote, avatar, and icon-label", () => {
    const code = node("code", "core.group", 200, 80, {
      props: {
        presentation: "code-block",
        text: "const n = 1;",
        inkRole: "@theme.ink.primary",
      },
      style: { radius: 6 },
    });
    const quote = node("quote", "core.group", 200, 60, {
      props: {
        presentation: "quote",
        text: "Measure twice.",
        inkRole: "ink.secondary",
      },
    });
    const avatar = node("avatar", "core.group", 48, 48, {
      props: { presentation: "avatar", icon: "user" },
      style: { radius: 48 },
    });
    const iconLabel = node("labeled", "core.group", 160, 32, {
      props: {
        presentation: "icon-label",
        label: "Ready",
        inkRole: "@theme.ink.primary",
      },
    });

    expect(hasNativeSemanticChrome(code)).toBe(true);
    expect(resolveSemanticChrome(code, code.geometry!).texts[0]).toMatchObject({
      text: "const n = 1;",
      fontFamily: "monospace",
      inkRole: "@theme.ink.primary",
    });
    expect(resolveSemanticChrome(quote, quote.geometry!).texts[0]).toMatchObject({
      text: "Measure twice.",
      fontStyle: "italic",
      inkRole: "ink.secondary",
    });
    expect(resolveSemanticChrome(avatar, avatar.geometry!).rootBox?.radius).toBe(48);
    expect(resolveSemanticChrome(iconLabel, iconLabel.geometry!).texts[0]).toMatchObject({
      text: "Ready",
      x: 40,
    });
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

  it("reflows a rail after chips auto-grow, preserving gap", () => {
    const rail = node("rail", "layout.rail", 160, 22, {
      style: { direction: "row", gap: 8 },
    });
    const children = [
      node("a", "core.chip", 84, 26, {
        props: { label: "authoritative" },
      }),
      node("b", "core.chip", 84, 26, { props: { label: "ok" } }),
    ];
    const a = resolveCapabilityLayout(children[0]!, []).bounds;
    const b = resolveCapabilityLayout(children[1]!, []).bounds;
    const layout = resolveCapabilityLayout(rail, [
      { ...children[0]!, geometry: a },
      { ...children[1]!, geometry: b },
    ]);

    expect(layout.childGeometries[1]?.x).toBe(a.width + 8);
    expect(layout.bounds.width).toBe(a.width + 8 + b.width);
  });

  it("rejects duplicate capability registrations", () => {
    const identity: NativeSceneCapability = {
      capabilityId: "core.group",
      resolveLayout: (value, children) => ({
        bounds: value.geometry!,
        contentBounds: value.geometry!,
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

