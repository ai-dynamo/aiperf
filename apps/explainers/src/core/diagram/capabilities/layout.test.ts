/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { SceneNodeLike } from "../scene-types.js";
import {
  CHIP_PAD_X,
  DETAIL_HEIGHT,
  estimateTextWidth,
  INSET,
  scaledSceneFontSize,
  SCENE_LINE_HEIGHT_RATIO,
  stepperChipWidth,
  SUBTITLE_HEIGHT,
  TITLE_HEIGHT,
  wrapTextToWidth,
} from "../text-metrics.js";
import { LEGEND_DEFINITION } from "../../../flow/sdk/generic/chrome.js";
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

  it("permits absolute overlay children outside local bounds", () => {
    const overlay = node("overlay", "layout.overlay", 80, 40);
    const children = [
      node("badge", "core.chip", 70, 22, {
        geometry: { x: 155, y: 120, width: 70, height: 22 },
        style: { position: "absolute" },
      }),
    ];

    const layout = resolveCapabilityLayout(overlay, children);

    expect(layout.diagnostics).not.toContainEqual(
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

    expect(layout.contentBounds.y).toBe(38);
    expect(layout.childGeometries[0]?.y).toBeGreaterThanOrEqual(38);
  });

  it("reserves frame chrome vertical budget for subtitle", () => {
    const subtitle = "thread-per-core worker sink";
    const frame = node("frame", "layout.frame", 180, 80, {
      props: {
        title: "Worker",
        detail: "One event loop",
        subtitle,
      },
      style: { padding: 6, gap: 8 },
    });

    const layout = resolveCapabilityLayout(frame, [
      node("content", "core.panel", 100, 30),
    ]);
    const subtitleBottom =
      INSET + TITLE_HEIGHT + DETAIL_HEIGHT + 6 + SUBTITLE_HEIGHT;
    const expectedChromeBand = Math.max(52, subtitleBottom);

    expect(layout.contentBounds.y).toBe(expectedChromeBand + 6);
    expect(layout.childGeometries[0]?.y).toBeGreaterThanOrEqual(
      expectedChromeBand + 6,
    );
    expect(layout.bounds.width).toBe(
      Math.max(
        180,
        estimateTextWidth("Worker", 14, "bold") + 12,
        estimateTextWidth("One event loop", 11.5) + 12,
        estimateTextWidth(subtitle, 10) + 12,
      ),
    );
    expect(layout.bounds.height).toBeGreaterThanOrEqual(
      expectedChromeBand + 30 + 12,
    );
  });

  it("grows note chrome to fit its subtitle band", () => {
    const subtitle = "single source of truth for benchmark measurements";
    const note = node("note", "core.note", 100, 56, {
      props: { title: "Remember", detail: "metrics", subtitle },
    });

    const layout = resolveCapabilityLayout(note, []);

    expect(layout.bounds.width).toBe(
      Math.max(100, estimateTextWidth(subtitle, 10) + INSET * 2),
    );
    expect(layout.bounds.height).toBeGreaterThanOrEqual(78);
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

  it("aligns frame chrome text with managed padding", () => {
    const frame = node("frame", "layout.frame", 200, 120, {
      props: { title: "Worker", detail: "One event loop", subtitle: "sink" },
      style: { padding: 20 },
      geometry: { x: 10, y: 15, width: 200, height: 120 },
    });

    const chrome = resolveSemanticChrome(frame, frame.geometry!);

    expect(chrome.texts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "frame__title",
          x: 10 + 20,
          width: 200 - 40,
        }),
        expect.objectContaining({
          id: "frame__detail",
          x: 10 + 20,
          width: 200 - 40,
        }),
        expect.objectContaining({
          id: "frame__subtitle",
          x: 10 + 20,
          width: 200 - 40,
        }),
      ]),
    );
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
      Math.max(84, estimateTextWidth("authoritative", 11, "bold") + CHIP_PAD_X),
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

  it("grows panel chrome to fit its subtitle band and text", () => {
    const subtitle = "single source of truth for benchmark measurements";
    const panel = node("card", "core.panel", 100, 56, {
      props: { title: "Worker", detail: "register", subtitle },
    });

    const layout = resolveCapabilityLayout(panel, []);

    expect(layout.bounds.width).toBe(
      Math.max(100, estimateTextWidth(subtitle, 10) + INSET * 2),
    );
    expect(layout.bounds.height).toBeGreaterThanOrEqual(78);
  });

  it("keeps authored clipped panel bounds despite subtitle content", () => {
    const panel = node("card", "core.panel", 100, 56, {
      props: {
        title: "Worker",
        detail: "register",
        subtitle: "single source of truth for benchmark measurements",
      },
      style: { overflow: "hidden" },
    });

    expect(resolveCapabilityLayout(panel, []).bounds).toEqual(panel.geometry);
  });

  it("grows callouts to fit their labels unless clipping is authored", () => {
    const label = "A long callout label that must remain visible";
    const callout = node("callout", "core.callout", 100, 24, {
      props: { text: label },
    });
    const clipped = { ...callout, style: { clip: true } };

    const layout = resolveCapabilityLayout(callout, []);

    expect(layout.bounds.width).toBe(
      Math.max(100, estimateTextWidth(label, 12) + INSET * 2),
    );
    expect(layout.bounds.height).toBeGreaterThanOrEqual(24);
    expect(resolveCapabilityLayout(clipped, []).bounds).toEqual(callout.geometry);
  });

  it("grows legend factory geometry for scale-aware entry labels", () => {
    const label = "Authoritative endpoint measurements";
    const result = LEGEND_DEFINITION.factory(
      { id: "legend", width: 80, entries: [{ label }] },
      {},
      {
        instanceId: "legend",
        sourceMap: {
          source: "layout.test.flow",
          start: { offset: 0, line: 1, column: 1 },
          end: { offset: 1, line: 1, column: 2 },
        },
        themeTokens: new Map(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      const root = result.value.roots[0]!;
      // LEGEND_SWATCH_SIZE + LEGEND_LABEL_GAP + label width + INSET (all 2.7x-scaled).
      const expectedWidth = 27 + 21.6 + estimateTextWidth(label, 29.7) + INSET;
      expect(root.geometry.width).toBe(expectedWidth);
      expect(
        root.kind === "group" ? root.children[0]?.geometry.width : undefined,
      ).toBe(expectedWidth);
    }
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

  it("wraps text within its authored width, growing height instead of width", () => {
    // A positive authored width (with no `nowrap`) is a wrap constraint, not
    // a minimum to grow past — SceneRenderer wraps this same text to this
    // same width at paint time, so this resolver must NOT grow width to fit
    // the text on one line (that would silently defeat the paint-time wrap
    // and let the text overflow past its box — the exact bug this fixes).
    const text = node("label", "core.text", 80, 10, {
      kind: "text",
      text: "An authoritative long label",
      style: { fontSize: 20 },
    });

    const layout = resolveCapabilityLayout(text, []);
    const scaledFontSize = scaledSceneFontSize(20);
    const expectedLineCount = wrapTextToWidth(text.text!, 80, scaledFontSize, "normal").length;

    expect(layout.bounds.width).toBe(80);
    expect(layout.bounds.height).toBe(
      Math.max(10, expectedLineCount * scaledFontSize * SCENE_LINE_HEIGHT_RATIO),
    );
    expect(layout.bounds.height).toBeGreaterThan(10);
  });

  it("still grows width to fit a single line when the node has no authored width constraint", () => {
    // authored width 0 means "no width constraint, size to content" — the
    // original grow-to-fit-single-line behavior is correct here since there
    // is no box to wrap within.
    const text = node("label", "core.text", 0, 10, {
      kind: "text",
      text: "An authoritative long label",
      style: { fontSize: 20 },
    });

    const layout = resolveCapabilityLayout(text, []);

    expect(layout.bounds.width).toBe(estimateTextWidth(text.text!, 20));
    expect(layout.bounds.height).toBeGreaterThan(10);
  });

  it("still grows width to fit a single line when whiteSpace is nowrap", () => {
    const text = node("label", "core.text", 80, 10, {
      kind: "text",
      text: "An authoritative long label",
      style: { fontSize: 20, whiteSpace: "nowrap" },
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

  it("grows every diagram capability to fit title and detail chrome", () => {
    const title = "Authoritative benchmark database";
    const detail = "records endpoint measurements";
    const capabilityIds = [
      "diagram.actor",
      "diagram.compute",
      "diagram.storage",
      "diagram.messaging",
      "diagram.network",
      "diagram.control",
      "diagram.boundary",
      "diagram.symbol",
    ];

    for (const capabilityId of capabilityIds) {
      const diagram = node("diagram", capabilityId, 100, 40, {
        props: { title, detail },
      });
      const layout = resolveCapabilityLayout(diagram, []);
      const horizontalChrome = capabilityId === "diagram.boundary" ? 24 : 56;

      expect(layout.bounds.width).toBe(
        Math.max(
          100,
          estimateTextWidth(title, capabilityId === "diagram.boundary" ? 12 : 13, "bold") +
            horizontalChrome,
          estimateTextWidth(detail, 10) + horizontalChrome,
        ),
      );
      expect(layout.bounds.height).toBeGreaterThanOrEqual(62);
    }
  });

  it("keeps authored clipped diagram bounds despite long chrome", () => {
    const diagram = node("diagram", "diagram.compute", 100, 40, {
      props: {
        title: "Authoritative benchmark database",
        detail: "records endpoint measurements",
      },
      style: { clip: true },
    });

    expect(resolveCapabilityLayout(diagram, []).bounds).toEqual(diagram.geometry);
  });

  it("grows presentation groups around code, quote, and icon-label text", () => {
    const code = node("code", "core.group", 100, 20, {
      props: {
        presentation: "code-block",
        text: "const authoritative = true;\nreturn authoritative;",
      },
    });
    const quote = node("quote", "core.group", 100, 20, {
      props: {
        presentation: "quote",
        text: "Measure endpoint behavior, not assumptions.",
      },
    });
    const iconLabel = node("icon-label", "core.group", 100, 20, {
      props: {
        presentation: "icon-label",
        label: "Authoritative metrics",
      },
    });

    expect(resolveCapabilityLayout(code, []).bounds).toMatchObject({
      width:
        Math.max(
          estimateTextWidth("const authoritative = true;", 12),
          estimateTextWidth("return authoritative;", 12),
        ) + 24,
    });
    expect(resolveCapabilityLayout(code, []).bounds.height).toBeGreaterThan(20);
    expect(resolveCapabilityLayout(quote, []).bounds.width).toBe(
      estimateTextWidth("Measure endpoint behavior, not assumptions.", 12) + 24,
    );
    expect(resolveCapabilityLayout(quote, []).bounds.height).toBeGreaterThan(20);
    expect(resolveCapabilityLayout(iconLabel, []).bounds).toMatchObject({
      width: estimateTextWidth("Authoritative metrics", 12) + 48,
    });
    expect(resolveCapabilityLayout(iconLabel, []).bounds.height).toBeGreaterThan(20);
  });

  it("keeps authored clipped presentation bounds", () => {
    const code = node("code", "core.group", 100, 20, {
      props: {
        presentation: "code-block",
        text: "const authoritative = true;",
      },
      style: { overflow: "hidden" },
    });

    expect(resolveCapabilityLayout(code, []).bounds).toEqual(code.geometry);
  });

  it("grows avatar presentation to a square icon-safe minimum", () => {
    const defaultSize = node("avatar", "core.group", 48, 48, {
      props: { presentation: "avatar", icon: "user" },
    });
    const undersized = node("avatar", "core.group", 30, 30, {
      props: { presentation: "avatar", icon: "user" },
    });
    const rectangular = node("avatar", "core.group", 100, 48, {
      props: { presentation: "avatar", icon: "user" },
    });
    const hero = node("avatar", "core.group", 130, 130, {
      props: { presentation: "avatar", icon: "user", label: "Mina, on call" },
    });

    expect(resolveCapabilityLayout(defaultSize, []).bounds).toEqual({
      x: 0,
      y: 0,
      width: 48,
      height: 48,
    });
    expect(resolveCapabilityLayout(undersized, []).bounds).toEqual({
      x: 0,
      y: 0,
      width: 40,
      height: 40,
    });
    expect(resolveCapabilityLayout(rectangular, []).bounds).toEqual({
      x: 0,
      y: 0,
      width: 100,
      height: 100,
    });
    expect(resolveCapabilityLayout(hero, []).bounds).toEqual({
      x: 0,
      y: 0,
      width: 130,
      height: 130,
    });
  });

  it("keeps authored clipped avatar bounds despite icon-safe minimum", () => {
    const avatar = node("avatar", "core.group", 30, 30, {
      props: { presentation: "avatar", icon: "user" },
      style: { clip: true },
    });

    expect(resolveCapabilityLayout(avatar, []).bounds).toEqual(avatar.geometry);
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

    expect(layout.bounds.height).toBe(178);
    expect(layout.childGeometries).toEqual([
      { x: 10, y: 32, width: 160, height: 64 },
      { x: 10, y: 104, width: 160, height: 64 },
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

