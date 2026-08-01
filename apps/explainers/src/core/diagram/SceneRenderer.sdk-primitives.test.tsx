/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SceneRenderer,
  fanSegmentFromAtomic,
  shortenPathForArrowhead,
  splitFanTrajectoryAtJunction,
} from "./SceneRenderer";
import { resolveScene } from "./resolution/resolve-scene.js";
import type { SceneIrLike } from "./scene-types.js";
import {
  DEFAULT_SCENE_FONT_SIZE,
  SCENE_FONT,
  SCENE_LINE_HEIGHT_RATIO,
  SCENE_TEXT_SCALE,
  estimateTextWidth,
  stepperChipWidth,
} from "./text-metrics.js";

afterEach(cleanup);

describe("SceneRenderer SDK foundations", () => {
  it("paints authored core.path glyphs when they are excluded from connector resolution", () => {
    const scene: SceneIrLike = {
      id: "icon-glyph-scene",
      roots: [
        {
          id: "check-icon",
          kind: "connector",
          capabilityId: "core.path",
          geometry: { x: 40, y: 50, width: 24, height: 24 },
          style: {
            fill: "none",
            stroke: "#76b900",
            strokeWidth: 1.75,
            markerEnd: "none",
          },
          path: "M4 12 L9 17 L20 5",
          accessibility: { label: "check" },
        },
      ],
      timeline: [],
    };

    // Glyph icons reuse path IR but are intentionally excluded from routed
    // connector resolution — the renderer must still paint authored `path`.
    expect(resolveScene(scene).connectorsById.has("check-icon")).toBe(false);

    const { container } = render(
      <SceneRenderer scene={scene} playing={false} restartKey={0} />,
    );

    const node = container.querySelector('[data-flow-node-id="check-icon"]');
    const path = node?.querySelector("path");
    expect(path).not.toBeNull();
    expect(path?.getAttribute("d")).toContain("M");
    // Local glyph coordinates are translated by the node geometry origin.
    const translated = [...(node?.querySelectorAll("g[transform]") ?? [])].some(
      (group) => {
        const transform = group.getAttribute("transform") ?? "";
        return /translate\(\s*40[\s,]+50\s*\)/.test(transform);
      },
    );
    const d = path?.getAttribute("d") ?? "";
    const pathOffset =
      d.includes("M44") || d.includes("M 44") || /M\s*44[\s,]/.test(d);
    expect(translated || pathOffset).toBe(true);
  });

  it("prefers authored cloud glyph path over dummy from/to fallback endpoints", () => {
    // sdk.IconLabel stamps local from/to (0,0)-(24,24) beside the glyph path.
    // SceneRenderer must paint the authored cloud outline, not the diagonal
    // connector fallback, when core.path is excluded from connector resolution.
    const cloudPath =
      "M7 19 H18 A4 4 0 0 0 18 11 A6 6 0 0 0 6.5 9 A5 5 0 0 0 4.5 15";
    const scene: SceneIrLike = {
      id: "cloud-glyph-scene",
      roots: [
        {
          id: "il-v3__icon",
          kind: "connector",
          capabilityId: "core.path",
          geometry: { x: 8, y: 8, width: 24, height: 24 },
          style: {
            fill: "none",
            stroke: "#76b900",
            strokeWidth: 1.75,
            markerEnd: "none",
          },
          from: { x: 0, y: 0 },
          to: { x: 24, y: 24 },
          path: cloudPath,
          accessibility: { label: "cloud" },
        },
      ],
      timeline: [],
    };

    expect(resolveScene(scene).connectorsById.has("il-v3__icon")).toBe(false);

    const { container } = render(
      <SceneRenderer scene={scene} playing={false} restartKey={0} />,
    );

    const d =
      container
        .querySelector('[data-flow-node-id="il-v3__icon"] path')
        ?.getAttribute("d") ?? "";
    expect(d).toContain("A4 4 0 0 0 18 11");
    expect(d).not.toMatch(/^M\s*0[\s,]+\s*0\s+L\s*24[\s,]+\s*24/);
  });

  it("renders core.image component nodes as SVG images", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          id: "image-scene",
          roots: [
            {
              id: "hero",
              kind: "component",
              capabilityId: "core.image",
              geometry: { x: 10, y: 20, width: 160, height: 90 },
              style: {},
              props: { src: "/hero.svg", fit: "cover" },
              accessibility: { label: "Hero image" },
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const image = container.querySelector("image");
    expect(image?.getAttribute("href")).toBe("/hero.svg");
    expect(image?.getAttribute("preserveAspectRatio")).toBe("xMidYMid slice");
  });

  it("renders multiline text as individually positioned SVG lines", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "code",
              kind: "text",
              capabilityId: "core.text",
              geometry: { x: 10, y: 20, width: 200, height: 80 },
              style: { whiteSpace: "pre", lineHeight: 18, fontFamily: "monospace" },
              text: "first()\nsecond()",
              accessibility: { label: "Code sample" },
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const lines = container.querySelectorAll("text tspan");
    expect([...lines].map((line) => line.textContent)).toEqual(["first()", "second()"]);
    expect(lines[1]?.getAttribute("dy")).toBe("18");
  });

  it("clips children when a group requests hidden overflow", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "viewport",
              kind: "group",
              capabilityId: "core.group",
              geometry: { x: 10, y: 20, width: 100, height: 60 },
              style: { coordinateSpace: "local", overflow: "hidden" },
              accessibility: { label: "Clipped viewport" },
              children: [
                {
                  id: "overflowing",
                  kind: "rect",
                  capabilityId: "core.rect",
                  geometry: { x: 80, y: 0, width: 80, height: 40 },
                  style: {},
                },
              ],
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const clipPath = container.querySelector("clipPath");
    const clippedGroup = container.querySelector("g[clip-path]");
    expect(clipPath?.querySelector("rect")?.getAttribute("width")).toBe("100");
    expect(clippedGroup?.getAttribute("clip-path")).toMatch(/^url\(#.+\)$/);
  });

  it("renders semantic panel chrome and copy without generated IR children", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "profile",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 20, y: 30, width: 160, height: 64 },
              style: {},
              props: { title: "Profile", detail: "source" },
              accessibility: { label: "Profile source" },
              children: [],
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    expect(
      container.querySelector('[data-flow-semantic-chrome="core.panel"]'),
    ).not.toBeNull();
    expect(
      [...container.querySelectorAll('[data-flow-semantic-text="core.panel"]')].map(
        (node) => node.textContent,
      ),
    ).toEqual(["Profile", "source"]);
  });

  it("paints only generated parts retained by canonical scene resolution", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "panel",
          kind: "group",
          capabilityId: "core.panel",
          geometry: { x: 20, y: 30, width: 180, height: 70 },
          props: { title: "Resolved title" },
          children: [
            {
              id: "panel__title",
              kind: "text",
              capabilityId: "core.text",
              // Wide enough that this text stays on one line (this test
              // checks duplicate-paint-ownership resolution, not wrapping —
              // a narrower width would now legitimately wrap across
              // multiple <tspan>s per the wrap-respecting layout fix, and
              // exact-string textContent equality wouldn't hold post-wrap).
              geometry: { x: 8, y: 8, width: 702, height: 22 },
              text: "Authored compatibility title",
            },
          ],
        },
      ],
      timeline: [],
    };
    expect(
      resolveScene(scene).diagnostics,
    ).toContainEqual(
      expect.objectContaining({ code: "SCENE_DUPLICATE_PAINT_OWNER" }),
    );

    const { container } = render(
      <SceneRenderer scene={scene} playing={false} restartKey={0} />,
    );

    expect(
      [...container.querySelectorAll("text")].filter(
        (node) => node.textContent === "Resolved title",
      ),
    ).toHaveLength(0);
    expect(
      [...container.querySelectorAll("text")].filter(
        (node) => node.textContent === "Authored compatibility title",
      ),
    ).toHaveLength(1);
  });

  it("paints one copy for every native semantic chrome family", () => {
    const cases = [
      ["header", "core.header", { title: "Header copy" }],
      ["panel", "core.panel", { title: "Panel copy" }],
      ["chip", "core.chip", { label: "Chip copy" }],
      ["note", "core.note", { text: "Note copy" }],
      ["lane", "core.lane", { title: "Lane copy" }],
      ["band", "core.band", { title: "Band copy" }],
    ] as const;
    const scene: SceneIrLike = {
      viewport: { width: 900, height: 700 },
      roots: [
        ...cases.map(([id, capabilityId, props], index) => ({
          id,
          kind: "group",
          capabilityId,
          geometry: { x: 20, y: 20 + index * 70, width: 200, height: 54 },
          props,
          children: [],
        })),
        {
          id: "stepper",
          kind: "group",
          capabilityId: "core.stepper",
          geometry: { x: 20, y: 460, width: 300, height: 26 },
          props: { steps: ["Stepper copy"] },
          children: [
            {
              id: "stepper-step-0",
              kind: "group",
              capabilityId: "core.chip",
              geometry: { x: 0, y: 0, width: 100, height: 26 },
              props: { label: "1. Stepper copy", index: 0 },
              children: [],
            },
          ],
        },
      ],
      timeline: [],
    };

    const { container } = render(
      <SceneRenderer scene={scene} playing={false} restartKey={0} />,
    );

    // Chrome copy wraps to the part's box width, so a <text> may hold several
    // <tspan> lines; `textContent` would concatenate them without the break's
    // space. Compare the rejoined copy so the assertion tracks what is painted
    // rather than how many lines it happened to need.
    const renderedCopy = (node: Element): string =>
      [...node.childNodes]
        .map((child) => child.textContent ?? "")
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();

    for (const copy of [
      "Header copy",
      "Panel copy",
      "Chip copy",
      "Note copy",
      "Lane copy",
      "Band copy",
      "1. Stepper copy",
    ]) {
      expect(
        [...container.querySelectorAll("text")].filter(
          (node) => renderedCopy(node) === copy,
        ),
        copy,
      ).toHaveLength(1);
    }
  });

  it("keeps a connector hidden until all node-backed endpoints appear", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "source",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 20, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Source" },
            },
            {
              id: "target",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 300, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Target" },
            },
            {
              id: "edge",
              kind: "connector",
              capabilityId: "core.route",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              style: {},
              from: { nodeId: "source", anchor: "e" },
              to: { nodeId: "target", anchor: "w" },
            },
            {
              id: "fan",
              kind: "fan",
              capabilityId: "core.fan-out",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              style: {},
              from: { nodeId: "source", anchor: "e" },
              to: [
                { nodeId: "target", anchor: "w" },
                { x: 500, y: 160 },
              ],
            },
          ],
          timeline: [
            {
              id: "edge-trace",
              at: -1000,
              duration: 100,
              action: "trace",
              target: "edge",
            },
            {
              id: "source-reveal",
              at: 1000,
              duration: 200,
              action: "reveal",
              target: "source",
            },
            {
              id: "target-reveal",
              at: 1200,
              duration: 200,
              action: "reveal",
              target: "target",
            },
          ],
        }}
        playing
        restartKey={0}
      />,
    );

    expect(
      container.querySelector<SVGGElement>('[data-flow-node-id="edge"]')?.style
        .opacity,
    ).toBe("0");
    expect(
      container.querySelector<SVGGElement>('[data-flow-node-id="fan"]')?.style
        .opacity,
    ).toBe("0");
  });

  it("renders canonical resolved connector paths and marker policy", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "source",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 20, y: 40, width: 80, height: 50 },
          children: [],
        },
        {
          id: "target",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 260, y: 40, width: 80, height: 50 },
          children: [],
        },
        {
          id: "edge",
          kind: "connector",
          capabilityId: "core.connector",
          geometry: { x: 0, y: 0, width: 0, height: 0 },
          style: { route: "curve" },
          from: { nodeId: "source", anchor: "e" },
          to: { nodeId: "target", anchor: "w" },
        },
        {
          id: "guide",
          kind: "connector",
          capabilityId: "core.line",
          geometry: { x: 0, y: 0, width: 0, height: 0 },
          style: { arrowhead: false },
          from: { x: 20, y: 120 },
          to: { x: 340, y: 120 },
        },
      ],
      timeline: [],
    };
    const resolved = resolveScene(scene);
    const { container } = render(
      <SceneRenderer
        scene={scene}
        playing={false}
        restartKey={0}
      />,
    );

    const edgePath = container.querySelector(
      '[data-flow-node-id="edge"] path[data-flow-arrowhead]',
    );
    const guidePath = container.querySelector(
      '[data-flow-node-id="guide"] path[data-flow-arrowhead]',
    );
    const resolvedEdge = resolved.connectorsById.get("edge")?.d;
    expect(resolvedEdge).toBeDefined();
    // Directed edges reserve tip length so the marker does not overshoot.
    expect(edgePath?.getAttribute("d")).toBeDefined();
    expect(edgePath?.getAttribute("d")).not.toBe(resolvedEdge);
    expect(edgePath?.getAttribute("d")?.startsWith("M100 65")).toBe(true);
    expect(edgePath?.getAttribute("data-flow-resolved-path")).toBe(resolvedEdge);
    expect(edgePath?.getAttribute("data-flow-arrowhead")).toBe("true");
    expect(guidePath?.getAttribute("d")).toBe(
      resolved.connectorsById.get("guide")?.d,
    );
    expect(guidePath?.getAttribute("data-flow-resolved-path")).toBe(
      resolved.connectorsById.get("guide")?.d,
    );
    expect(guidePath?.getAttribute("data-flow-arrowhead")).toBe("false");
  });

  it("renders edge-bound motion signals from the resolved connector path", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "source",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 20, y: 40, width: 80, height: 50 },
          children: [],
        },
        {
          id: "target",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 260, y: 40, width: 80, height: 50 },
          children: [],
        },
        {
          id: "credit",
          kind: "connector",
          capabilityId: "core.connector",
          geometry: { x: 0, y: 0, width: 0, height: 0 },
          style: {},
          from: { nodeId: "source", anchor: "e" },
          to: { nodeId: "target", anchor: "w" },
        },
        {
          id: "motion",
          kind: "connector",
          capabilityId: "motion.signal",
          geometry: { x: 0, y: 0, width: 0, height: 0 },
          edgeRef: "credit",
          style: {},
        },
      ],
      timeline: [
        {
          id: "reveal-source",
          at: 0,
          duration: 200,
          action: "reveal",
          target: "source",
        },
        {
          id: "reveal-target",
          at: 0,
          duration: 200,
          action: "reveal",
          target: "target",
        },
        {
          id: "trace-motion",
          at: 0,
          duration: 800,
          action: "trace",
          target: "motion",
        },
      ],
    };
    const resolved = resolveScene(scene);
    const { container } = render(
      <SceneRenderer scene={scene} playing={false} restartKey={0} />,
    );

    expect(resolved.connectorsById.get("motion")?.d).toBe(
      resolved.connectorsById.get("credit")?.d,
    );
    expect(resolved.connectorsById.get("motion")?.showArrowhead).toBe(false);
    expect(
      container.querySelector('[data-flow-node-id="motion"] animateMotion'),
    ).not.toBeNull();
  });

  it("keeps a motion signal hidden until its endpoint nodes appear", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "source",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 20, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Source" },
            },
            {
              id: "target",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 300, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Target" },
            },
            {
              id: "signal",
              kind: "motion",
              capabilityId: "motion.signal",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              style: {},
              from: { nodeId: "source", anchor: "e" },
              to: { nodeId: "target", anchor: "w" },
            },
          ],
          timeline: [
            {
              id: "source-reveal",
              at: 1000,
              duration: 200,
              action: "reveal",
              target: "source",
            },
            {
              id: "target-reveal",
              at: 1200,
              duration: 200,
              action: "reveal",
              target: "target",
            },
          ],
        }}
        playing
        restartKey={0}
      />,
    );

    expect(
      container.querySelector<SVGGElement>('[data-flow-node-id="signal"]')?.style
        .opacity,
    ).toBe("0");
  });

  it("scales every scene text font to ninety percent", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "profile",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 20, y: 30, width: 160, height: 64 },
              style: {},
              props: { title: "Profile", detail: "source" },
              accessibility: { label: "Profile source" },
              children: [],
            },
            {
              id: "authored",
              kind: "text",
              capabilityId: "core.text",
              geometry: { x: 20, y: 110, width: 160, height: 30 },
              style: { fontSize: 20 },
              text: "Authored",
            },
            {
              id: "default",
              kind: "text",
              capabilityId: "core.text",
              geometry: { x: 20, y: 150, width: 160, height: 30 },
              style: {},
              text: "Default",
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const textByContent = (content: string) =>
      [...container.querySelectorAll("text")].find(
        (node) => node.textContent === content,
      );

    // A renderer-owned panel title and an unstyled `core.text` must land on the
    // same baseline: both are scene-world text the author never sized. They
    // diverged (12.6 vs 34.02) while `chrome.ts` still held pre-4K-rescale font
    // sizes, which is what left titles tiny inside 2.7x-padded boxes.
    expect(textByContent("Profile")?.getAttribute("font-size")).toBe(
      String(DEFAULT_SCENE_FONT_SIZE * SCENE_TEXT_SCALE),
    );
    expect(textByContent("Default")?.getAttribute("font-size")).toBe(
      String(DEFAULT_SCENE_FONT_SIZE * SCENE_TEXT_SCALE),
    );
    expect(textByContent("source")?.getAttribute("font-size")).toBe(
      String(SCENE_FONT.detail * SCENE_TEXT_SCALE),
    );
    // An authored `fontSize` still wins over the chrome ladder.
    expect(textByContent("Authored")?.getAttribute("font-size")).toBe("18");
    expect(textByContent("Authored")?.style.fontSize).toBe("18px");
  });

  it("wraps semantic chrome copy to its box instead of overflowing", () => {
    const detail =
      "batching and KV cache and prefix caching and chunked prefill and routing and disagg handoff";
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "panel",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 0, y: 0, width: 320, height: 240 },
              style: {},
              props: { title: "Unchanged inside the seam", detail },
              children: [],
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const detailText = [...container.querySelectorAll("text")].find((node) =>
      (node.textContent ?? "").startsWith("batching"),
    );
    const lines = [...(detailText?.querySelectorAll("tspan") ?? [])];

    // SVG <text> does not wrap, so long copy used to run straight off the box.
    expect(lines.length).toBeGreaterThan(1);
    expect(lines.map((line) => line.textContent).join(" ")).toBe(detail);

    // Every line has to fit the box it is painted into. The authored 320 is not
    // the bound to check: `layout.ts` auto-grows a panel to fit its copy, so the
    // invariant is against the box as actually resolved.
    const fontSize = SCENE_FONT.detail * SCENE_TEXT_SCALE;
    const chromeRect = container.querySelector("rect[data-flow-semantic-chrome]");
    const boxWidth = Number(chromeRect?.getAttribute("width"));
    expect(boxWidth).toBeGreaterThan(0);
    for (const line of lines) {
      expect(
        estimateTextWidth(line.textContent ?? "", fontSize),
      ).toBeLessThanOrEqual(boxWidth);
    }

    // A centered block centers on its full stacked height, so the first line
    // sits above the midpoint rather than on it.
    expect(Number(detailText?.getAttribute("y"))).toBeLessThan(
      Number(detailText?.getAttribute("y")) +
        ((lines.length - 1) * fontSize * SCENE_LINE_HEIGHT_RATIO) / 2,
    );
  });

  it("renders intrinsically sized semantic stepper labels", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "steps",
          kind: "group",
          capabilityId: "core.stepper",
          geometry: { x: 10, y: 20, width: 160, height: 90 },
          style: { gap: 16 },
          props: {
            steps: ["layout", "slots", "timeline"],
            linked: true,
          },
          accessibility: { label: "Layout steps" },
          children: [],
        },
      ],
      timeline: [],
    };
    expect(
      [...resolveScene(scene).generatedPartsById.values()]
        .filter(({ ownerId, kind }) => ownerId === "steps" && kind === "text")
        .map(({ id }) => id),
    ).toEqual([
      "steps-step-0__label",
      "steps-step-1__label",
      "steps-step-2__label",
    ]);
    const { container } = render(
      <SceneRenderer
        scene={scene}
        playing={false}
        restartKey={0}
      />,
    );

    expect(
      [...container.querySelectorAll('[data-flow-semantic-text="core.stepper"]')].map(
        (node) => node.textContent,
      ),
    ).toEqual(["1. layout", "2. slots", "3. timeline"]);
    expect(
      container
        .querySelector('[data-flow-semantic-chrome="core.stepper"]')
        ?.getAttribute("width"),
    ).toBe(String(stepperChipWidth("layout", 0)));
  });

  it("reflows rail children using intrinsic chip widths", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "rail",
              kind: "group",
              capabilityId: "layout.rail",
              geometry: { x: 10, y: 20, width: 160, height: 22 },
              style: { direction: "row", gap: 8 },
              children: [
                {
                  id: "long",
                  kind: "group",
                  capabilityId: "core.chip",
                  geometry: { x: 0, y: 0, width: 84, height: 26 },
                  style: {},
                  props: { label: "authoritative" },
                },
                {
                  id: "short",
                  kind: "group",
                  capabilityId: "core.chip",
                  geometry: { x: 0, y: 0, width: 84, height: 26 },
                  style: {},
                  props: { label: "ok" },
                },
              ],
            },
          ],
          timeline: [],
        }}
        playing={false}
        restartKey={0}
      />,
    );

    const longWidth = Math.max(
      84,
      estimateTextWidth("authoritative", 11, "bold") + 64.8,
    );
    const chips = container.querySelectorAll(
      '[data-flow-semantic-chrome="core.chip"]',
    );
    expect(chips[0]?.getAttribute("width")).toBe(String(longWidth));
    const firstX = Number(chips[0]?.getAttribute("x"));
    const secondX = Number(chips[1]?.getAttribute("x"));
    // Spacing is the contract; absolute paint coords may be local or world.
    expect(secondX - firstX).toBe(longWidth + 8);
  });

  // Regression coverage for SceneRenderer timeline/fan correctness fixes.
  // `reducedMotion` is used as a deterministic time-travel knob: it snapshots
  // the scene at `durationMs` (the authored max of `cue.at + cue.duration`
  // across the timeline), so a timeline can be crafted to land the snapshot
  // at an exact millisecond without fake timers or animation-frame stepping.
  describe("timeline and fan correctness fixes", () => {
    it("lets a later reveal bring a node back after an earlier fade completes", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "box",
                kind: "rect",
                capabilityId: "core.rect",
                geometry: { x: 10, y: 10, width: 80, height: 40 },
                style: {},
              },
            ],
            timeline: [
              { id: "in", at: 0, duration: 100, action: "reveal", target: "box" },
              { id: "out", at: 200, duration: 100, action: "fade", target: "box" },
              // Authored later than the fade: supersedes it once active.
              { id: "back", at: 400, duration: 100, action: "reveal", target: "box" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      const opacity = container.querySelector<SVGGElement>(
        '[data-flow-node-id="box"]',
      )?.style.opacity;
      // Snapshot lands at durationMs = 500 (end of the "back" reveal), well
      // past the fade's own window — the node must be visible again, not
      // stuck hidden by the earlier, now-superseded fade.
      expect(opacity).not.toBe("0");
      expect(opacity).toBe("1");
    });

    it("does not hide a node forever after a completed non-exit fade with no later reveal", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "box",
                kind: "rect",
                capabilityId: "core.rect",
                geometry: { x: 10, y: 10, width: 80, height: 40 },
                style: {},
              },
            ],
            timeline: [
              { id: "out", at: 0, duration: 100, action: "fade", target: "box" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      // A completed `fade` (not `exit`) is not terminal: with no enter cue
      // authored at all, appearance falls through to "unchanged" / full
      // opacity rather than staying hidden.
      const group = container.querySelector<SVGGElement>(
        '[data-flow-node-id="box"]',
      );
      expect(group?.style.opacity).not.toBe("0");
    });

    it("keeps an `exit` cue terminal even after it completes", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "box",
                kind: "rect",
                capabilityId: "core.rect",
                geometry: { x: 10, y: 10, width: 80, height: 40 },
                style: {},
              },
            ],
            timeline: [
              { id: "out", at: 0, duration: 100, action: "exit", target: "box" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      expect(
        container.querySelector<SVGGElement>('[data-flow-node-id="box"]')?.style
          .opacity,
      ).toBe("0");
    });

    it("restores an authored dashed stroke once a draw cue completes", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "conn",
                kind: "connector",
                capabilityId: "core.connector",
                geometry: { x: 0, y: 0, width: 0, height: 0 },
                style: { dashed: true },
                from: { x: 0, y: 0 },
                to: { x: 200, y: 0 },
              },
            ],
            timeline: [
              { id: "draw", at: 0, duration: 200, action: "draw", target: "conn" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      // Snapshot lands at durationMs = 200 — the draw cue has fully
      // completed (progress === 1) — so the authored dash must restore
      // instead of staying parked at the 0/1 pathLength draw-reveal trick.
      const path = container.querySelector(
        '[data-flow-node-id="conn"] path[data-flow-arrowhead]',
      );
      expect(path?.getAttribute("stroke-dasharray")).toBe("16.2 13.5");
      expect(path?.getAttribute("stroke-dashoffset")).toBeNull();
      expect(path?.getAttribute("pathLength")).toBeNull();
    });

    it("keeps a mid-draw dashed connector hidden by the draw-reveal dash trick", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "conn",
                kind: "connector",
                capabilityId: "core.connector",
                geometry: { x: 0, y: 0, width: 0, height: 0 },
                style: { dashed: true },
                from: { x: 0, y: 0 },
                to: { x: 200, y: 0 },
              },
            ],
            timeline: [
              { id: "draw", at: 0, duration: 200, action: "draw", target: "conn" },
            ],
          }}
          playing={false}
          restartKey={0}
        />,
      );

      const path = container.querySelector(
        '[data-flow-node-id="conn"] path[data-flow-arrowhead]',
      );
      // At playbackTimeMs = 0 (mount, not playing), the draw cue has not
      // progressed: still mid-draw, dash trick still owns dasharray.
      expect(path?.getAttribute("stroke-dasharray")).toBe("1");
      expect(path?.getAttribute("stroke-dashoffset")).toBe("1");
    });

    it("fails closed (opacity 0) when a connector endpoint references an unresolved node id", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "source",
                kind: "rect",
                capabilityId: "core.rect",
                geometry: { x: 20, y: 40, width: 80, height: 50 },
                style: {},
              },
              {
                id: "broken-edge",
                kind: "connector",
                capabilityId: "core.connector",
                geometry: { x: 0, y: 0, width: 0, height: 0 },
                style: {},
                from: { nodeId: "source", anchor: "e" },
                to: { nodeId: "does-not-exist", anchor: "w" },
              },
            ],
            timeline: [],
          }}
          playing={false}
          restartKey={0}
        />,
      );

      expect(
        container.querySelector<SVGGElement>('[data-flow-node-id="broken-edge"]')
          ?.style.opacity,
      ).toBe("0");
    });

    it("renders a zero-duration emphasize cue as a one-frame peak within a small tolerance window", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "box",
                kind: "rect",
                capabilityId: "core.rect",
                geometry: { x: 10, y: 10, width: 80, height: 40 },
                style: {},
              },
            ],
            timeline: [
              {
                id: "pulse",
                at: 100,
                duration: 0,
                action: "emphasize",
                target: "box",
              },
              // Padding cue pushes the snapshot 10ms past the zero-duration
              // cue's exact instant, inside the tolerance window.
              { id: "pad", at: 110, duration: 0, action: "draw", target: "box" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      expect(
        container
          .querySelector('[data-flow-node-id="box"]')
          ?.getAttribute("data-emphasis-intensity"),
      ).toBe("1");
    });

    it("does not crash and renders no garbage geometry for a fan with fewer than two many-side endpoints", () => {
      let container: HTMLElement | undefined;
      expect(() => {
        ({ container } = render(
          <SceneRenderer
            scene={{
              roots: [
                {
                  id: "fan",
                  kind: "fan",
                  capabilityId: "core.fan-out",
                  geometry: { x: 0, y: 0, width: 0, height: 0 },
                  style: {},
                  from: { x: 0, y: 100 },
                  to: [],
                },
              ],
              timeline: [],
            }}
            playing={false}
            restartKey={0}
          />,
        ));
      }).not.toThrow();

      const fanGroup = container?.querySelector('[data-flow-node-id="fan"]');
      expect(fanGroup?.innerHTML ?? "").not.toMatch(/Infinity|NaN/);
    });

    it("reveals a fan's trunk and branch segments in the two MotionSignal ball phases", () => {
      const { container } = render(
        <SceneRenderer
          scene={{
            roots: [
              {
                id: "fan",
                kind: "fan",
                capabilityId: "core.fan-out",
                geometry: { x: 0, y: 0, width: 0, height: 0 },
                style: {},
                axis: "x",
                from: { x: 0, y: 100 },
                to: [
                  { x: 300, y: 50 },
                  { x: 300, y: 150 },
                ],
              },
            ],
            timeline: [
              // `at` clamps to 0 inside cueProgress but not inside
              // timelineDurationMs, landing the reducedMotion snapshot at
              // playbackTimeMs = 250 -> traceProgress = 0.25 (first half).
              { id: "fan-trace", at: -750, duration: 1000, action: "trace", target: "fan" },
            ],
          }}
          playing={false}
          restartKey={0}
          reducedMotion
        />,
      );

      const trunkPaths = container.querySelectorAll(
        '[data-flow-node-id="fan"] path[data-flow-fan-role="trunk"]',
      );
      const branchPaths = container.querySelectorAll(
        '[data-flow-node-id="fan"] path[data-flow-fan-role="branch"]',
      );
      expect(trunkPaths.length).toBeGreaterThan(0);
      expect(branchPaths.length).toBeGreaterThan(0);
      // Fan-out: trunk travels the first half (0.25 * 2 = 0.5 drawn)...
      for (const path of trunkPaths) {
        expect(path.getAttribute("stroke-dashoffset")).toBe("0.5");
      }
      // ...branches travel the second half (not yet started -> fully hidden).
      for (const path of branchPaths) {
        expect(path.getAttribute("stroke-dashoffset")).toBe("1");
      }
    });

    it("keeps an idle motion signal's SMIL loop mounted across a pause/resume toggle", () => {
      const scene: SceneIrLike = {
        roots: [
          {
            id: "source",
            kind: "group",
            capabilityId: "core.panel",
            geometry: { x: 20, y: 80, width: 120, height: 60 },
            style: {},
            props: { title: "Source" },
          },
          {
            id: "target",
            kind: "group",
            capabilityId: "core.panel",
            geometry: { x: 300, y: 80, width: 120, height: 60 },
            style: {},
            props: { title: "Target" },
          },
          {
            id: "signal",
            kind: "motion",
            capabilityId: "motion.signal",
            geometry: { x: 0, y: 0, width: 0, height: 0 },
            style: {},
            from: { nodeId: "source", anchor: "e" },
            to: { nodeId: "target", anchor: "w" },
          },
        ],
        timeline: [],
      };

      const { container, rerender } = render(
        <SceneRenderer scene={scene} playing restartKey={0} />,
      );
      const dotWhilePlaying = container.querySelector(
        '[data-flow-motion-signal="signal"] animateMotion',
      );
      expect(dotWhilePlaying).not.toBeNull();

      rerender(<SceneRenderer scene={scene} playing={false} restartKey={0} />);
      const dotWhilePaused = container.querySelector(
        '[data-flow-motion-signal="signal"] animateMotion',
      );
      // Pausing must not unmount/remount the SMIL loop — that would restart
      // `begin={delay}` from zero and desync the dot from the timeline.
      expect(dotWhilePaused).not.toBeNull();
    });

    it("freezes (not just keeps mounted) an idle motion signal's SMIL loop on pause", () => {
      // jsdom does not implement pauseAnimations/unpauseAnimations at all;
      // stub them so the regression can assert MotionSignal actually calls
      // the browser's own freeze mechanism instead of merely staying mounted
      // while the animation keeps running underneath.
      const pauseAnimations = vi.fn();
      const unpauseAnimations = vi.fn();
      const proto = SVGSVGElement.prototype as SVGSVGElement & {
        pauseAnimations?: () => void;
        unpauseAnimations?: () => void;
      };
      const originalPause = proto.pauseAnimations;
      const originalUnpause = proto.unpauseAnimations;
      proto.pauseAnimations = pauseAnimations;
      proto.unpauseAnimations = unpauseAnimations;
      try {
        const scene: SceneIrLike = {
          roots: [
            {
              id: "source",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 20, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Source" },
            },
            {
              id: "target",
              kind: "group",
              capabilityId: "core.panel",
              geometry: { x: 300, y: 80, width: 120, height: 60 },
              style: {},
              props: { title: "Target" },
            },
            {
              id: "signal",
              kind: "motion",
              capabilityId: "motion.signal",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              style: {},
              from: { nodeId: "source", anchor: "e" },
              to: { nodeId: "target", anchor: "w" },
            },
          ],
          timeline: [],
        };

        const { container, rerender } = render(
          <SceneRenderer scene={scene} playing restartKey={0} />,
        );
        const dot = () =>
          container.querySelector('[data-flow-motion-signal="signal"]');
        expect(dot()?.getAttribute("data-motion-paused")).toBeNull();
        expect(pauseAnimations).not.toHaveBeenCalled();

        rerender(<SceneRenderer scene={scene} playing={false} restartKey={0} />);
        expect(dot()?.getAttribute("data-motion-paused")).toBe("true");
        expect(pauseAnimations).toHaveBeenCalled();

        rerender(<SceneRenderer scene={scene} playing restartKey={0} />);
        expect(dot()?.getAttribute("data-motion-paused")).toBeNull();
        expect(unpauseAnimations).toHaveBeenCalled();
      } finally {
        proto.pauseAnimations = originalPause;
        proto.unpauseAnimations = originalUnpause;
      }
    });

    it("extends scene duration to the last authored camera keyframe, not just cue ends", () => {
      const scene: SceneIrLike = {
        roots: [
          {
            id: "panel",
            kind: "group",
            capabilityId: "core.panel",
            geometry: { x: 20, y: 20, width: 120, height: 60 },
            style: {},
            props: { title: "Panel" },
          },
        ],
        // The only cue ends at 100ms — far shorter than the camera track.
        timeline: [
          { id: "enter", at: 0, duration: 100, action: "enter", target: "panel" },
        ],
        camera: [
          { at: 0, x: 350, y: 200, zoom: 1 },
          { at: 1000, x: 350, y: 200, zoom: 4 },
        ],
      };

      // `reducedMotion` snaps `playbackTimeMs` to the scene's computed
      // duration — if duration were derived from cues alone (100ms) the
      // camera would sample far short of its final keyframe.
      const { container } = render(
        <SceneRenderer scene={scene} playing={false} restartKey={0} reducedMotion />,
      );
      const svg = container.querySelector("svg.scene-renderer");
      const viewBox = svg?.getAttribute("viewBox") ?? "";
      const [, , visibleWidth, visibleHeight] = viewBox
        .split(/\s+/)
        .map(Number);
      // Fully zoomed (zoom = 4): 1920/4 = 480, 1080/4 = 270.
      expect(visibleWidth).toBeCloseTo(480, 0);
      expect(visibleHeight).toBeCloseTo(270, 0);
    });

    it("inserts an off-trajectory junction into the polyline instead of dropping the ball split", () => {
      // A single straight segment: the nearest vertex to the junction is a
      // trajectory endpoint, which used to make the split bail out entirely.
      const d = "M0 0 L100 0";
      const result = splitFanTrajectoryAtJunction(d, { x: 40, y: 5 });
      expect(result).not.toBeUndefined();
      expect(result?.head).toContain("M0 0");
      expect(result?.head).toContain("L40 0");
      expect(result?.tail).toContain("M40 0");
      expect(result?.tail).toContain("L100 0");
    });

    it("orients a fan segment toward its destination by distance when neither atomic endpoint matches it exactly", () => {
      const span = {
        axis: "h" as const,
        fixed: 0,
        from: 0,
        to: 100,
        role: "branch" as const,
        // Far from both atomic endpoints (not within the pointsNear epsilon)
        // but much closer to `from` (0) than `to` (100).
        destination: { x: 10, y: 0 },
      };
      const segment = fanSegmentFromAtomic("branch-span", span);
      expect(segment.showMarker).toBe(true);
      // The marker sits at the path's terminal point — it must land on the
      // end closest to the destination (the `from` side), not default to
      // the atomic span's `to` side.
      expect(segment.d).toBe("M100 0 L0 0");
    });

    it("truncates a multi-segment arrow path at true arc length instead of rewriting only the last command's endpoint", () => {
      // jsdom implements neither getTotalLength nor getPointAtLength (and
      // does not even expose `SVGPathElement` as a global) — stub a
      // straight-polyline-accurate version on the live prototype so the DOM
      // cut path can run.
      const samplePath = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
      );
      const proto = Object.getPrototypeOf(samplePath) as SVGPathElement & {
        getTotalLength?: () => number;
        getPointAtLength?: (length: number) => DOMPoint;
      };
      const originalTotalLength = proto.getTotalLength;
      const originalPointAtLength = proto.getPointAtLength;
      const pointsOf = (path: SVGPathElement): Array<Readonly<{ x: number; y: number }>> => {
        const raw = path.getAttribute("d") ?? "";
        const points: Array<{ x: number; y: number }> = [];
        for (const match of raw.matchAll(
          /[ML]\s*(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)/gi,
        )) {
          points.push({ x: Number(match[1]), y: Number(match[2]) });
        }
        return points;
      };
      proto.getTotalLength = function (this: SVGPathElement) {
        const points = pointsOf(this);
        let total = 0;
        for (let i = 1; i < points.length; i++) {
          total += Math.hypot(
            points[i]!.x - points[i - 1]!.x,
            points[i]!.y - points[i - 1]!.y,
          );
        }
        return total;
      };
      proto.getPointAtLength = function (
        this: SVGPathElement,
        length: number,
      ) {
        const points = pointsOf(this);
        if (points.length === 0) {
          return { x: 0, y: 0 } as DOMPoint;
        }
        let remaining = Math.max(0, length);
        for (let i = 1; i < points.length; i++) {
          const segmentLength = Math.hypot(
            points[i]!.x - points[i - 1]!.x,
            points[i]!.y - points[i - 1]!.y,
          );
          if (remaining <= segmentLength || i === points.length - 1) {
            const t =
              segmentLength === 0 ? 0 : Math.min(1, remaining / segmentLength);
            return {
              x: points[i - 1]!.x + (points[i]!.x - points[i - 1]!.x) * t,
              y: points[i - 1]!.y + (points[i]!.y - points[i - 1]!.y) * t,
            } as DOMPoint;
          }
          remaining -= segmentLength;
        }
        const last = points[points.length - 1]!;
        return { x: last.x, y: last.y } as DOMPoint;
      };

      try {
        // Total length 203 (200 + 3); an inset of 6 cuts 6 units back from
        // the end, landing at length 197 — inside the *first* segment, not
        // the short trailing vertical one.
        const d = "M0 0 L200 0 L200 3";
        const result = shortenPathForArrowhead(d, 1, 6);
        const numbers = [...result.matchAll(/-?\d+(?:\.\d+)?/g)].map(Number);
        const [x1, _y1, x2, y2] = numbers.slice(-4);
        // The old chord-rewrite would move only the last L's endpoint to
        // (197, 0), leaving a backward stub from (200, 0) to (197, 0) whose
        // tangent points in -x — the wrong arrowhead direction. The fixed
        // arc-length truncation instead ends the whole path at (~197, 0)
        // while still heading in the path's true (+x) direction.
        expect(x2).toBeGreaterThan(x1 ?? 0);
        expect(y2).toBeCloseTo(0, 1);
        expect(x2).toBeCloseTo(197, 0);
      } finally {
        proto.getTotalLength = originalTotalLength;
        proto.getPointAtLength = originalPointAtLength;
      }
    });
  });
});
