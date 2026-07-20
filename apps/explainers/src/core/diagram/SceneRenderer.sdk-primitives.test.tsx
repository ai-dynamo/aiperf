/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SceneRenderer } from "./SceneRenderer";
import { resolveScene } from "./resolution/resolve-scene.js";
import type { SceneIrLike } from "./scene-types.js";
import { estimateTextWidth, stepperChipWidth } from "./text-metrics.js";

afterEach(cleanup);

describe("SceneRenderer SDK foundations", () => {
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
              geometry: { x: 8, y: 8, width: 160, height: 22 },
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
              capabilityId: "core.step",
              geometry: { x: 0, y: 0, width: 100, height: 26 },
              props: { label: "Stepper copy", index: 0 },
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
          (node) => node.textContent === copy,
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

    expect(textByContent("Profile")?.getAttribute("font-size")).toBe("12.6");
    expect(textByContent("source")?.getAttribute("font-size")).toBe("10.35");
    expect(textByContent("Authored")?.getAttribute("font-size")).toBe("18");
    expect(textByContent("Authored")?.style.fontSize).toBe("18px");
    expect(textByContent("Default")?.getAttribute("font-size")).toBe("12.6");
  });

  it("renders intrinsically sized semantic stepper labels", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
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
        }}
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
      estimateTextWidth("authoritative", 11, "bold") + 24,
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
      expect(path?.getAttribute("stroke-dasharray")).toBe("6 5");
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
  });
});
