/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SceneRenderer } from "./SceneRenderer";

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
    ).toBe("80");
  });
});
