// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import type { DisplayList } from "../../src/display-list.js";
import { SvgFallback } from "../../src/backends/svg/svg-fallback.js";
import type { EvaluatedScene } from "../../src/evaluate/types.js";

afterEach(cleanup);

const displayList = {
  commands: [
    {
      kind: "path",
      id: "request-shape",
      order: 0,
      path: "M 8 12 H 88 V 44 H 8 Z",
      fill: "#76b900",
      paintBounds: { x: 8, y: 12, width: 80, height: 32 },
      damageBounds: { x: 8, y: 12, width: 80, height: 32 },
    },
    {
      kind: "text",
      id: "request-label",
      order: 1,
      text: "Request",
      origin: { x: 16, y: 32 },
      font: { family: "sans-serif", sizePx: 16 },
      fill: "#ffffff",
      paintBounds: { x: 16, y: 16, width: 56, height: 20 },
      damageBounds: { x: 16, y: 16, width: 56, height: 20 },
    },
  ],
  hitRegions: [
    {
      id: "request-hit",
      semanticId: "request",
      order: 0,
      bounds: { x: 8, y: 12, width: 80, height: 32 },
    },
  ],
  paintBounds: { x: 0, y: 0, width: 320, height: 180 },
  damageBounds: { x: 0, y: 0, width: 320, height: 180 },
} satisfies DisplayList;

const scene = {
  sceneId: "request-scene",
  atMs: 120,
  displayList,
  semantic: {
    readingOrder: ["request"],
    entities: [
      {
        id: "request",
        label: "Inference request",
        description: "Request awaiting model service",
      },
    ],
    relations: [],
  },
} satisfies EvaluatedScene;

describe("SVG fallback", () => {
  test("preserves entity ids, labels, focus targets, and selection", () => {
    const { container } = render(
      <SvgFallback
        displayList={displayList}
        focusedEntityId="request"
        scene={scene}
        selectedEntityIds={["request"]}
      />,
    );

    const entity = container.querySelector('[data-entity-id="request"]');
    expect(entity?.getAttribute("aria-label")).toBe("Inference request");
    expect(entity?.getAttribute("data-focus-target")).toBe("request");
    expect(entity?.getAttribute("data-selected")).toBe("true");
    expect(entity?.getAttribute("data-focused")).toBe("true");
    expect(entity?.getAttribute("id")).toBe("flow-svg-request");
    expect(container.querySelector('[data-draw-command-id="request-shape"]')).not.toBeNull();
    expect(
      container.querySelector('[data-draw-command-id="request-label"]')
        ?.textContent,
    ).toBe("Request");
  });

  test("preserves fallback meaning in SVG descriptions", () => {
    const { container } = render(
      <SvgFallback displayList={displayList} scene={scene} />,
    );

    expect(container.querySelector("svg > desc")?.textContent).toBe(
      "Inference request. Request awaiting model service",
    );
    expect(
      container.querySelector('[data-entity-id="request"] > desc')?.textContent,
    ).toBe("Request awaiting model service");
  });

  test("renders an empty-list fallback without inspecting Flow IR", () => {
    const emptyList = {
      ...displayList,
      commands: [],
      hitRegions: [],
    } satisfies DisplayList;

    const { container } = render(
      <SvgFallback displayList={emptyList} scene={scene} />,
    );

    expect(container.querySelector('[role="note"]')?.textContent).toBe(
      "Inference request. Request awaiting model service",
    );
  });
});
