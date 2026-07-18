// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { ComponentNodeIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";
import type { ReactNode } from "react";

import {
  CapabilityRegistry,
  type RenderContext,
  type RuntimeCapability,
} from "../src/registry.js";
import { SceneRenderer } from "../src/renderer.js";

const sourceMap = {
  source: "binding.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

afterEach(cleanup);

function spanMapCapability(): RuntimeCapability<ComponentNodeIr> {
  return {
    descriptor: {
      id: "core.span-map",
      version: "1.0.0",
      kind: "layout",
      description: "Span map",
      nodeKinds: ["component"],
      deterministic: true,
      accessibility: {
        requiresLabel: true,
        keyboardOperable: false,
        screenReaderFallback: true,
      },
      fallback: "Span map unavailable",
      cost: { base: 2, perNode: 1 },
    },
    render(_node: ComponentNodeIr, _context: RenderContext): ReactNode {
      return <text data-testid="span-map">span-map</text>;
    },
  };
}

describe("capability binding", () => {
  test("dispatches component nodes by capabilityId", () => {
    const registry = new CapabilityRegistry();
    registry.register(spanMapCapability());

    const scene: SceneIr = {
      id: "main",
      title: "Main",
      summary: "Binding test",
      roots: [
        {
          kind: "component",
          id: "tok-map",
          capabilityId: "core.span-map",
          props: {},
          children: [],
          geometry: { x: 0, y: 0, width: 100, height: 40 },
          style: {},
          accessibility: { label: "Span map" },
          fallback: "Span map unavailable",
          sourceMap,
        },
      ],
      camera: [],
      timeline: [],
      narration: "",
      interactions: [],
      responsive: [],
      accessibility: { label: "Main", readingOrder: ["tok-map"] },
      fallback: "Scene unavailable",
      sourceMap,
    };

    render(<SceneRenderer registry={registry} scene={scene} />);
    expect(screen.getByTestId("span-map").textContent).toBe("span-map");
  });
});
