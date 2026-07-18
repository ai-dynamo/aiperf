// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "./App";
import { previewNavigation, previewScene } from "./fixture";

afterEach(cleanup);

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
});

describe("immersive preview host", () => {
  test("opens on the real hub-and-spoke request investigation scene", () => {
    const navigation = previewNavigation();
    const scene = previewScene();

    expect(navigation.active).toMatchObject({
      flowId: "request-flow",
      sceneId: "request-investigation",
    });
    expect(scene.title).toBe("What made this slow?");
    expect(scene.accessibility.readingOrder).toHaveLength(15);

    const { container } = render(<App />);
    expect(container.querySelector('[data-preview-layout="hub-spoke"]')).toBeTruthy();
    expect(screen.getByText("AIPerf Flow · Scene study 02")).toBeTruthy();
    expect(
      screen.getByRole("heading", {
        name: "From one request to the whole system",
      }),
    ).toBeTruthy();
    expect(screen.getByText("SYSTEMS CHALK")).toBeTruthy();

    const sceneOutput = screen.getByRole("img", {
      name: /What made this slow\?/iu,
    });
    for (const label of [
      "1. Prompt enters the gateway",
      "2. Admission queues the work",
      "3. Prefix cache is consulted",
      "4. Prefill claims compute",
      "5. Decode streams tokens",
      "6. Telemetry supplies evidence",
      "7. The causal path resolves",
    ]) {
      expect(sceneOutput.getAttribute("aria-label")).toContain(label);
    }
  });

  test("authors a narrow single-column scene variant without visible wires", () => {
    const scene = previewScene();
    const narrow = scene.responsive.find(
      (variant) => variant.condition === "(max-width: 860px)",
    );

    expect(narrow).toBeDefined();
    const cards = narrow?.roots.filter(
      (node) =>
        node.kind === "group" &&
        node.id !== "request-hub",
    );
    expect(cards).toHaveLength(7);
    expect(new Set(cards?.map((node) => node.geometry.x))).toEqual(new Set([34]));
    expect(
      narrow?.roots
        .filter((node) => node.kind === "connector")
        .every((node) => node.style.strokeWidth === 0),
    ).toBe(true);
  });

  test("mounts one shared Causal Field without legacy player chrome", () => {
    const { container } = render(<App />);

    expect(screen.getAllByRole("region", { name: "Scene field" })).toHaveLength(1);
    expect(screen.getByRole("navigation", { name: "Causal path" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Open commands" })).toBeTruthy();
    expect(container.querySelector(".story-stage")).toBeNull();
    expect(container.querySelector('input[type="range"]')).toBeNull();
    expect(screen.queryByText("Back")).toBeNull();
  });

  test("opens the document browser as an overlay without replacing the scene", () => {
    render(<App />);

    const field = screen.getByRole("region", { name: "Scene field" });
    fireEvent.click(screen.getByRole("button", { name: "Open Flow browser" }));

    expect(screen.getByRole("complementary", { name: "Flow browser" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Scene field" })).toBe(field);
    expect(document.querySelector(".flow-workspace")?.getAttribute(
      "data-browser-collapsed",
    )).toBe("false");
  });

  test("routes Command-K to the shared Command Constellation", () => {
    render(<App />);

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });

    expect(
      screen.getByRole("dialog", {
        name: "Command Constellation",
      }),
    ).toBeTruthy();
  });
});
