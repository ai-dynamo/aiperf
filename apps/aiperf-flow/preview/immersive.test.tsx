// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { evaluateTimelineState } from "../packages/runtime/src/evaluate/timeline-state";
import type { RenderNodeIr } from "../packages/schema/src/ir";
import { App } from "./App";
import { previewNavigation, previewScene } from "./fixture";

function previewCss(): string {
  return readFileSync(join(process.cwd(), "preview/styles.css"), "utf8");
}

function matchMediaFor(query: string, matches: boolean) {
  return {
    matches,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  };
}

function findNode(
  roots: readonly RenderNodeIr[],
  id: string,
): RenderNodeIr | undefined {
  for (const node of roots) {
    if (node.id === id) {
      return node;
    }
    if (node.kind === "group" || node.kind === "component") {
      const nested = findNode(node.children, id);
      if (nested !== undefined) {
        return nested;
      }
    }
  }
  return undefined;
}

afterEach(cleanup);

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string) => matchMediaFor(query, false),
  });
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
});

describe("immersive preview host", () => {
  test.skip("opens on the real hub-and-spoke request investigation scene", () => {
    // SKIPPED: Demo request-flow scene removed in favor of explainer decks
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
    expect(screen.getByText("Systems Chalk")).toBeTruthy();

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

  test("renders a scrollable narrow hub sequence when matchMedia reports 860px", () => {
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: (query: string) =>
        matchMediaFor(query, query.includes("max-width: 860px")),
    });

    render(<App />);

    const scene = screen.getByRole("img", { name: /What made this slow\?/iu });
    expect(scene.getAttribute("viewBox")).toBe("0 0 320 1230");

    const css = previewCss();
    const scrollContract = css.match(
      /@media \(width <= 860px\)[\s\S]*?\.preview-shell\[data-preview-layout="hub-spoke"\] \.runtime-story\s*\{[^}]*overflow:\s*auto/u,
    );
    expect(scrollContract).toBeTruthy();
    expect(css).not.toMatch(
      /@media \(width <= 680px\)[\s\S]*?\.preview-shell\[data-preview-layout="hub-spoke"\] \.runtime-story\s*\{[^}]*overflow:\s*auto/u,
    );
  });

  test("authors Systems Chalk cards with neutral borders and circular badges", () => {
    const scene = previewScene();
    const gateway = findNode(scene.roots, "gateway");
    expect(gateway?.kind).toBe("group");

    const panel = findNode(scene.roots, "gateway-panel");
    expect(panel?.kind).toBe("rect");
    if (panel?.kind === "rect") {
      expect(panel.style.stroke).toBe("rgba(255,255,255,0.1)");
      expect(panel.style.stroke).not.toBe("#f6bd60");
    }

    const badge = findNode(scene.roots, "gateway-number-badge");
    expect(badge?.kind).toBe("rect");
    if (badge?.kind === "rect") {
      expect(badge.geometry.width).toBe(badge.geometry.height);
      expect(badge.style.stroke).toBe("#f6bd60");
    }

    const number = findNode(scene.roots, "gateway-number");
    expect(number?.kind).toBe("text");
    if (number?.kind === "text") {
      expect(number.text).toBe("1");
    }

    expect(findNode(scene.roots, "gateway-client")?.kind).toBe("rect");
    expect(findNode(scene.roots, "gateway-edge")?.kind).toBe("rect");
    expect(findNode(scene.roots, "gateway-diagram-link")?.kind).toBe(
      "connector",
    );
    expect(findNode(scene.roots, "gateway-diagram")?.kind).toBeUndefined();

    const hubDetail = findNode(scene.roots, "request-hub-detail");
    expect(hubDetail?.kind).toBe("text");
    if (hubDetail?.kind === "text") {
      expect(hubDetail.text).toBe(
        "Follow one causal path across every layer of inference.",
      );
    }

    expect(findNode(scene.roots, "prefill-bar-3")?.kind).toBe("rect");
    expect(findNode(scene.roots, "admission-queue-9")?.kind).toBe("rect");
  });

  test("active cause path draw-on follows the authored timeline trace", () => {
    const css = previewCss();
    expect(css).not.toContain("preview-cause-draw");
    expect(css).not.toMatch(
      /\[data-draw-command-id="request-to-gateway"\][\s\S]{0,200}animation:/u,
    );

    const scene = previewScene();
    const before = evaluateTimelineState(scene.timeline, 0);
    const mid = evaluateTimelineState(scene.timeline, 1_000);
    const after = evaluateTimelineState(scene.timeline, 2_000);

    expect(before.targets["request-to-gateway"]).toMatchObject({
      action: "trace",
      progress: 0,
    });
    expect(mid.targets["request-to-gateway"]?.action).toBe("trace");
    expect(mid.targets["request-to-gateway"]?.progress).toBeGreaterThan(0);
    expect(mid.targets["request-to-gateway"]?.progress).toBeLessThan(1);
    expect(after.targets["request-to-gateway"]).toMatchObject({
      action: "trace",
      progress: 1,
    });
  });

  test("mounts one shared Causal Field without legacy player chrome", () => {
    const { container } = render(<App />);

    expect(screen.getAllByRole("region", { name: "Scene field" })).toHaveLength(1);
    expect(screen.getByRole("navigation", { name: "Causal path" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Open commands" })).toBeTruthy();
    expect(container.querySelector(".story-stage")).toBeNull();
    expect(container.querySelector('input[type="range"]')).toBeNull();
    // New persistent bottom nav should be present with Back/Next buttons
    const bottomNav = container.querySelector(".preview-bottom-nav");
    expect(bottomNav).toBeTruthy();
  });

  test("shows the document browser as a persistent sidebar", () => {
    const { container } = render(<App />);

    const field = screen.getByRole("region", { name: "Scene field" });

    // Sidebar should be present and part of the main layout
    const sidebar = container.querySelector(".flow-browser");
    expect(sidebar).toBeTruthy();

    // Scene field should still be present alongside sidebar
    expect(screen.getByRole("region", { name: "Scene field" })).toBe(field);

    // Flow workspace should be present in the DOM
    const workspace = container.querySelector(".flow-workspace");
    expect(workspace).toBeTruthy();

    // Main section should be present
    const mainSection = container.querySelector(".flow-main-section");
    expect(mainSection).toBeTruthy();
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
