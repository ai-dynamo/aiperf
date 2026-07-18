// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "./App";
import { discoverAllScenes, discoverScenesByFlow } from "./fixture";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

beforeEach(() => {
  sessionStorage.clear();
  Object.defineProperty(window, "matchMedia", {
    writable: true,
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

  class FakeUtterance {
    rate = 1;
    volume = 1;
    voice: SpeechSynthesisVoice | null = null;
    constructor(readonly text: string) {}
  }

  const synthesis = {
    speaking: false,
    pending: false,
    paused: false,
    onvoiceschanged: null,
    getVoices: () => [],
    speak: vi.fn(),
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  };

  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    writable: true,
    value: synthesis,
  });
  Object.defineProperty(window, "SpeechSynthesisUtterance", {
    configurable: true,
    writable: true,
    value: FakeUtterance,
  });

  HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
});

describe("scene discovery", () => {
  test("discovers all available scenes", () => {
    const scenes = discoverAllScenes();

    expect(scenes.length).toBeGreaterThan(0);
    expect(scenes).toContainEqual(
      expect.objectContaining({
        flowId: "request-flow",
        sceneId: "request-investigation",
        title: "What made this slow?",
      }),
    );
    expect(scenes).toContainEqual(
      expect.objectContaining({
        flowId: "architecture",
        sceneId: "control-plane",
      }),
    );
    expect(scenes).toContainEqual(
      expect.objectContaining({
        flowId: "endpoint-lifecycle",
        sceneId: "resolve-endpoint",
      }),
    );
  });

  test("groups scenes by flow for display", () => {
    const groups = discoverScenesByFlow();

    expect(groups.length).toBe(3);
    expect(groups[0]).toMatchObject({
      flowId: "request-flow",
      flowTitle: expect.stringMatching(/request/i),
    });
    expect(groups[0]?.scenes.length).toBeGreaterThan(0);
  });

  test("each scene card has required fields", () => {
    const scenes = discoverAllScenes();

    scenes.forEach((scene) => {
      expect(scene.flowId).toBeTruthy();
      expect(scene.flowTitle).toBeTruthy();
      expect(scene.sceneId).toBeTruthy();
      expect(scene.title).toBeTruthy();
      expect(scene.description).toBeTruthy();
      expect(scene.chapterId).toBeTruthy();
    });
  });
});

describe("preview app home page", () => {
  test("loads with home page displayed by default", () => {
    render(<App />);

    expect(screen.getByText(/AIPerf Flow Scenes/i)).toBeTruthy();
    expect(screen.getByText(/interactive scenes/i)).toBeTruthy();
  });

  test("displays all flow sections with scene cards", () => {
    render(<App />);

    const flowTitles = screen.getAllByText(/\.flow$/);
    const flowTitlesText = flowTitles.map(el => el.textContent);
    expect(flowTitlesText).toContain("request-flow.flow");
    expect(flowTitlesText).toContain("architecture.flow");
    expect(flowTitlesText).toContain("endpoint-lifecycle.flow");
  });

  test("renders scene cards with titles and descriptions", () => {
    render(<App />);

    const cards = screen.getAllByRole("button", {
      name: /Load .* scene/i,
    });
    expect(cards.length).toBeGreaterThan(0);

    // Check for some specific scene titles in card titles (h3 elements)
    const titles = screen.getAllByRole("heading", { level: 3 });
    const titleTexts = titles.map(t => t.textContent);
    expect(titleTexts).toContain("What made this slow?");
    expect(titleTexts).toContain("Control plane");
    expect(titleTexts).toContain("Resolve endpoint");
  });

  test("clicking a scene card loads the scene", () => {
    render(<App />);

    // Home page should be visible
    expect(screen.getByText(/AIPerf Flow Scenes/i)).toBeTruthy();

    // Click on the request investigation scene
    const card = screen.getByRole("button", {
      name: /Load What made this slow/i,
    });
    fireEvent.click(card);

    // Home page should be gone and scene should load
    expect(screen.queryByText(/AIPerf Flow Scenes/i)).toBeNull();
  });

  test("clicking home button returns to home page", () => {
    render(<App />);

    // Load a scene
    const card = screen.getByRole("button", {
      name: /Load What made this slow/i,
    });
    fireEvent.click(card);

    expect(screen.queryByText(/AIPerf Flow Scenes/i)).toBeNull();

    // Click home button
    const homeButton = screen.getByRole("button", {
      name: /AIPerf Flow home/i,
    });
    fireEvent.click(homeButton);

    // Home page should be visible again
    expect(screen.getByText(/AIPerf Flow Scenes/i)).toBeTruthy();
  });

  test("all scene cards are accessible", () => {
    render(<App />);

    const cards = screen.getAllByRole("button", {
      name: /Load .* scene/i,
    });

    cards.forEach((card) => {
      expect(card).toHaveProperty("type", "button");
    });
  });

  test("scene cards display flow titles as kickers", () => {
    render(<App />);

    const kickers = screen.getAllByText(/\.flow/i);
    const requestFlowKicker = kickers.find(el =>
      el.className && el.className.includes("scene-card-kicker")
    );
    expect(requestFlowKicker).toBeTruthy();
  });

  test("home page shows correct scene counts", () => {
    render(<App />);

    const subtitle = screen.getByText(/interactive scenes/i);
    expect(subtitle.textContent).toMatch(/\d+ flows with \d+ interactive scenes/);
  });

  test("can navigate between multiple scenes from home", () => {
    render(<App />);

    // Load first scene
    const card1 = screen.getByRole("button", {
      name: /Load What made this slow/i,
    });
    fireEvent.click(card1);
    expect(screen.queryByText(/AIPerf Flow Scenes/i)).toBeNull();

    // Return to home
    const homeButton = screen.getByRole("button", {
      name: /AIPerf Flow home/i,
    });
    fireEvent.click(homeButton);
    expect(screen.getByText(/AIPerf Flow Scenes/i)).toBeTruthy();

    // Load different scene
    const card2 = screen.getByRole("button", {
      name: /Load Control plane/i,
    });
    fireEvent.click(card2);
    expect(screen.queryByText(/AIPerf Flow Scenes/i)).toBeNull();

    // Return to home again
    fireEvent.click(homeButton);
    expect(screen.getByText(/AIPerf Flow Scenes/i)).toBeTruthy();
  });

  test("sidebar is always visible and never hidden", () => {
    const { container } = render(<App />);

    // On home page, sidebar should be present in the DOM
    const sidebar = container.querySelector(".flow-browser");
    expect(sidebar).toBeTruthy();

    // Load a scene
    const card = screen.getByRole("button", {
      name: /Load What made this slow/i,
    });
    fireEvent.click(card);

    // Sidebar should still be present after loading a scene
    const sidebarAfter = container.querySelector(".flow-browser");
    expect(sidebarAfter).toBeTruthy();

    // Sidebar should be displayed as flex (part of flow-workspace flex layout)
    const workspace = container.querySelector(".flow-workspace");
    const style = window.getComputedStyle(workspace);
    expect(style.display).toBe("flex");
  });

  test("theme menu is hidden on home page", () => {
    render(<App />);

    // On home page, theme cluster should be hidden
    const themeClusters = document.querySelectorAll(".preview-theme-cluster");
    expect(themeClusters.length).toBeGreaterThan(0);

    const themeCluster = themeClusters[0];
    if (themeCluster) {
      const style = window.getComputedStyle(themeCluster);
      expect(style.display).toBe("none");
    }

    // Load a scene
    const card = screen.getByRole("button", {
      name: /Load What made this slow/i,
    });
    fireEvent.click(card);

    // Theme menu should be visible
    if (themeCluster) {
      const styleAfter = window.getComputedStyle(themeCluster);
      expect(styleAfter.display).not.toBe("none");
    }
  });
});
