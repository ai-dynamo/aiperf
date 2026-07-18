// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "./App";

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
        name: "Jump to a scene, beat, entity, or action",
      }),
    ).toBeTruthy();
  });
});
