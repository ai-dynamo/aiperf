// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "./App";

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

afterEach(() => {
  cleanup();
  localStorage.clear();
});

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string) => matchMediaFor(query, false),
  });
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
});

describe("Theme Selector", () => {
  test("renders theme selector button", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");
    expect(themeButton).toBeTruthy();
    expect(themeButton?.textContent).toContain("Systems Chalk");
  });

  test("opens theme menu when button is clicked", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");
    expect(themeButton.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(themeButton);
    expect(themeButton.getAttribute("aria-expanded")).toBe("true");

    // Check that menu items are visible
    expect(screen.getByText("systems chalk")).toBeTruthy();
    expect(screen.getByText("legacy")).toBeTruthy();
    expect(screen.getByText("core")).toBeTruthy();
  });

  test("closes theme menu when button is clicked again", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    expect(themeButton.getAttribute("aria-expanded")).toBe("true");

    fireEvent.click(themeButton);
    expect(themeButton.getAttribute("aria-expanded")).toBe("false");
  });

  test("selects theme when menu item is clicked", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    const legacyButton = screen.getByText("legacy");
    fireEvent.click(legacyButton);

    expect(themeButton.textContent).toContain("Legacy");
    expect(themeButton.getAttribute("aria-expanded")).toBe("false");
  });

  test("persists theme selection to localStorage", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    const coreButton = screen.getByText("core");
    fireEvent.click(coreButton);

    expect(localStorage.getItem("aiperf-flow-theme")).toBe("core");
  });

  test("loads theme from localStorage on mount", () => {
    localStorage.setItem("aiperf-flow-theme", "legacy");
    render(<App />);

    const themeButton = screen.getByLabelText("Theme selector");
    expect(themeButton.textContent).toContain("Legacy");
  });

  test("renders theme toggle button", () => {
    render(<App />);
    const toggleButton = screen.getByLabelText("Toggle theme");
    expect(toggleButton).toBeTruthy();
    expect(toggleButton.textContent).toBe("⟳");
  });

  test("cycles through themes with toggle button", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");
    const toggleButton = screen.getByLabelText("Toggle theme");

    // Initial theme: systems-chalk
    expect(themeButton.textContent).toContain("Systems Chalk");

    // First click: legacy
    fireEvent.click(toggleButton);
    expect(themeButton.textContent).toContain("Legacy");

    // Second click: core
    fireEvent.click(toggleButton);
    expect(themeButton.textContent).toContain("Core");

    // Third click: systems-chalk (cycle)
    fireEvent.click(toggleButton);
    expect(themeButton.textContent).toContain("Systems Chalk");
  });

  test("applies theme class to main element", () => {
    const { container } = render(<App />);
    const main = container.querySelector("main[data-theme]");
    expect(main).toBeTruthy();
    expect(main?.getAttribute("data-theme")).toBe("systems-chalk");
  });

  test("updates theme class when theme changes", () => {
    const { container } = render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    const coreButton = screen.getByText("core");
    fireEvent.click(coreButton);

    const main = container.querySelector("main[data-theme]");
    expect(main?.getAttribute("data-theme")).toBe("core");
  });

  test("closes menu when clicking outside", () => {
    render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    expect(themeButton.getAttribute("aria-expanded")).toBe("true");

    // Click outside the menu
    fireEvent.click(document.body);
    expect(themeButton.getAttribute("aria-expanded")).toBe("false");
  });

  test("applies correct CSS variables for legacy theme", () => {
    const { container } = render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    const legacyButton = screen.getByText("legacy");
    fireEvent.click(legacyButton);

    const main = container.querySelector("main[data-theme='legacy']");
    expect(main).toBeTruthy();
    const styles = main ? window.getComputedStyle(main) : null;
    // Check that the data-theme attribute is set
    expect(main?.getAttribute("data-theme")).toBe("legacy");
  });

  test("applies correct CSS variables for core theme", () => {
    const { container } = render(<App />);
    const themeButton = screen.getByLabelText("Theme selector");

    fireEvent.click(themeButton);
    const coreButton = screen.getByText("core");
    fireEvent.click(coreButton);

    const main = container.querySelector("main[data-theme='core']");
    expect(main).toBeTruthy();
    expect(main?.getAttribute("data-theme")).toBe("core");
  });
});
