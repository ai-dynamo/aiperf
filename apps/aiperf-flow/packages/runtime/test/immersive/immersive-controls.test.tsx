// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { ImmersiveControls } from "../../src/immersive/immersive-controls.js";
import type {
  ImmersiveControlsProps,
} from "../../src/immersive/immersive-controls.js";

afterEach(cleanup);

function renderControls(
  overrides: Partial<ImmersiveControlsProps> = {},
): Readonly<{
  onPlayPause: ReturnType<typeof vi.fn>;
  onExploreResume: ReturnType<typeof vi.fn>;
  onOpenCommands: ReturnType<typeof vi.fn>;
  onToggleTwin: ReturnType<typeof vi.fn>;
  onToggleFullscreen: ReturnType<typeof vi.fn>;
}> {
  const handlers = {
    onPlayPause: vi.fn(),
    onExploreResume: vi.fn(),
    onOpenCommands: vi.fn(),
    onToggleTwin: vi.fn(),
    onToggleFullscreen: vi.fn(),
  };
  render(
    <ImmersiveControls
      exploring={false}
      fullscreen="windowed"
      hud="present"
      playing={false}
      {...handlers}
      {...overrides}
    />,
  );
  return handlers;
}

function controls(): HTMLElement {
  return screen.getByRole("region", { name: "Immersive controls" });
}

describe("ImmersiveControls labels", () => {
  test("names every immersive playback action for the accessibility tree", () => {
    renderControls();

    expect(screen.getByRole("button", { name: "Play" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Explore" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Semantic twin" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Open commands" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Enter fullscreen" })).not.toBeNull();
  });

  test("reflects playing and exploring state in the toggle labels", () => {
    renderControls({ playing: true, exploring: true });

    expect(screen.getByRole("button", { name: "Pause" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Resume lesson" })).not.toBeNull();
  });

  test("marks fullscreen as pressed and offers to exit when not windowed", () => {
    renderControls({ fullscreen: "native" });

    const button = screen.getByRole("button", { name: "Exit fullscreen" });
    expect(button.getAttribute("aria-pressed")).toBe("true");
  });

  test("leaves fullscreen unpressed while windowed", () => {
    renderControls({ fullscreen: "windowed" });

    expect(
      screen.getByRole("button", { name: "Enter fullscreen" }).getAttribute("aria-pressed"),
    ).toBeNull();
  });
});

describe("ImmersiveControls actions", () => {
  test("routes each control to its callback exactly once", () => {
    const handlers = renderControls();

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    fireEvent.click(screen.getByRole("button", { name: "Semantic twin" }));
    fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
    fireEvent.click(screen.getByRole("button", { name: "Enter fullscreen" }));

    expect(handlers.onPlayPause).toHaveBeenCalledTimes(1);
    expect(handlers.onExploreResume).toHaveBeenCalledTimes(1);
    expect(handlers.onToggleTwin).toHaveBeenCalledTimes(1);
    expect(handlers.onOpenCommands).toHaveBeenCalledTimes(1);
    expect(handlers.onToggleFullscreen).toHaveBeenCalledTimes(1);
  });

  test("disables playback and suppresses its callback when requested", () => {
    const handlers = renderControls({ playbackDisabled: true });

    const play = screen.getByRole("button", { name: "Play" }) as HTMLButtonElement;
    expect(play.disabled).toBe(true);

    fireEvent.click(play);
    expect(handlers.onPlayPause).not.toHaveBeenCalled();
  });
});

describe("ImmersiveControls HUD visibility", () => {
  test("mirrors the requested HUD visibility on the chrome container", () => {
    renderControls({ hud: "quiet" });

    expect(controls().getAttribute("data-hud")).toBe("quiet");
    expect(controls().getAttribute("data-focused-within")).toBe("false");
  });

  test("restores chrome to present while a control holds focus", () => {
    renderControls({ hud: "hidden" });

    expect(controls().getAttribute("data-hud")).toBe("hidden");

    act(() => {
      (screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).focus();
    });

    expect(controls().getAttribute("data-focused-within")).toBe("true");
    expect(controls().getAttribute("data-hud")).toBe("present");
  });

  test("returns chrome to its quiet policy after focus leaves", () => {
    renderControls({ hud: "hidden" });
    const play = screen.getByRole("button", { name: "Play" }) as HTMLButtonElement;

    act(() => {
      play.focus();
    });
    expect(controls().getAttribute("data-hud")).toBe("present");

    act(() => {
      play.blur();
    });
    expect(controls().getAttribute("data-focused-within")).toBe("false");
    expect(controls().getAttribute("data-hud")).toBe("hidden");
  });
});
