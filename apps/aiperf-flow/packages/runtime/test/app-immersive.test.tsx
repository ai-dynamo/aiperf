// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

// Mounted integration contracts for the immersive Causal Field shell (Task 6).
//
// This suite deliberately does NOT re-cover the assertions already exercised by
// app.test.tsx (SVG display-list mount vs SceneRenderer, twin mount/reachability,
// canvas-hit twin selection, twin keyboard activation, played-time exploration
// pause, scene-navigation reset, invalid-scene fallback chrome). It focuses on
// the remaining immersive contracts:
//
// - the scene stage is the dominant "Scene field" region and no media/range
//   scrubber survives from the retired video-player grammar;
// - causal beats seek the exact authored integer virtual time;
// - the Command Constellation jumps to scenes, beats, entities, and the twin;
// - Context Lens opens from both Canvas hits and semantic activation;
// - Focus World enters and exits without moving the active beat;
// - exploration/resume restores the exact seeked beat, selection, and focus;
// - a Canvas-unavailable fallback keeps HUD, commands, twin, and causal path;
// - fullscreen denial announces a recoverable message on the live region;
// - quiet/hidden HUD policy never conceals captions or focused controls.

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { FlowApp } from "../src/app.js";
import { FULLSCREEN_DENIED_MESSAGE, type FullscreenAdapter } from "../src/fullscreen.js";
import type { NarratorBackend } from "../src/narrative/narrator.js";
import type { Clock } from "../src/player.js";

const sourceMap = {
  source: "request-flow.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  window.history.replaceState(null, "", "/");
  vi.restoreAllMocks();
});

/** Deterministic narrator so playback never touches Kokoro or Web Speech. */
const silentNarrator: NarratorBackend = Object.freeze({
  available: false,
  voices: () => Object.freeze([]),
  speak: () => undefined,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
});

/** Manually advanced clock so scene time only moves when a test asks for it. */
class VirtualClock implements Clock {
  #nowNs: bigint;
  #nextHandle = 1;
  readonly #callbacks = new Map<number, () => void>();

  constructor(nowNs = 0n) {
    this.#nowNs = nowNs;
  }

  nowNs(): bigint {
    return this.#nowNs;
  }

  requestFrame(callback: () => void): number {
    const handle = this.#nextHandle;
    this.#nextHandle += 1;
    this.#callbacks.set(handle, callback);
    return handle;
  }

  cancelFrame(handle: number): void {
    this.#callbacks.delete(handle);
  }

  advanceMs(elapsedMs: number): void {
    this.#nowNs += BigInt(elapsedMs) * 1_000_000n;
    const callbacks = [...this.#callbacks.values()];
    this.#callbacks.clear();
    callbacks.forEach((callback) => callback());
  }
}

/**
 * A three-beat request-lifecycle scene: `arrival` (0ms), `admission` (600ms),
 * and `first-token` (1500ms) each reveal a diagram entity. Distinct authored
 * times let tests assert exact beat seeks and stable active-beat identity.
 */
function interactiveScene(overrides: Partial<SceneIr> = {}): SceneIr {
  return {
    id: "execution",
    title: "Execution boundary",
    summary: "The CLI starts a runtime.",
    roots: [
      {
        kind: "rect",
        id: "cli",
        geometry: { x: 40, y: 100, width: 160, height: 72 },
        style: { fill: "#315b8a" },
        accessibility: { label: "CLI", description: "Command-line process" },
        fallback: "CLI",
        sourceMap,
      },
      {
        kind: "rect",
        id: "runtime",
        geometry: { x: 300, y: 100, width: 180, height: 72 },
        style: { fill: "#244a35" },
        accessibility: { label: "Runtime", description: "Execution runtime" },
        fallback: "Runtime",
        sourceMap,
      },
      {
        kind: "text",
        id: "label",
        geometry: { x: 60, y: 130, width: 120, height: 24 },
        style: { fill: "#ffffff" },
        text: "CLI",
        accessibility: { label: "CLI label", description: "Label inside the CLI node" },
        fallback: "CLI",
        sourceMap,
      },
      {
        kind: "connector",
        id: "spawn",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#7aa2f7" },
        from: { nodeId: "cli" },
        to: { nodeId: "runtime" },
        accessibility: { label: "Spawn connection", description: "CLI starts the runtime" },
        fallback: "CLI starts Runtime",
        sourceMap,
      },
    ],
    camera: [],
    timeline: [
      { id: "arrival", at: 0, duration: 400, action: "arrival", target: "cli", sourceMap },
      { id: "admission", at: 600, duration: 400, action: "admission", target: "runtime", sourceMap },
      { id: "first-token", at: 1500, duration: 400, action: "first-token", target: "runtime", sourceMap },
    ],
    narration: "The CLI starts a fresh runtime and dispatches work.",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Execution boundary diagram",
      readingOrder: ["cli", "runtime", "spawn"],
    },
    fallback: "CLI starts Runtime.",
    sourceMap,
    ...overrides,
  };
}

function flowWith(scenes: readonly SceneIr[]): FlowIr {
  return {
    irVersion: 2,
    id: "request-flow",
    title: "Request flow",
    capabilities: [],
    tokens: {},
    themes: [],
    scenes,
    sourceMap,
  } as unknown as FlowIr;
}

function mockCanvas2d(): void {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => {
    return {
      setTransform() {},
      clearRect() {},
      save() {},
      restore() {},
      beginPath() {},
      closePath() {},
      moveTo() {},
      lineTo() {},
      rect() {},
      fill() {},
      stroke() {},
      fillText() {},
      strokeText() {},
      drawImage() {},
      clip() {},
      scale() {},
      translate() {},
      measureText: () => ({ width: 10 }),
      canvas: document.createElement("canvas"),
    } as unknown as CanvasRenderingContext2D;
  }) as typeof HTMLCanvasElement.prototype.getContext;
}

type RenderOptions = Readonly<{
  scenes?: readonly SceneIr[];
  forceSvgFallback?: boolean;
  clock?: Clock;
  fullscreenAdapter?: FullscreenAdapter;
}>;

/** Mounts FlowApp with a frozen clock and silent narrator by default. */
function renderApp(options: RenderOptions = {}): ReturnType<typeof render> {
  const {
    scenes = [interactiveScene()],
    forceSvgFallback = true,
    clock = new VirtualClock(),
    fullscreenAdapter,
  } = options;
  return render(
    <FlowApp
      clock={clock}
      flow={flowWith(scenes)}
      forceSvgFallback={forceSvgFallback}
      narratorBackend={silentNarrator}
      {...(fullscreenAdapter === undefined ? {} : { fullscreenAdapter })}
    />,
  );
}

function playbackTime(): string {
  return screen.getByRole("status", { name: "Playback time" }).textContent ?? "";
}

function causalPath(): HTMLElement {
  return screen.getByRole("navigation", { name: "Causal path" });
}

function seekBeatViaPath(label: string): void {
  fireEvent.click(within(causalPath()).getByRole("button", { name: label }));
}

function openCommandPalette(): HTMLElement {
  fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
  return screen.getByRole("dialog", { name: "Command Constellation" });
}

function runCommand(dialog: HTMLElement, query: string, commandId: string): void {
  fireEvent.change(within(dialog).getByRole("searchbox"), {
    target: { value: query },
  });
  const option = dialog.querySelector(`[data-command-id="${commandId}"]`);
  expect(option).not.toBeNull();
  fireEvent.click(option as Element);
}

// ============================================================================
// Dominant Scene field and retired scrubber
// ============================================================================

describe("FlowApp Causal Field shell", () => {
  test("mounts the Scene field as the dominant stage without any media scrubber", () => {
    const { container } = renderApp();

    const sceneField = screen.getByRole("region", { name: "Scene field" });
    expect(sceneField.querySelector("svg.aiperf-flow__svg-fallback")).not.toBeNull();

    // The retired video-player grammar left no free-form or media scrubber.
    expect(screen.queryByRole("slider")).toBeNull();
    expect(container.querySelector('input[type="range"]')).toBeNull();
    expect(container.querySelector("progress")).toBeNull();
    expect(container.querySelector("[data-scrubber]")).toBeNull();
  });

  test("keeps beat traversal on a labelled navigation control, not a range input", () => {
    renderApp();

    const path = causalPath();
    expect(within(path).getByRole("button", { name: "arrival" })).not.toBeNull();
    expect(within(path).getByRole("button", { name: "admission" })).not.toBeNull();
    expect(within(path).getByRole("button", { name: "first token" })).not.toBeNull();
    expect(path.querySelector('input[type="range"]')).toBeNull();
  });
});

// ============================================================================
// Exact integer beat seeks
// ============================================================================

describe("FlowApp causal beat seeking", () => {
  test.each([
    ["arrival", "arrival", /^\s*0\s*ms/u],
    ["admission", "admission", /600\s*ms/u],
    ["first token", "first-token", /1500\s*ms/u],
  ])(
    "seeks the exact authored time when the %s beat is chosen",
    (label, beatId, expected) => {
      renderApp();

      seekBeatViaPath(label);

      expect(playbackTime()).toMatch(expected);
      expect(causalPath().getAttribute("data-current-beat")).toBe(beatId);
    },
  );
});

// ============================================================================
// Command Constellation jumps
// ============================================================================

describe("FlowApp Command Constellation jumps", () => {
  test("jumps to another scene by title", () => {
    renderApp({
      scenes: [
        interactiveScene(),
        interactiveScene({
          id: "results",
          title: "Results boundary",
          narration: "Results scene transcript.",
        }),
      ],
    });

    runCommand(openCommandPalette(), "Results", "scene:results");

    expect(screen.getByRole("heading", { name: "Results boundary" })).not.toBeNull();
    expect(
      screen.getByRole("region", { name: "Semantic outline" }).getAttribute("data-scene-id"),
    ).toBe("results");
  });

  test("jumps to a beat and seeks its exact authored time", () => {
    renderApp();

    runCommand(openCommandPalette(), "first token", "beat:first-token");

    expect(playbackTime()).toMatch(/1500\s*ms/u);
    expect(causalPath().getAttribute("data-current-beat")).toBe("first-token");
  });

  test("jumps to an entity and opens its Context Lens", () => {
    renderApp();

    runCommand(openCommandPalette(), "Runtime", "entity:runtime");

    const lens = screen.getByRole("region", { name: "Context Lens" });
    expect(lens.getAttribute("data-entity-id")).toBe("runtime");
  });

  test("jumps to the semantic twin and expands it", () => {
    renderApp();

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    expect(twin.getAttribute("data-compact")).toBe("true");

    runCommand(openCommandPalette(), "Expand semantic twin", "action:twin");

    expect(twin.getAttribute("data-compact")).toBe("false");
    expect(twin.className).not.toContain("aiperf-flow__semantic-twin--compact");
  });
});

// ============================================================================
// Context Lens activation surfaces
// ============================================================================

describe("FlowApp Context Lens", () => {
  test("opens from a Canvas pointer hit on the matching entity", () => {
    mockCanvas2d();
    const { container } = renderApp({ forceSvgFallback: false });

    const canvas = container.querySelector("canvas.aiperf-flow__canvas");
    expect(canvas).not.toBeNull();
    Object.defineProperty(canvas, "getBoundingClientRect", {
      value: () => ({
        x: 0,
        y: 0,
        top: 0,
        left: 0,
        width: 480,
        height: 240,
        right: 480,
        bottom: 240,
        toJSON() {
          return {};
        },
      }),
    });

    // CLI hit region spans (40,100)-(200,172) in scene coordinates.
    fireEvent.pointerDown(canvas as Element, { clientX: 80, clientY: 130 });

    const lens = screen.getByRole("region", { name: "Context Lens" });
    expect(lens.getAttribute("data-entity-id")).toBe("cli");
    expect(within(lens).getByText("Command-line process")).not.toBeNull();
  });

  test("opens from semantic twin activation", () => {
    renderApp();

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    fireEvent.click(within(twin).getByRole("button", { name: "Runtime" }));

    const lens = screen.getByRole("region", { name: "Context Lens" });
    expect(lens.getAttribute("data-entity-id")).toBe("runtime");
    expect(within(lens).getByText("Execution runtime")).not.toBeNull();
  });
});

// ============================================================================
// Focus World preserves the active beat
// ============================================================================

describe("FlowApp Focus World", () => {
  test("enters and leaves Focus World without moving the active beat", () => {
    renderApp();

    seekBeatViaPath("first token");
    expect(playbackTime()).toMatch(/1500\s*ms/u);
    expect(causalPath().getAttribute("data-current-beat")).toBe("first-token");

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    fireEvent.click(within(twin).getByRole("button", { name: "CLI" }));

    const lens = screen.getByRole("region", { name: "Context Lens" });
    fireEvent.click(within(lens).getByRole("button", { name: "Focus World" }));

    expect(screen.getByRole("region", { name: "Focus World" })).not.toBeNull();
    expect(playbackTime()).toMatch(/1500\s*ms/u);
    expect(causalPath().getAttribute("data-current-beat")).toBe("first-token");

    // Escape closes the lens first, then leaves Focus World.
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.queryByRole("region", { name: "Focus World" })).toBeNull();
    expect(playbackTime()).toMatch(/1500\s*ms/u);
    expect(causalPath().getAttribute("data-current-beat")).toBe("first-token");
  });
});

// ============================================================================
// Exploration and resume restore the seeked beat
// ============================================================================

describe("FlowApp exploration and resume", () => {
  test("resume restores the exact seeked beat, selection, and focus", () => {
    renderApp();

    seekBeatViaPath("first token");
    expect(playbackTime()).toMatch(/1500\s*ms/u);

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    fireEvent.click(within(twin).getByRole("button", { name: "CLI" }));

    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    expect(screen.getByRole("button", { name: "Resume lesson" })).not.toBeNull();

    // Change the selection while exploring; resume must restore the authored one.
    fireEvent.click(within(twin).getByRole("button", { name: "Runtime" }));
    expect(
      within(twin).getByRole("button", { name: "Runtime" }).getAttribute("data-selected"),
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Resume lesson" }));

    expect(screen.getByRole("button", { name: "Explore" })).not.toBeNull();
    expect(playbackTime()).toMatch(/1500\s*ms/u);
    expect(
      within(twin).getByRole("button", { name: "CLI" }).getAttribute("data-selected"),
    ).toBe("true");
    expect(
      within(twin).getByRole("button", { name: "CLI" }).getAttribute("data-focused"),
    ).toBe("true");
    expect(
      within(twin).getByRole("button", { name: "Runtime" }).getAttribute("data-selected"),
    ).toBe("false");
  });
});

// ============================================================================
// Canvas-unavailable fallback retains immersive chrome
// ============================================================================

describe("FlowApp fallback chrome", () => {
  test("Canvas failure falls back to SVG while keeping HUD, commands, twin, and causal path", () => {
    HTMLCanvasElement.prototype.getContext = vi.fn(
      () => null,
    ) as typeof HTMLCanvasElement.prototype.getContext;

    renderApp({ forceSvgFallback: false });

    expect(
      screen.getByRole("region", { name: "Scene field" }).getAttribute("data-backend"),
    ).toBe("svg");
    expect(screen.getByRole("region", { name: "Semantic outline" })).not.toBeNull();
    expect(screen.getByRole("region", { name: "Immersive controls" })).not.toBeNull();
    expect(screen.getByRole("navigation", { name: "Causal path" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Open commands" })).not.toBeNull();

    // Commands stay discoverable through the fallback shell.
    expect(openCommandPalette()).not.toBeNull();
  });
});

// ============================================================================
// Fullscreen denial announcement
// ============================================================================

describe("FlowApp fullscreen denial", () => {
  test("announces a recoverable message when the browser blocks fullscreen", async () => {
    const denyingAdapter: FullscreenAdapter = {
      supported: () => true,
      active: () => false,
      enter: () => Promise.reject(new Error("blocked by permissions policy")),
      exit: () => Promise.resolve(),
    };
    renderApp({ fullscreenAdapter: denyingAdapter });

    fireEvent.click(screen.getByRole("button", { name: "Enter fullscreen" }));

    const announcement = await screen.findByText(FULLSCREEN_DENIED_MESSAGE);
    expect(announcement).not.toBeNull();
    // The layout is unchanged: the shell stays windowed and recoverable.
    expect(screen.getByRole("main").getAttribute("data-fullscreen")).toBe("windowed");
    expect(screen.getByRole("button", { name: "Enter fullscreen" })).not.toBeNull();
  });
});

// ============================================================================
// HUD policy never hides captions or focused controls
// ============================================================================

describe("FlowApp HUD and captions", () => {
  test("quiet HUD keeps captions and controls while playing", () => {
    renderApp();

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    const main = screen.getByRole("main");
    expect(main.getAttribute("data-hud")).toBe("quiet");

    const subtitles = screen.getByRole("region", { name: "Subtitles" });
    expect(subtitles.getAttribute("data-cue-id")).toBe("execution:narration");
    const openCommands = screen.getByRole("button", { name: "Open commands" });
    expect(openCommands.getAttribute("aria-hidden")).not.toBe("true");
    expect(getComputedStyle(openCommands).display).not.toBe("none");
  });

  test("focusing a control restores present HUD even while playing", () => {
    renderApp();

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    const main = screen.getByRole("main");
    expect(main.getAttribute("data-hud")).toBe("quiet");

    act(() => {
      screen.getByRole("button", { name: "Open commands" }).focus();
    });

    expect(main.getAttribute("data-hud")).toBe("present");
  });

  test("hidden HUD after inactivity still keeps captions and reachable controls", () => {
    vi.useFakeTimers();
    renderApp();

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      vi.advanceTimersByTime(3_000);
    });

    const main = screen.getByRole("main");
    expect(main.getAttribute("data-hud")).toBe("hidden");
    expect(
      screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id"),
    ).toBe("execution:narration");
    const openCommands = screen.getByRole("button", { name: "Open commands" });
    expect(openCommands.getAttribute("aria-hidden")).not.toBe("true");
    expect(getComputedStyle(openCommands).display).not.toBe("none");
  });
});
