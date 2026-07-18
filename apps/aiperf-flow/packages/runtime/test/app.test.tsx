// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

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
import type { Clock } from "../src/player.js";

const sourceMap = {
  source: "request-flow.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

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

function foundationScene(overrides: Partial<SceneIr> = {}): SceneIr {
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
        accessibility: {
          label: "CLI",
          description: "Command-line process",
        },
        fallback: "CLI",
        sourceMap,
      },
      {
        kind: "rect",
        id: "runtime",
        geometry: { x: 300, y: 100, width: 180, height: 72 },
        style: { fill: "#244a35" },
        accessibility: {
          label: "Runtime",
          description: "Execution runtime",
        },
        fallback: "Runtime",
        sourceMap,
      },
      {
        kind: "text",
        id: "label",
        geometry: { x: 60, y: 130, width: 120, height: 24 },
        style: { fill: "#ffffff" },
        text: "CLI",
        accessibility: {
          label: "CLI label",
          description: "Label inside the CLI node",
        },
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
        accessibility: {
          label: "Spawn connection",
          description: "CLI starts the runtime",
        },
        fallback: "CLI starts Runtime",
        sourceMap,
      },
    ],
    camera: [],
    timeline: [
      {
        id: "reveal-cli",
        at: 0,
        duration: 400,
        action: "reveal",
        target: "cli",
        sourceMap,
      },
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

describe("FlowApp cinematic mount", () => {
  test("evaluates scene IR once and mounts SVG display-list path without SceneRenderer", () => {
    const { container } = render(
      <FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />,
    );

    const stage = screen.getByRole("region", { name: "Scene field" });
    expect(stage.getAttribute("data-backend")).toBe("svg");

    const svg = container.querySelector("svg.aiperf-flow__svg-fallback");
    expect(svg).not.toBeNull();
    expect(container.querySelector('[data-draw-command-id="cli"]')).not.toBeNull();
    expect(container.querySelector('[data-draw-command-id="runtime"]')).not.toBeNull();

    // Foundation SceneRenderer used role="img" + viewBox on .aiperf-flow__stage.
    expect(container.querySelector('svg.aiperf-flow__stage[role="img"]')).toBeNull();
    expect(container.querySelector("svg[viewBox='0 0 640 360']")).toBeNull();
  });

  test("always mounts a compact semantic twin that stays reachable", () => {
    render(<FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />);

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    expect(twin.getAttribute("data-scene-id")).toBe("execution");
    expect(twin.getAttribute("data-compact")).toBe("true");
    expect(twin.className).toContain("aiperf-flow__semantic-twin--compact");
    expect(twin.getAttribute("aria-hidden")).not.toBe("true");
    expect(getComputedStyle(twin).display).not.toBe("none");

    const entityButtons = within(twin).getAllByRole("button");
    expect(entityButtons.map((node) => node.getAttribute("data-entity-id"))).toEqual([
      "cli",
      "runtime",
    ]);

    const relation = within(twin)
      .getByLabelText("Relations")
      .querySelector('[data-relation-id="spawn"]');
    expect(relation).not.toBeNull();
    expect(relation?.textContent).toContain("Spawn connection");
    expect(relation?.outerHTML).toMatch(/data-from="cli"/);
    expect(relation?.outerHTML).toMatch(/data-to="runtime"/);
  });

  test("keeps shell chrome, playback controls, transcript skip link, and exploration actions", () => {
    render(<FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />);

    expect(screen.getByText(/Scene 1 of 1/u)).not.toBeNull();
    expect(screen.getByRole("heading", { name: "Execution boundary" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Play" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Explore" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Open commands" })).not.toBeNull();
    expect(screen.getByRole("navigation", { name: "Causal path" })).not.toBeNull();
    expect(screen.getByRole("link", { name: "Skip to transcript" })).not.toBeNull();
    expect(screen.getByRole("heading", { name: "Transcript" })).not.toBeNull();
    expect(
      screen.getAllByText("The CLI starts a fresh runtime and dispatches work."),
    ).not.toHaveLength(0);
  });

  test("prefers Canvas when a 2D context is available and still mounts the twin", () => {
    mockCanvas2d();
    const { container } = render(<FlowApp flow={flowWith([foundationScene()])} />);

    const stage = screen.getByRole("region", { name: "Scene field" });
    expect(stage.getAttribute("data-backend")).toBe("canvas");
    expect(container.querySelector("canvas.aiperf-flow__canvas")).not.toBeNull();
    expect(container.querySelector("svg.aiperf-flow__svg-fallback")).toBeNull();
    expect(screen.getByRole("region", { name: "Semantic outline" })).not.toBeNull();
  });

  test("falls back to SVG when Canvas is forced off", () => {
    mockCanvas2d();
    const { container } = render(
      <FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />,
    );

    expect(screen.getByRole("region", { name: "Scene field" }).getAttribute("data-backend")).toBe(
      "svg",
    );
    expect(container.querySelector("svg.aiperf-flow__svg-fallback")).not.toBeNull();
    expect(container.querySelector("canvas.aiperf-flow__canvas")).toBeNull();
  });

  test("synchronizes twin selection from SVG entity activation", () => {
    render(<FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />);

    const svgEntity = document.querySelector('[data-entity-id="runtime"]');
    expect(svgEntity).not.toBeNull();
    fireEvent.click(svgEntity as Element);

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    const runtimeButton = within(twin).getByRole("button", { name: "Runtime" });
    expect(runtimeButton.getAttribute("data-selected")).toBe("true");
    expect(runtimeButton.getAttribute("aria-selected")).toBe("true");
    expect(runtimeButton.getAttribute("data-focused")).toBe("true");
  });

  test("canvas pointer hits select the matching twin entity through the focus coordinator", () => {
    mockCanvas2d();
    const { container } = render(<FlowApp flow={flowWith([foundationScene()])} />);

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

    // CLI hit region is at (40,100)-(200,172) within paint bounds covering the scene.
    fireEvent.pointerDown(canvas as Element, { clientX: 80, clientY: 130 });

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    const cliButton = within(twin).getByRole("button", { name: "CLI" });
    expect(cliButton.getAttribute("data-selected")).toBe("true");
    expect(cliButton.getAttribute("data-focused")).toBe("true");
  });

  test("twin keyboard activation selects the visual entity and opens Context Lens when exploring", () => {
    render(<FlowApp flow={flowWith([foundationScene()])} forceSvgFallback />);

    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    expect(screen.getByRole("button", { name: "Resume lesson" })).not.toBeNull();

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    const runtimeButton = within(twin).getByRole("button", { name: "Runtime" });
    fireEvent.click(runtimeButton);

    expect(runtimeButton.getAttribute("data-selected")).toBe("true");
    expect(screen.getByRole("region", { name: "Context Lens" })).not.toBeNull();
    expect(
      within(screen.getByRole("region", { name: "Context Lens" })).getByText(
        "Execution runtime",
      ),
    ).not.toBeNull();
  });

  test("exploration pause freezes evaluation time and resume restores focus", () => {
    const clock = new VirtualClock();
    render(
      <FlowApp
        clock={clock}
        flow={flowWith([foundationScene()])}
        forceSvgFallback
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      clock.advanceMs(150);
    });
    expect(screen.getByRole("status", { name: "Playback time" }).textContent).toMatch(
      /150\s*ms/,
    );

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    fireEvent.click(within(twin).getByRole("button", { name: "CLI" }));
    fireEvent.click(screen.getByRole("button", { name: "Explore" }));

    fireEvent.click(within(twin).getByRole("button", { name: "Runtime" }));
    expect(
      within(twin).getByRole("button", { name: "Runtime" }).getAttribute("data-selected"),
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Resume lesson" }));
    expect(screen.getByRole("button", { name: "Explore" })).not.toBeNull();
    expect(screen.getByRole("status", { name: "Playback time" }).textContent).toMatch(
      /150\s*ms/,
    );
    expect(
      within(twin).getByRole("button", { name: "CLI" }).getAttribute("data-selected"),
    ).toBe("true");
    expect(
      within(twin).getByRole("button", { name: "CLI" }).getAttribute("data-focused"),
    ).toBe("true");
  });

  test("scene navigation resets player time", () => {
    const clock = new VirtualClock();
    const second = foundationScene({
      id: "results",
      title: "Results",
      narration: "Results scene transcript.",
    });
    render(
      <FlowApp
        clock={clock}
        flow={flowWith([foundationScene(), second])}
        forceSvgFallback
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      clock.advanceMs(200);
    });
    expect(screen.getByRole("status", { name: "Playback time" }).textContent).toMatch(
      /200\s*ms/,
    );

    fireEvent.click(screen.getByRole("button", { name: "Next scene" }));
    expect(screen.getByRole("heading", { name: "Results" })).not.toBeNull();
    expect(screen.getByRole("status", { name: "Playback time" }).textContent).toMatch(
      /^0\s*ms$/,
    );
    expect(screen.getByRole("region", { name: "Semantic outline" }).getAttribute("data-scene-id")).toBe(
      "results",
    );
  });

  test("preserves fallback chrome for an invalid scene without mounting the twin", () => {
    const invalid = {
      ...foundationScene(),
      roots: null,
      summary: "Execution scene summary",
      fallback: "Execution scene text fallback",
    };
    const next = foundationScene({
      id: "results",
      title: "Results",
    });

    render(
      <FlowApp
        flow={flowWith([invalid as unknown as SceneIr, next])}
        forceSvgFallback
      />,
    );

    expect(screen.getByText("Execution scene summary")).not.toBeNull();
    expect(screen.getByText("Execution scene text fallback")).not.toBeNull();
    expect(screen.queryByRole("region", { name: "Semantic outline" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Scene field" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Next scene" }));
    expect(screen.getByRole("heading", { name: "Results" })).not.toBeNull();
    expect(screen.getByRole("region", { name: "Semantic outline" })).not.toBeNull();
  });
});
