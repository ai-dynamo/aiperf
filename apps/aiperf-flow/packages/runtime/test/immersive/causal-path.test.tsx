// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { SceneIr } from "@aiperf/flow-schema";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { CausalPath } from "../../src/immersive/causal-path.js";
import { projectCausalBeats, type CausalBeat } from "../../src/causal-replay.js";

afterEach(cleanup);

const sourceMap = {
  source: "request-lifecycle.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

// Authored request-lifecycle beats projected through the real scene projection
// so the control operates over frozen, production-shaped CausalBeat values.
function lifecycleBeats(): readonly CausalBeat[] {
  const scene: SceneIr = {
    id: "request-lifecycle",
    title: "Request lifecycle",
    summary: "A request moves from arrival to first token.",
    roots: [],
    camera: [],
    timeline: [
      { id: "arrival", at: 0, duration: 500, target: "client", action: "Arrival", sourceMap },
      { id: "admission", at: 1000, duration: 1000, target: "router", action: "Admission", sourceMap },
      { id: "first-token", at: 3000, duration: 500, target: "worker", action: "First token", sourceMap },
    ],
    narration: "Narration",
    interactions: [],
    responsive: [],
    accessibility: { label: "Request lifecycle", readingOrder: [] },
    fallback: "Fallback",
    sourceMap,
  };
  return projectCausalBeats(scene);
}

function beatButton(label: string): HTMLButtonElement {
  return screen.getByRole("button", { name: label }) as HTMLButtonElement;
}

describe("CausalPath rendering", () => {
  test("renders a labelled navigation control listing every authored beat", () => {
    render(<CausalPath beats={lifecycleBeats()} onSeek={vi.fn()} timeMs={0} />);

    const nav = screen.getByRole("navigation", { name: "Causal path" });
    expect(nav.getAttribute("data-beat-count")).toBe("3");

    const list = within(nav).getByRole("list", { name: "Causal beats" });
    const buttons = within(list).getAllByRole("button");
    expect(buttons.map((button) => button.textContent)).toEqual([
      "Arrival",
      "Admission",
      "First token",
    ]);
  });

  test("is a keyboard navigation control rather than a range scrubber", () => {
    const { container } = render(
      <CausalPath beats={lifecycleBeats()} onSeek={vi.fn()} timeMs={0} />,
    );

    expect(container.querySelector('input[type="range"]')).toBeNull();
  });

  test("marks the active beat as the current step at the given time", () => {
    render(<CausalPath beats={lifecycleBeats()} onSeek={vi.fn()} timeMs={1500} />);

    const nav = screen.getByRole("navigation", { name: "Causal path" });
    expect(nav.getAttribute("data-current-beat")).toBe("admission");
    expect(beatButton("Admission").getAttribute("aria-current")).toBe("step");
    expect(beatButton("Arrival").getAttribute("aria-current")).toBeNull();
  });

  test("derives complete, active, and future beat states from the clock", () => {
    render(<CausalPath beats={lifecycleBeats()} onSeek={vi.fn()} timeMs={1500} />);

    expect(beatButton("Arrival").getAttribute("data-state")).toBe("complete");
    expect(beatButton("Admission").getAttribute("data-state")).toBe("active");
    expect(beatButton("First token").getAttribute("data-state")).toBe("future");
  });
});

describe("CausalPath roving focus", () => {
  test("gives exactly one beat a tab stop anchored to the active beat", () => {
    render(<CausalPath beats={lifecycleBeats()} onSeek={vi.fn()} timeMs={1500} />);

    const tabbable = screen
      .getAllByRole("button")
      .filter((button) => button.getAttribute("tabindex") === "0");
    expect(tabbable).toHaveLength(1);
    expect(tabbable[0]?.getAttribute("data-beat-id")).toBe("admission");
  });
});

describe("CausalPath keyboard traversal", () => {
  test("ArrowRight seeks the next beat at its exact integer time", () => {
    const onSeek = vi.fn();
    render(<CausalPath beats={lifecycleBeats()} onSeek={onSeek} timeMs={1500} />);

    fireEvent.keyDown(beatButton("Admission"), { key: "ArrowRight" });

    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenCalledWith(3000, "first-token");
    expect(document.activeElement).toBe(beatButton("First token"));
  });

  test("ArrowLeft seeks the previous beat", () => {
    const onSeek = vi.fn();
    render(<CausalPath beats={lifecycleBeats()} onSeek={onSeek} timeMs={1500} />);

    fireEvent.keyDown(beatButton("Admission"), { key: "ArrowLeft" });

    expect(onSeek).toHaveBeenCalledWith(0, "arrival");
  });

  test("Home and End seek the first and last beats", () => {
    const onSeek = vi.fn();
    render(<CausalPath beats={lifecycleBeats()} onSeek={onSeek} timeMs={1500} />);

    fireEvent.keyDown(beatButton("Admission"), { key: "End" });
    expect(onSeek).toHaveBeenLastCalledWith(3000, "first-token");

    fireEvent.keyDown(beatButton("First token"), { key: "Home" });
    expect(onSeek).toHaveBeenLastCalledWith(0, "arrival");
  });

  test("does not wrap or seek past the first beat", () => {
    const onSeek = vi.fn();
    render(<CausalPath beats={lifecycleBeats()} onSeek={onSeek} timeMs={0} />);

    fireEvent.keyDown(beatButton("Arrival"), { key: "ArrowLeft" });

    expect(onSeek).not.toHaveBeenCalled();
  });
});

describe("CausalPath pointer seeks", () => {
  test("clicking a beat seeks its authored integer time", () => {
    const onSeek = vi.fn();
    render(<CausalPath beats={lifecycleBeats()} onSeek={onSeek} timeMs={0} />);

    fireEvent.click(beatButton("First token"));

    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenCalledWith(3000, "first-token");
    expect(Number.isSafeInteger(onSeek.mock.calls[0]?.[0])).toBe(true);
  });
});
