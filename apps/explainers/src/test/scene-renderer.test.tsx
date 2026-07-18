/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SceneRenderer,
  type SceneIrLike,
} from "../core/diagram/SceneRenderer";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

function minimalScene(): SceneIrLike {
  return {
    roots: [
      {
        id: "box",
        kind: "rect",
        geometry: { x: 80, y: 120, width: 160, height: 72 },
        style: { fill: "#3FA266" },
        accessibility: { label: "Coordinator" },
      },
    ],
    timeline: [
      {
        id: "enter-box",
        at: 0,
        duration: 400,
        action: "enter",
        target: "box",
      },
    ],
  };
}

describe("SceneRenderer", () => {
  it("renders a minimal scene with one core.rect and a timeline enter cue", () => {
    const { container } = render(
      <SceneRenderer scene={minimalScene()} playing={false} restartKey={0} />,
    );

    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("viewBox")).toBe("0 0 700 400");

    const node = container.querySelector('[data-flow-node-id="box"]');
    expect(node).not.toBeNull();
    expect(node?.querySelector("rect")).not.toBeNull();
    expect(node?.getAttribute("data-timeline-state")).toBe("hidden");
    expect(node?.getAttribute("aria-label")).toBe("Coordinator");
  });

  it("shows the final timeline frame when reducedMotion is set", () => {
    const { container } = render(
      <SceneRenderer
        scene={minimalScene()}
        playing={false}
        restartKey={0}
        reducedMotion
      />,
    );

    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
  });

  it("plays the timeline from the start when playing is true", () => {
    vi.useFakeTimers();
    const { container } = render(
      <SceneRenderer scene={minimalScene()} playing restartKey={0} />,
    );

    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("hidden");

    act(() => {
      vi.advanceTimersByTime(400);
    });

    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
  });

  it("restarts the timeline when restartKey changes", () => {
    vi.useFakeTimers();
    const { container, rerender } = render(
      <SceneRenderer scene={minimalScene()} playing restartKey={0} />,
    );

    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");

    rerender(<SceneRenderer scene={minimalScene()} playing restartKey={1} />);

    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("hidden");

    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
  });
});
