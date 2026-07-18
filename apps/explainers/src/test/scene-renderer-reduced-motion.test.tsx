/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SceneRenderer,
  type SceneIrLike,
} from "../core/diagram/SceneRenderer";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function delayedRevealScene(): SceneIrLike {
  return {
    roots: [
      {
        id: "box",
        kind: "rect",
        geometry: { x: 80, y: 120, width: 160, height: 72 },
        accessibility: { label: "Coordinator" },
      },
    ],
    timeline: [
      {
        id: "enter-box",
        at: 2_000,
        duration: 3_000,
        action: "enter",
        target: "box",
      },
    ],
  };
}

describe("SceneRenderer reducedMotion", () => {
  it("shows the final frame immediately without awaiting the timeline", () => {
    vi.useFakeTimers();
    const rafSpy = vi.spyOn(globalThis, "requestAnimationFrame");

    const { container } = render(
      <SceneRenderer
        scene={delayedRevealScene()}
        playing
        restartKey={0}
        reducedMotion
      />,
    );

    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
    expect(rafSpy).not.toHaveBeenCalled();

    // Still revealed after wall-clock time that would only cover part of the cue.
    vi.advanceTimersByTime(2_500);
    expect(
      container
        .querySelector('[data-flow-node-id="box"]')
        ?.getAttribute("data-timeline-state"),
    ).toBe("revealed");
  });
});
