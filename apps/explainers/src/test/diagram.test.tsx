/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FlowArrow } from "../core/diagram/FlowArrow";
import { MotionSignal } from "../core/diagram/MotionSignal";
import { SceneBox } from "../core/diagram/SceneBox";

function mockMatchMedia(matches: boolean) {
  const mediaQueryList: MediaQueryList = {
    matches,
    media: "(prefers-reduced-motion: reduce)",
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(() => false),
  };

  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockImplementation((query: string) => {
      expect(query).toBe("(prefers-reduced-motion: reduce)");
      return mediaQueryList;
    }),
  );

  return mediaQueryList;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("shared diagram primitives", () => {
  it("renders a themed scene box with its title and detail", () => {
    const { container } = render(
      <svg>
        <SceneBox
          x={10}
          y={20}
          width={180}
          height={90}
          title="Coordinator"
          detail="partitions work"
          accent="yellow"
          data-testid="scene-box"
        />
      </svg>,
    );

    expect(screen.getByText("Coordinator")).toBeTruthy();
    expect(screen.getByText("partitions work")).toBeTruthy();
    expect(container.querySelector('[data-testid="scene-box"] rect')?.getAttribute("stroke")).toBe(
      "#F1B467",
    );
  });

  it("renders an arrow path with the requested marker", () => {
    const { container } = render(
      <svg>
        <FlowArrow d="M10 20 H90" markerId="deck-arrow" data-testid="flow-arrow" />
      </svg>,
    );

    const arrow = container.querySelector('[data-testid="flow-arrow"]');
    expect(arrow?.getAttribute("d")).toBe("M10 20 H90");
    expect(arrow?.getAttribute("marker-end")).toBe("url(#deck-arrow)");
  });

  it("renders a timed animated signal on the requested path", () => {
    mockMatchMedia(false);
    const { container } = render(
      <svg>
        <MotionSignal
          path="M20 30 H120"
          color="#9386F2"
          duration="3s"
          delay="750ms"
          data-testid="motion-signal"
        />
      </svg>,
    );

    const signal = container.querySelector('[data-testid="motion-signal"]');
    const motion = signal?.querySelector("animateMotion");
    expect(signal?.getAttribute("class")).toContain("motion-signal");
    expect(signal?.getAttribute("fill")).toBe("#9386F2");
    expect(signal?.getAttribute("aria-hidden")).toBe("true");
    expect(motion?.getAttribute("path")).toBe("M20 30 H120");
    expect(motion?.getAttribute("dur")).toBe("3s");
    expect(motion?.getAttribute("begin")).toBe("750ms");
  });

  it("suppresses SMIL animation when the user prefers reduced motion", () => {
    mockMatchMedia(true);
    const { container } = render(
      <svg>
        <MotionSignal
          path="M20 30 H120"
          color="#9386F2"
          duration="3s"
          delay="750ms"
          data-testid="motion-signal"
        />
      </svg>,
    );

    expect(container.querySelector('[data-testid="motion-signal"]')).toBeNull();
    expect(container.querySelector("animateMotion")).toBeNull();
    expect(container.querySelector("animate")).toBeNull();
  });
});
