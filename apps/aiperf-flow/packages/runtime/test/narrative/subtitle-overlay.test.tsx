// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  SubtitleOverlay,
  type SubtitleState,
} from "../../src/narrative/subtitle-overlay.js";

afterEach(cleanup);

const activeState: SubtitleState = {
  enabled: true,
  activeCue: {
    id: "cue-admission",
    speaker: "Narrator",
    text: "The worker admits the request.",
  },
};

describe("SubtitleOverlay", () => {
  test("renders the active cue in an HTML caption safe area", () => {
    const { container } = render(
      <SubtitleOverlay onEnabledChange={() => {}} state={activeState} />,
    );

    const overlay = screen.getByRole("region", { name: "Subtitles" });
    const cue = container.querySelector(".aiperf-flow__subtitle-cue");

    expect(overlay.getAttribute("data-cue-id")).toBe("cue-admission");
    expect(cue?.textContent).toContain("Narrator");
    expect(cue?.textContent).toContain("The worker admits the request.");
    expect(container.querySelector("canvas")).toBeNull();
    expect(container.querySelector("svg")).toBeNull();
  });

  test("exposes a controlled subtitle on and off contract", () => {
    const onEnabledChange = vi.fn();
    const { rerender } = render(
      <SubtitleOverlay
        onEnabledChange={onEnabledChange}
        state={activeState}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Turn subtitles off" }));
    expect(onEnabledChange).toHaveBeenCalledWith(false);

    rerender(
      <SubtitleOverlay
        onEnabledChange={onEnabledChange}
        state={{ ...activeState, enabled: false }}
      />,
    );

    const control = screen.getByRole("button", {
      name: "Turn subtitles on",
    });
    expect(control.getAttribute("aria-pressed")).toBe("false");
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue"),
    ).toBeNull();
    expect(screen.getByRole("status").textContent).toBe("");
  });

  test("announces each cue identity once without duplicating visible text", () => {
    const { container, rerender } = render(
      <SubtitleOverlay onEnabledChange={() => {}} state={activeState} />,
    );
    const liveRegion = screen.getByRole("status");

    expect(liveRegion.getAttribute("aria-live")).toBe("polite");
    expect(liveRegion.getAttribute("aria-atomic")).toBe("true");
    expect(liveRegion.textContent).toBe("Narrator: The worker admits the request.");
    expect(
      container.querySelector(".aiperf-flow__subtitle-cue")?.getAttribute(
        "aria-hidden",
      ),
    ).toBe("true");

    rerender(
      <SubtitleOverlay
        onEnabledChange={() => {}}
        state={{
          ...activeState,
          activeCue: {
            ...activeState.activeCue!,
            text: "Updated visual wording.",
          },
        }}
      />,
    );

    expect(
      container.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toContain("Updated visual wording.");
    expect(liveRegion.textContent).toBe("Narrator: The worker admits the request.");

    rerender(
      <SubtitleOverlay
        onEnabledChange={() => {}}
        state={{
          enabled: true,
          activeCue: {
            id: "cue-observation",
            text: "The observer records terminal metrics.",
          },
        }}
      />,
    );

    expect(liveRegion.textContent).toBe(
      "The observer records terminal metrics.",
    );
  });

  test("exposes explicit contrast and reduced-motion presentation state", () => {
    render(
      <SubtitleOverlay
        contrast="high"
        onEnabledChange={() => {}}
        reducedMotion
        state={activeState}
      />,
    );

    const overlay = screen.getByRole("region", { name: "Subtitles" });
    expect(overlay.getAttribute("data-contrast")).toBe("high");
    expect(overlay.getAttribute("data-reduced-motion")).toBe("true");
  });

  test("keeps the control available when no cue is active", () => {
    render(
      <SubtitleOverlay
        onEnabledChange={() => {}}
        state={{ enabled: true, activeCue: null }}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Turn subtitles off" }),
    ).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("");
  });
});
