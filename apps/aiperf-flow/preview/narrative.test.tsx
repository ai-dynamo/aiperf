// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  App,
  sceneNarrativeCues,
  splitNarrationClauses,
  unlockPreviewSpeech,
} from "./App";
import { previewScene } from "./fixture";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

beforeEach(() => {
  sessionStorage.clear();
  Object.defineProperty(window, "matchMedia", {
    writable: true,
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

  class FakeUtterance {
    rate = 1;
    volume = 1;
    voice: SpeechSynthesisVoice | null = null;
    constructor(readonly text: string) {}
  }

  const synthesis = {
    speaking: false,
    pending: false,
    paused: false,
    onvoiceschanged: null,
    getVoices: () => [],
    speak: vi.fn(),
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  };

  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    writable: true,
    value: synthesis,
  });
  Object.defineProperty(window, "SpeechSynthesisUtterance", {
    configurable: true,
    writable: true,
    value: FakeUtterance,
  });

  HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
});

describe("preview narrative adapters", () => {
  test("splits the execution narration into synchronized clauses", () => {
    const scene = previewScene();
    const clauses = splitNarrationClauses(scene.narration);
    const cues = sceneNarrativeCues(scene);

    expect(clauses.length).toBeGreaterThan(1);
    expect(cues.length).toBe(clauses.length);
    expect(cues[0]?.atMs).toBe(0);
    expect(cues.at(-1)?.atMs).toBeGreaterThanOrEqual(4000);
    expect(cues.map((cue) => cue.spokenText).join(" ")).toContain(
      "stable seam",
    );
  });

  test("unlocks speech synthesis when the platform is present", () => {
    expect(unlockPreviewSpeech()).toBe(true);
    expect(window.speechSynthesis.cancel).toHaveBeenCalled();
  });
});

describe("preview App narration vertical slice", () => {
  function transportStatus(): HTMLElement {
    const statuses = screen.getAllByRole("status");
    const transport = statuses.find((node) =>
      /Scene \d+ of \d+/i.test(node.textContent ?? ""),
    );
    if (transport === undefined) {
      throw new Error("Transport status not found");
    }
    return transport;
  }

  function chooseAudio(choice: "with-audio" | "without-audio" = "with-audio"): void {
    fireEvent.click(
      screen.getByRole("button", {
        name: choice === "with-audio" ? "Play with audio" : "Play without audio",
      }),
    );
  }

  test("asks for audio preference before playback starts", () => {
    render(<App />);

    expect(
      screen.getByRole("dialog", { name: "Audio preference" }),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Play scene" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    chooseAudio("with-audio");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Mute narrator" }).getAttribute(
        "data-narrator-mode",
      ),
    ).toBe("on");
  });

  test("renders a visible subtitle overlay for the execution scene", () => {
    render(<App />);
    chooseAudio("without-audio");

    expect(screen.getByRole("region", { name: "Subtitles" })).toBeTruthy();
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toMatch(/runtime/i);
    expect(
      screen.getByRole("button", { name: "Turn subtitles off" }),
    ).toBeTruthy();
  });

  test("cycles narrator on, mute, and off from the canvas control", () => {
    render(<App />);
    chooseAudio("with-audio");

    const narrator = screen.getByRole("button", { name: "Mute narrator" });
    expect(narrator.getAttribute("data-narrator-mode")).toBe("on");

    fireEvent.click(narrator);
    expect(
      screen.getByRole("button", { name: "Turn narrator off" }).getAttribute(
        "data-narrator-mode",
      ),
    ).toBe("muted");
    expect(transportStatus().textContent).toMatch(/narrator muted/i);

    fireEvent.click(screen.getByRole("button", { name: "Turn narrator off" }));
    expect(
      screen.getByRole("button", { name: "Turn narrator on" }).getAttribute(
        "data-narrator-mode",
      ),
    ).toBe("off");
  });

  test("pauses narration with exploration and resumes the exact beat", () => {
    render(<App />);
    chooseAudio("with-audio");

    fireEvent.click(screen.getByRole("button", { name: "Select tool" }));
    expect(transportStatus().textContent).toMatch(/exploring/i);
    expect(
      screen.getByRole("button", { name: "Resume lesson from current beat" }),
    ).toBeTruthy();

    const pausedProgress = transportStatus().textContent;
    fireEvent.click(
      screen.getByRole("button", { name: "Resume lesson from current beat" }),
    );

    expect(transportStatus().textContent).not.toMatch(/exploring/i);
    expect(pausedProgress).toMatch(/\d+\.\d+s/);
  });

  test("stops narration when the preview shell unmounts", () => {
    const cancel = window.speechSynthesis.cancel as ReturnType<typeof vi.fn>;
    cancel.mockClear();
    const { unmount } = render(<App />);
    chooseAudio("with-audio");
    unmount();
    expect(cancel.mock.calls.length).toBeGreaterThan(0);
  });
});
