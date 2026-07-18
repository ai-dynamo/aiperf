// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { sceneNarrativeCues } from "../packages/runtime/src/narrative/scene-cues";

import { App, unlockPreviewSpeech } from "./App";
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
  test("uses the shared runtime projection for authored narration", () => {
    const scene = previewScene();
    const cues = sceneNarrativeCues(scene);

    expect(cues).toHaveLength(2);
    expect(cues[0]?.atMs).toBe(0);
    expect(cues.at(-1)?.spokenText).toContain("resolution");
  });

  test("unlocks speech synthesis when the platform is present", async () => {
    await expect(unlockPreviewSpeech()).resolves.toBe(true);
    expect(window.speechSynthesis.cancel).toHaveBeenCalled();
  });
});

describe("preview App narration vertical slice", () => {
  function playbackTime(): HTMLElement {
    return screen.getByRole("status", { name: "Playback time" });
  }

  function runCommand(name: string): void {
    fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
    fireEvent.click(
      screen.getByRole("option", { name: new RegExp(`${name}$`, "iu") }),
    );
  }

  function chooseAudio(choice: "with-audio" | "without-audio" = "with-audio"): void {
    fireEvent.click(
      screen.getByRole("button", {
        name: choice === "with-audio" ? "Play with audio" : "Play without audio",
      }),
    );
  }

  async function chooseAudioAndWait(
    choice: "with-audio" | "without-audio" = "with-audio",
  ): Promise<void> {
    chooseAudio(choice);
    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    });
  }

  test("asks for audio preference before playback starts", async () => {
    render(<App />);

    expect(
      screen.getByRole("dialog", { name: "Audio preference" }),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Play" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await chooseAudioAndWait("with-audio");
    fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
    expect(
      screen.getByRole("option", { name: /Mute narration$/iu }),
    ).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
  });

  test("renders a visible subtitle overlay for the request investigation", async () => {
    render(<App />);
    await chooseAudioAndWait("without-audio");

    expect(screen.getByRole("region", { name: "Subtitles" })).toBeTruthy();
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toMatch(/R-017/i);
    expect(
      screen.getByRole("button", { name: "Turn subtitles off" }),
    ).toBeTruthy();
  });

  test("mutes and unmutes narration through shared commands", async () => {
    render(<App />);
    await chooseAudioAndWait("with-audio");

    runCommand("Mute narration");
    fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
    expect(
      screen.getByRole("option", { name: /Unmute narration$/iu }),
    ).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    runCommand("Unmute narration");
  });

  test("pauses narration with exploration and resumes the exact beat", async () => {
    render(<App />);
    await chooseAudioAndWait("with-audio");

    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    expect(
      screen.getByRole("button", { name: "Resume lesson" }),
    ).toBeTruthy();

    const pausedProgress = playbackTime().textContent;
    expect(playbackTime().textContent).toBe(pausedProgress);
    fireEvent.click(screen.getByRole("button", { name: "Resume lesson" }));

    expect(screen.getByRole("button", { name: "Explore" })).toBeTruthy();
  });

  test("stops narration when the preview shell unmounts", async () => {
    const cancel = window.speechSynthesis.cancel as ReturnType<typeof vi.fn>;
    cancel.mockClear();
    const { unmount } = render(<App />);
    await chooseAudioAndWait("with-audio");
    unmount();
    expect(cancel.mock.calls.length).toBeGreaterThan(0);
  });
});
