// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  AudioConsentModal,
  loadAudioConsentChoice,
  saveAudioConsentChoice,
} from "../../src/narrative/audio-consent-modal.js";

afterEach(() => {
  cleanup();
  sessionStorage.clear();
});

describe("audio consent visit persistence", () => {
  test("remembers a choice for the current tab visit", () => {
    expect(loadAudioConsentChoice()).toBeNull();

    saveAudioConsentChoice("with-audio");

    expect(loadAudioConsentChoice()).toBe("with-audio");
  });

  test("ignores invalid stored values", () => {
    sessionStorage.setItem("aiperf-flow:audio-consent", "unexpected");

    expect(loadAudioConsentChoice()).toBeNull();
  });
});

describe("AudioConsentModal", () => {
  test("asks whether to play with or without audio", () => {
    const onChoose = vi.fn();
    render(<AudioConsentModal open onChoose={onChoose} />);

    expect(
      screen.getByRole("dialog", { name: "Audio preference" }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Play with audio" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Play without audio" }),
    ).toBeTruthy();
  });

  test("reports with-audio and without-audio choices", () => {
    const onChoose = vi.fn();
    render(<AudioConsentModal open onChoose={onChoose} />);

    fireEvent.click(screen.getByRole("button", { name: "Play with audio" }));
    expect(onChoose).toHaveBeenCalledWith("with-audio");

    fireEvent.click(
      screen.getByRole("button", { name: "Play without audio" }),
    );
    expect(onChoose).toHaveBeenCalledWith("without-audio");
  });

  test("renders nothing when closed", () => {
    render(<AudioConsentModal open={false} onChoose={() => undefined} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
