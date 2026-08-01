/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import { act, renderHook } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { estimateNarrationMs } from "./narration.js";
import { useNarratedDeck } from "./useNarratedDeck.js";

const NARRATIONS = ["first step spoken", "second step spoken", "third step spoken"];

function wrapper({ children }: { children: ReactNode }) {
  return <MemoryRouter>{children}</MemoryRouter>;
}

function renderDeck() {
  return renderHook(
    () => useNarratedDeck({ narrations: NARRATIONS, storagePrefix: "test-deck" }),
    { wrapper },
  );
}

/** Runs out the estimated narration for one step. */
function playOneStep() {
  act(() => {
    vi.advanceTimersByTime(estimateNarrationMs(NARRATIONS[0]) + 100);
  });
}

describe("useNarratedDeck", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("stays silent until the start gate is cleared", () => {
    const { result } = renderDeck();

    expect(result.current.started).toBe(false);
    expect(result.current.index).toBe(0);

    playOneStep();
    expect(result.current.index).toBe(0);
  });

  it("advances when a step's narration completes", () => {
    const { result } = renderDeck();

    act(() => result.current.begin(false));
    expect(result.current.playing).toBe(true);

    playOneStep();
    expect(result.current.index).toBe(1);

    playOneStep();
    expect(result.current.index).toBe(2);
  });

  it("stops playing at the last step instead of looping", () => {
    const { result } = renderDeck();

    act(() => result.current.begin(false));
    playOneStep();
    playOneStep();
    expect(result.current.index).toBe(NARRATIONS.length - 1);

    playOneStep();
    expect(result.current.index).toBe(NARRATIONS.length - 1);
    expect(result.current.playing).toBe(false);
  });

  it("halts advancement while paused", () => {
    const { result } = renderDeck();

    act(() => result.current.begin(false));
    act(() => result.current.togglePlayback());
    expect(result.current.playing).toBe(false);

    playOneStep();
    expect(result.current.index).toBe(0);
  });

  it("rewinds to the start when played from the last step", () => {
    const { result } = renderDeck();

    act(() => result.current.begin(false));
    act(() => result.current.goTo(NARRATIONS.length - 1));
    act(() => result.current.togglePlayback()); // pause
    act(() => result.current.togglePlayback()); // replay

    expect(result.current.index).toBe(0);
    expect(result.current.playing).toBe(true);
  });

  it("persists voice and speed but not the current step", () => {
    const first = renderDeck();
    act(() => first.result.current.begin(false));
    act(() => first.result.current.setSpeed(1.5));
    act(() => first.result.current.goTo(2));
    first.unmount();

    const second = renderDeck();
    expect(second.result.current.speed).toBe(1.5);
    expect(second.result.current.index).toBe(0);
    expect(second.result.current.started).toBe(false);
  });
});
