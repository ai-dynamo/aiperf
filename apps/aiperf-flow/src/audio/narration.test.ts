/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { estimateNarrationMs, speakNarration, speechRateForSpeed } from "./narration.js";

// jsdom provides no `speechSynthesis`, so these exercise the silent path that
// also runs in browsers without speech support.
describe("narration", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("scales estimated duration inversely with speed", () => {
    const text = "one two three four five six seven eight nine ten";
    expect(estimateNarrationMs(text, 2)).toBeLessThan(estimateNarrationMs(text, 1));
  });

  it("clamps speech rate into the range engines actually honor", () => {
    expect(speechRateForSpeed(10)).toBe(2);
    expect(speechRateForSpeed(0.01)).toBe(0.5);
  });

  it("completes after the estimated duration and walks the words", () => {
    const onComplete = vi.fn();
    const onWord = vi.fn();
    const text = "alpha bravo charlie";

    speakNarration(text, { useSpeech: false, onWord, onComplete });

    expect(onComplete).not.toHaveBeenCalled();
    vi.advanceTimersByTime(estimateNarrationMs(text));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onWord).toHaveBeenCalledWith(text.split(" ").length - 1);
  });

  it("does not complete after the returned cancel is called", () => {
    const onComplete = vi.fn();
    const text = "alpha bravo charlie";

    const cancel = speakNarration(text, { useSpeech: false, onComplete });
    cancel();
    vi.advanceTimersByTime(estimateNarrationMs(text) * 4);

    expect(onComplete).not.toHaveBeenCalled();
  });
});
