/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStepSimulator } from "./useStepSimulator.js";

describe("useStepSimulator", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts at the first step", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    expect(result.current.index).toBe(0);
    expect(result.current.current).toBe("a");
    expect(result.current.total).toBe(3);
    expect(result.current.isFirst).toBe(true);
    expect(result.current.isLast).toBe(false);
    expect(result.current.isPlaying).toBe(false);
  });

  it("advances with next()", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    act(() => {
      result.current.next();
    });
    expect(result.current.index).toBe(1);
    expect(result.current.current).toBe("b");
  });

  it("does not advance past the last step", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b"]));
    act(() => {
      result.current.next();
    });
    act(() => {
      result.current.next();
    });
    expect(result.current.index).toBe(1);
    expect(result.current.current).toBe("b");
    expect(result.current.isLast).toBe(true);
  });

  it("moves back with back()", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    act(() => {
      result.current.next();
      result.current.next();
    });
    expect(result.current.index).toBe(2);
    act(() => {
      result.current.back();
    });
    expect(result.current.index).toBe(1);
  });

  it("does not move back before the first step", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b"]));
    act(() => {
      result.current.back();
    });
    expect(result.current.index).toBe(0);
    expect(result.current.isFirst).toBe(true);
  });

  it("resets to the first step and stops playback", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    act(() => {
      result.current.next();
      result.current.togglePlay();
    });
    expect(result.current.isPlaying).toBe(true);
    act(() => {
      result.current.reset();
    });
    expect(result.current.index).toBe(0);
    expect(result.current.isPlaying).toBe(false);
  });

  it("toggles isPlaying", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    act(() => {
      result.current.togglePlay();
    });
    expect(result.current.isPlaying).toBe(true);
    act(() => {
      result.current.togglePlay();
    });
    expect(result.current.isPlaying).toBe(false);
  });

  it("auto-advances on a timer while playing, using the default interval", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"]));
    act(() => {
      result.current.togglePlay();
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current.index).toBe(1);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current.index).toBe(2);
  });

  it("honors a custom autoPlayMs interval", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b", "c"], { autoPlayMs: 250 }));
    act(() => {
      result.current.togglePlay();
    });
    act(() => {
      vi.advanceTimersByTime(250);
    });
    expect(result.current.index).toBe(1);
  });

  it("stops playback automatically upon reaching the last step", () => {
    const { result } = renderHook(() => useStepSimulator(["a", "b"]));
    act(() => {
      result.current.togglePlay();
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current.index).toBe(1);
    expect(result.current.isLast).toBe(true);
    expect(result.current.isPlaying).toBe(false);

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(result.current.index).toBe(1);
  });

  it("returns undefined current for an empty steps array", () => {
    const { result } = renderHook(() => useStepSimulator([]));
    expect(result.current.current).toBeUndefined();
    expect(result.current.total).toBe(0);
    expect(result.current.isFirst).toBe(true);
    expect(result.current.isLast).toBe(true);
  });
});
