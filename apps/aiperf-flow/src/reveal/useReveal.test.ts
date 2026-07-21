/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useReveal } from "./useReveal.js";

describe("useReveal", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reveals only the first id immediately", () => {
    const { result } = renderHook(() => useReveal(["a", "b", "c"], { stepMs: 100 }));
    expect(result.current).toEqual(new Set(["a"]));
  });

  it("reveals the next id after one step", () => {
    const { result } = renderHook(() => useReveal(["a", "b", "c"], { stepMs: 100 }));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(result.current).toEqual(new Set(["a", "b"]));
  });

  it("reveals all ids once enough steps have elapsed and stops there", () => {
    const { result } = renderHook(() => useReveal(["a", "b", "c"], { stepMs: 100 }));
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toEqual(new Set(["a", "b", "c"]));
  });

  it("reveals nothing for an empty order", () => {
    const { result } = renderHook(() => useReveal([], { stepMs: 100 }));
    expect(result.current).toEqual(new Set());
  });
});
