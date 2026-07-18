// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for the injectable Fullscreen API boundary and enter/exit policy.
//
// Focuses on:
// - the browser adapter delegating to an injected platform and degrading to an
//   unsupported adapter when feature detection fails;
// - `resolveFullscreenState` reconciling authored state with the live element
//   (Escape collapses native to windowed, layout immersion is untouched);
// - enter/exit/toggle returning the next state plus a recoverable live-region
//   announcement, never throwing on denial.
//
// Out of scope: HUD visibility policy (see hud-policy.test.ts) and the
// immersive reducer (see immersive-state.test.ts).

import { describe, expect, test, vi } from "vitest";

import {
  createBrowserFullscreenAdapter,
  enterFullscreenMode,
  exitFullscreenMode,
  resolveFullscreenState,
  toggleFullscreenMode,
  FULLSCREEN_DENIED_MESSAGE,
  FULLSCREEN_EXIT_DENIED_MESSAGE,
  type FullscreenAdapter,
  type FullscreenPlatform,
} from "../src/fullscreen.js";
import type { FullscreenState } from "../src/immersive-state.js";

const element = {} as unknown as HTMLElement;

type AdapterOptions = Readonly<{
  supported?: boolean;
  active?: boolean;
  enter?: () => Promise<void>;
  exit?: () => Promise<void>;
}>;

function fakeAdapter(options: AdapterOptions = {}): Readonly<{
  adapter: FullscreenAdapter;
  enter: ReturnType<typeof vi.fn>;
  exit: ReturnType<typeof vi.fn>;
}> {
  const enter = vi.fn(options.enter ?? (async () => undefined));
  const exit = vi.fn(options.exit ?? (async () => undefined));
  const adapter: FullscreenAdapter = {
    supported: () => options.supported ?? true,
    active: () => options.active ?? false,
    enter,
    exit,
  };
  return { adapter, enter, exit };
}

function browserPlatform(
  options: Readonly<{ enabled?: boolean; element?: Element | null }> = {},
): Readonly<{
  platform: FullscreenPlatform;
  requestFullscreen: ReturnType<typeof vi.fn>;
  exitFullscreen: ReturnType<typeof vi.fn>;
}> {
  const requestFullscreen = vi.fn(async (_target: HTMLElement) => undefined);
  const exitFullscreen = vi.fn(async () => undefined);
  const platform: FullscreenPlatform = {
    fullscreenEnabled: options.enabled ?? true,
    fullscreenElement: options.element ?? null,
    requestFullscreen,
    exitFullscreen,
  };
  return { platform, requestFullscreen, exitFullscreen };
}

describe("createBrowserFullscreenAdapter", () => {
  test("delegates support, activity, and enter to the injected platform", async () => {
    const { platform, requestFullscreen } = browserPlatform();
    const adapter = createBrowserFullscreenAdapter(platform);

    expect(adapter.supported()).toBe(true);
    expect(adapter.active()).toBe(false);

    await adapter.enter(element);
    expect(requestFullscreen).toHaveBeenCalledWith(element);
  });

  test("reports active while the platform owns a fullscreen element", () => {
    const { platform } = browserPlatform({ element: {} as unknown as Element });
    const adapter = createBrowserFullscreenAdapter(platform);

    expect(adapter.active()).toBe(true);
  });

  test("skips exitFullscreen when no element is currently fullscreen", async () => {
    const { platform, exitFullscreen } = browserPlatform({ element: null });
    const adapter = createBrowserFullscreenAdapter(platform);

    await adapter.exit();
    expect(exitFullscreen).not.toHaveBeenCalled();
  });

  test("calls exitFullscreen when the platform owns a fullscreen element", async () => {
    const { platform, exitFullscreen } = browserPlatform({
      element: {} as unknown as Element,
    });
    const adapter = createBrowserFullscreenAdapter(platform);

    await adapter.exit();
    expect(exitFullscreen).toHaveBeenCalledTimes(1);
  });

  test("returns an unsupported adapter when detection is disabled with null", async () => {
    const adapter = createBrowserFullscreenAdapter(null);

    expect(adapter.supported()).toBe(false);
    expect(adapter.active()).toBe(false);
    await expect(adapter.enter(element)).rejects.toThrow(/not supported/iu);
    await expect(adapter.exit()).resolves.toBeUndefined();
  });

  test("returns an unsupported adapter when no document is present", () => {
    const adapter = createBrowserFullscreenAdapter();

    expect(adapter.supported()).toBe(false);
  });
});

describe("resolveFullscreenState", () => {
  test.each<{ current: FullscreenState; active: boolean; expected: FullscreenState }>([
    { current: "windowed", active: true, expected: "native" },
    { current: "layout", active: true, expected: "native" },
    { current: "native", active: false, expected: "windowed" },
    { current: "layout", active: false, expected: "layout" },
    { current: "windowed", active: false, expected: "windowed" },
  ])(
    "maps $current with active=$active to $expected",
    ({ current, active, expected }) => {
      const { adapter } = fakeAdapter({ active });

      expect(resolveFullscreenState(adapter, current)).toBe(expected);
    },
  );
});

describe("enterFullscreenMode", () => {
  test("falls back to layout-only immersion when native is unsupported", async () => {
    const { adapter, enter } = fakeAdapter({ supported: false });

    const result = await enterFullscreenMode(adapter, element);

    expect(result).toEqual({ state: "layout", announcement: null });
    expect(enter).not.toHaveBeenCalled();
    expect(Object.isFrozen(result)).toBe(true);
  });

  test("enters native fullscreen when the adapter accepts the request", async () => {
    const { adapter, enter } = fakeAdapter();

    const result = await enterFullscreenMode(adapter, element);

    expect(result).toEqual({ state: "native", announcement: null });
    expect(enter).toHaveBeenCalledWith(element);
  });

  test("keeps the prior state and announces denial when the request is blocked", async () => {
    const { adapter } = fakeAdapter({
      enter: async () => {
        throw new Error("blocked by permissions policy");
      },
    });

    const result = await enterFullscreenMode(adapter, element, "layout");

    expect(result).toEqual({
      state: "layout",
      announcement: FULLSCREEN_DENIED_MESSAGE,
    });
  });

  test("defaults denial recovery to the windowed shell", async () => {
    const { adapter } = fakeAdapter({
      enter: async () => {
        throw new Error("blocked");
      },
    });

    const result = await enterFullscreenMode(adapter, element);

    expect(result.state).toBe<FullscreenState>("windowed");
    expect(result.announcement).toBe(FULLSCREEN_DENIED_MESSAGE);
  });
});

describe("exitFullscreenMode", () => {
  test("returns to the windowed shell after leaving native fullscreen", async () => {
    const { adapter, exit } = fakeAdapter({ active: true });

    const result = await exitFullscreenMode(adapter, "native");

    expect(result).toEqual({ state: "windowed", announcement: null });
    expect(exit).toHaveBeenCalledTimes(1);
  });

  test("keeps native state and announces when leaving fullscreen fails", async () => {
    const { adapter } = fakeAdapter({
      active: true,
      exit: async () => {
        throw new Error("exit rejected");
      },
    });

    const result = await exitFullscreenMode(adapter, "native");

    expect(result).toEqual({
      state: "native",
      announcement: FULLSCREEN_EXIT_DENIED_MESSAGE,
    });
  });

  test("windows layout immersion without touching an inactive adapter", async () => {
    const { adapter, exit } = fakeAdapter({ active: false });

    const result = await exitFullscreenMode(adapter, "layout");

    expect(result).toEqual({ state: "windowed", announcement: null });
    expect(exit).not.toHaveBeenCalled();
  });

  test("exits a live element even when authored state is not native", async () => {
    const { adapter, exit } = fakeAdapter({ active: true });

    const result = await exitFullscreenMode(adapter, "layout");

    expect(result.state).toBe<FullscreenState>("windowed");
    expect(exit).toHaveBeenCalledTimes(1);
  });
});

describe("toggleFullscreenMode", () => {
  test("enters native immersion from the windowed shell", async () => {
    const { adapter, enter } = fakeAdapter();

    const result = await toggleFullscreenMode(adapter, element, "windowed");

    expect(result.state).toBe<FullscreenState>("native");
    expect(enter).toHaveBeenCalledWith(element);
  });

  test("enters layout immersion from windowed when native is unsupported", async () => {
    const { adapter } = fakeAdapter({ supported: false });

    const result = await toggleFullscreenMode(adapter, element, "windowed");

    expect(result.state).toBe<FullscreenState>("layout");
  });

  test("exits when authored state is already immersive", async () => {
    const { adapter, enter, exit } = fakeAdapter({ active: true });

    const result = await toggleFullscreenMode(adapter, element, "native");

    expect(result.state).toBe<FullscreenState>("windowed");
    expect(enter).not.toHaveBeenCalled();
    expect(exit).toHaveBeenCalledTimes(1);
  });

  test("exits when the browser owns fullscreen despite windowed authored state", async () => {
    const { adapter, exit } = fakeAdapter({ active: true });

    const result = await toggleFullscreenMode(adapter, element, "windowed");

    expect(result.state).toBe<FullscreenState>("windowed");
    expect(exit).toHaveBeenCalledTimes(1);
  });
});
