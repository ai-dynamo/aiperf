// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Injectable Fullscreen API boundary and enter/exit policy for immersive mode.
//!
//! Native fullscreen is preferred when the browser allows it. Unsupported
//! environments fall back to layout-only immersion. Denial keeps the caller's
//! current layout and returns a recoverable announcement for the live region.

import type { FullscreenState } from "./immersive-state.js";

/** Browser Fullscreen API surface accepted as an injectable platform boundary. */
export interface FullscreenAdapter {
  supported(): boolean;
  active(): boolean;
  enter(element: HTMLElement): Promise<void>;
  exit(): Promise<void>;
}

/** Minimal document/element hooks required by the browser adapter. */
export type FullscreenPlatform = Readonly<{
  fullscreenEnabled: boolean;
  fullscreenElement: Element | null;
  requestFullscreen(element: HTMLElement): Promise<void>;
  exitFullscreen(): Promise<void>;
}>;

/** Next immersive fullscreen state plus an optional live-region announcement. */
export type FullscreenModeResult = Readonly<{
  state: FullscreenState;
  announcement: string | null;
}>;

/** Recoverable denial copy when the browser blocks a native fullscreen request. */
export const FULLSCREEN_DENIED_MESSAGE =
  "Fullscreen was blocked. The current layout is unchanged.";

/** Recoverable denial copy when leaving native fullscreen fails. */
export const FULLSCREEN_EXIT_DENIED_MESSAGE =
  "Unable to leave fullscreen. Try Escape or your browser controls.";

const unsupportedAdapter: FullscreenAdapter = Object.freeze({
  supported: () => false,
  active: () => false,
  enter: async () => {
    throw new Error("Fullscreen is not supported in this environment.");
  },
  exit: async () => undefined,
});

class BrowserFullscreenAdapter implements FullscreenAdapter {
  readonly #platform: FullscreenPlatform;

  constructor(platform: FullscreenPlatform) {
    this.#platform = platform;
  }

  supported(): boolean {
    return this.#platform.fullscreenEnabled;
  }

  active(): boolean {
    return this.#platform.fullscreenElement !== null;
  }

  enter(element: HTMLElement): Promise<void> {
    return this.#platform.requestFullscreen(element);
  }

  exit(): Promise<void> {
    if (this.#platform.fullscreenElement === null) {
      return Promise.resolve();
    }
    return this.#platform.exitFullscreen();
  }
}

function detectBrowserFullscreen(): FullscreenPlatform | null {
  if (typeof document === "undefined") {
    return null;
  }

  const doc = document;
  const requestFullscreen = Element.prototype.requestFullscreen;
  if (
    typeof requestFullscreen !== "function" ||
    typeof doc.exitFullscreen !== "function"
  ) {
    return null;
  }

  return {
    get fullscreenEnabled() {
      return doc.fullscreenEnabled;
    },
    get fullscreenElement() {
      return doc.fullscreenElement;
    },
    requestFullscreen(element: HTMLElement) {
      return element.requestFullscreen();
    },
    exitFullscreen() {
      return doc.exitFullscreen();
    },
  };
}

/**
 * Creates a Fullscreen API adapter for the current document.
 *
 * Passing a platform keeps tests free of ambient globals. Passing `null`
 * disables feature detection and yields an unsupported adapter so callers can
 * fall back to layout-only immersion.
 */
export function createBrowserFullscreenAdapter(
  platform?: FullscreenPlatform | null,
): FullscreenAdapter {
  const resolved =
    platform === undefined ? detectBrowserFullscreen() : platform;
  return resolved === null
    ? unsupportedAdapter
    : new BrowserFullscreenAdapter(resolved);
}

/**
 * Reconciles authored fullscreen state with the live browser element.
 *
 * When the user leaves native fullscreen via Escape, native mode collapses to
 * windowed. Layout-only immersion is unchanged by browser fullscreen events.
 */
export function resolveFullscreenState(
  adapter: FullscreenAdapter,
  current: FullscreenState,
): FullscreenState {
  if (adapter.active()) {
    return "native";
  }
  if (current === "native") {
    return "windowed";
  }
  return current;
}

/**
 * Enters immersive fullscreen: native when supported, otherwise layout-only.
 *
 * Denial keeps `current` and returns a recoverable announcement. Playback is
 * never interrupted by this policy.
 */
export async function enterFullscreenMode(
  adapter: FullscreenAdapter,
  element: HTMLElement,
  current: FullscreenState = "windowed",
): Promise<FullscreenModeResult> {
  if (!adapter.supported()) {
    return Object.freeze({ state: "layout", announcement: null });
  }

  try {
    await adapter.enter(element);
    return Object.freeze({ state: "native", announcement: null });
  } catch {
    return Object.freeze({
      state: current,
      announcement: FULLSCREEN_DENIED_MESSAGE,
    });
  }
}

/**
 * Leaves native or layout immersion and returns to the windowed shell.
 */
export async function exitFullscreenMode(
  adapter: FullscreenAdapter,
  current: FullscreenState,
): Promise<FullscreenModeResult> {
  if (current === "native" || adapter.active()) {
    try {
      await adapter.exit();
    } catch {
      return Object.freeze({
        state: current,
        announcement: FULLSCREEN_EXIT_DENIED_MESSAGE,
      });
    }
  }

  return Object.freeze({ state: "windowed", announcement: null });
}

/**
 * Toggles between windowed and immersive fullscreen using enter/exit policy.
 */
export async function toggleFullscreenMode(
  adapter: FullscreenAdapter,
  element: HTMLElement,
  current: FullscreenState,
): Promise<FullscreenModeResult> {
  if (current !== "windowed" || adapter.active()) {
    return exitFullscreenMode(adapter, current);
  }
  return enterFullscreenMode(adapter, element, current);
}
