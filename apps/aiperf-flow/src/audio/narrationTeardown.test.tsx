/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Leaving a narrated deck must silence it.
//!
//! `useNarratedDeck` stops narration whenever `location.pathname` changes, which covers moving
//! between two narrated decks — the arriving deck's effect fires. It does not cover *leaving* for
//! a page with no narration, because the hook unmounts and that effect never runs again. Then the
//! voice keeps reading a deck nobody is looking at.

import { act, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useNarratedDeck } from "./useNarratedDeck.js";

const cancel = vi.fn();

function Deck(): React.JSX.Element {
  const narrated = useNarratedDeck({
    narrations: ["one two three", "four five six"],
    storagePrefix: "test:teardown",
  });
  return (
    <button type="button" onClick={() => narrated.begin(true)}>
      begin
    </button>
  );
}

beforeEach(() => {
  cancel.mockClear();
  // jsdom ships no speech engine, so `narrationSupported()` is false and `stopNarration` becomes
  // a no-op. Install a minimal one so the teardown path is actually exercised.
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: {
      getVoices: () => [],
      speak: vi.fn(),
      cancel,
      // `useSpeechVoices` subscribes to voice changes; the stub has to accept that.
      addEventListener: () => {},
      removeEventListener: () => {},
    },
  });
  Object.defineProperty(window, "SpeechSynthesisUtterance", {
    configurable: true,
    value: class {
      constructor(public text: string) {}
    },
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("narration teardown", () => {
  it("cancels speech when the deck unmounts", () => {
    const { unmount } = render(
      <MemoryRouter>
        <Deck />
      </MemoryRouter>,
    );
    cancel.mockClear();

    act(() => {
      unmount();
    });

    // Navigating to a page that mounts no narrated deck leaves nothing else to stop it.
    expect(cancel).toHaveBeenCalled();
  });

  it("still cancels when the deck was actively speaking", () => {
    const { getByText, unmount } = render(
      <MemoryRouter>
        <Deck />
      </MemoryRouter>,
    );
    act(() => {
      getByText("begin").click();
    });
    cancel.mockClear();

    act(() => {
      unmount();
    });

    expect(cancel).toHaveBeenCalled();
  });
});
