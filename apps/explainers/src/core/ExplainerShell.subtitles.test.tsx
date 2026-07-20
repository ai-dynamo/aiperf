/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { ExplainerShell } from "./ExplainerShell";
import type { DeckDefinition } from "./types";

vi.mock("./narration", async () => {
  const actual = await vi.importActual<typeof import("./narration")>("./narration");
  return {
    ...actual,
    narrationSupported: () => true,
    unlockSpeech: () => true,
    stopNarration: () => undefined,
    speakNarration: (
      _text: string,
      opts: { onWord?: (index: number) => void; onComplete?: () => void },
    ) => {
      opts.onWord?.(0);
      return () => undefined;
    },
  };
});

function makeDeck(): DeckDefinition {
  return {
    id: "subtitles-test",
    route: "/subtitles-test",
    storagePrefix: "ex-subtitles-test",
    classPrefix: "subtitles-test",
    eyebrowLabel: "Subtitles",
    startGateTitle: "Subtitles restore test",
    hub: {
      title: "Subtitles",
      highlight: "test",
      description: "Karaoke subtitle restoration coverage.",
    },
    slides: [
      {
        eyebrow: "SLIDE",
        title: "First slide",
        lede: "A short lede.",
        narration: "Alpha bravo charlie",
        points: ["Keep notes free of karaoke"],
        caption: "Speaker caption",
      },
    ],
    glossary: [],
    MentalModel: () => <div data-testid="mental-model">diagram</div>,
    css: "",
  };
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe("ExplainerShell karaoke subtitles", () => {
  it("shows karaoke subtitles after start without opening speaker notes", () => {
    render(
      <MemoryRouter>
        <ExplainerShell deck={makeDeck()} />
      </MemoryRouter>,
    );

    expect(screen.queryByTestId("ex-subtitles-row")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /play with audio/i }));

    const row = screen.getByTestId("ex-subtitles-row");
    const footer = row.parentElement;
    expect(footer?.classList.contains("ex-stage-footer")).toBe(true);
    expect(row.nextElementSibling?.tagName).toBe("SECTION");
    expect(footer?.previousElementSibling?.contains(screen.getByTestId("mental-model"))).toBe(
      true,
    );
    expect(within(row).getByText("SUBTITLES")).toBeTruthy();
    expect(within(row).getByText("Alpha")).toBeTruthy();
    expect(within(row).getByText("bravo")).toBeTruthy();
    expect(within(row).getByText("charlie")).toBeTruthy();
    expect(screen.queryByLabelText("Speaker notes")).toBeNull();
  });

  it("does not duplicate karaoke subtitles inside speaker notes", () => {
    render(
      <MemoryRouter>
        <ExplainerShell deck={makeDeck()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: /play without audio/i }));
    fireEvent.click(screen.getByRole("button", { name: /speaker notes/i }));

    const notes = screen.getByLabelText("Speaker notes");
    expect(within(notes).queryByText("SUBTITLES")).toBeNull();
    expect(screen.getAllByText("SUBTITLES")).toHaveLength(1);
    expect(within(notes).getByText("Keep notes free of karaoke")).toBeTruthy();
  });
});
