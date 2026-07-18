// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

const activate = vi.fn(async () => undefined);
const prewarm = vi.fn(async () => undefined);
const speak = vi.fn();
const backend = {
  available: true,
  voices: () => [],
  speak,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
  activate,
  prewarm,
};

vi.mock("../src/narrative/kokoro-narrator.js", () => ({
  createKokoroNarratorBackend: vi.fn(() => backend),
}));

import { FlowApp } from "../src/app.js";

const sourceMap = {
  source: "audio-consent.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function scene(id: string, title: string): SceneIr {
  return {
    id,
    title,
    summary: `${title} summary`,
    roots: [
      {
        kind: "rect",
        id: `${id}-cli`,
        geometry: { x: 0, y: 0, width: 40, height: 20 },
        style: { fill: "#315b8a" },
        accessibility: { label: "CLI" },
        fallback: "CLI",
        sourceMap,
      },
    ],
    camera: [],
    timeline: [
      {
        id: `${id}-reveal`,
        at: 0,
        duration: 400,
        action: "reveal",
        target: `${id}-cli`,
        sourceMap,
      },
    ],
    narration: "Kokoro narrates this scene after audio consent.",
    interactions: [],
    responsive: [],
    accessibility: { label: title, readingOrder: [`${id}-cli`] },
    fallback: title,
    sourceMap,
  };
}

function flow(scenes: readonly SceneIr[] = [scene("scene", "Scene")]): FlowIr {
  return {
    irVersion: 2,
    id: "flow",
    title: "Flow",
    capabilities: [],
    tokens: {},
    themes: [],
    scenes: [...scenes],
    sourceMap,
  };
}

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  activate.mockReset();
  activate.mockResolvedValue(undefined);
  prewarm.mockReset();
  prewarm.mockResolvedValue(undefined);
  speak.mockClear();
});

test("Play with audio unlocks narration from the consent gesture", async () => {
  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  expect(screen.getByRole("dialog", { name: "Audio preference" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Play with audio" }));

  await vi.waitFor(() => {
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(activate).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
  });
});

test("starts browser narration while Kokoro startup is unresolved", async () => {
  activate.mockImplementation(() => new Promise<void>(() => undefined));
  prewarm.mockImplementation(() => new Promise<void>(() => undefined));
  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  fireEvent.click(screen.getByRole("button", { name: "Play with audio" }));

  await vi.waitFor(() => expect(speak).toHaveBeenCalled());
});

test("Play without audio starts muted playback", async () => {
  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  fireEvent.click(screen.getByRole("button", { name: "Play without audio" }));

  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
  await vi.waitFor(() => {
    expect(screen.getByRole("option", { name: "Unmute narration" })).toBeTruthy();
  });
});

test("asks again on a fresh mount instead of retaining visit storage", () => {
  const first = render(
    <FlowApp flow={flow()} forceSvgFallback requireAudioConsent />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Play without audio" }));
  first.unmount();

  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  expect(screen.getByRole("dialog", { name: "Audio preference" })).toBeTruthy();
});

test("autoplays the next scene after consent without asking again", async () => {
  render(
    <FlowApp
      flow={flow([scene("one", "One"), scene("two", "Two")])}
      forceSvgFallback
      requireAudioConsent
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Play with audio" }));
  await vi.waitFor(() =>
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy(),
  );

  fireEvent.click(screen.getByRole("button", { name: "Next scene" }));

  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("heading", { name: "Two" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
});
