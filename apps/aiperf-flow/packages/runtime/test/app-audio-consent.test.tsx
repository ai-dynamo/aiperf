// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

const activate = vi.fn(async () => undefined);
const prewarm = vi.fn(async () => undefined);
const backend = {
  available: true,
  voices: () => [],
  speak: () => undefined,
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

function flow(): FlowIr {
  const scene: SceneIr = {
    id: "scene",
    title: "Scene",
    summary: "Narrated scene",
    roots: [
      {
        kind: "rect",
        id: "cli",
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
        id: "reveal",
        at: 0,
        duration: 400,
        action: "reveal",
        target: "cli",
        sourceMap,
      },
    ],
    narration: "Kokoro narrates this scene after audio consent.",
    interactions: [],
    responsive: [],
    accessibility: { label: "Scene", readingOrder: ["cli"] },
    fallback: "Scene",
    sourceMap,
  };
  return {
    irVersion: 1,
    id: "flow",
    title: "Flow",
    capabilities: [],
    tokens: {},
    scenes: [scene],
    sourceMap,
  };
}

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  activate.mockClear();
  prewarm.mockClear();
});

test("Play with audio unlocks narration from the consent gesture", () => {
  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  expect(screen.getByRole("dialog", { name: "Audio preference" })).toBeTruthy();
  expect(
    (screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled,
  ).toBe(true);

  fireEvent.click(screen.getByRole("button", { name: "Play with audio" }));

  expect(screen.queryByRole("dialog")).toBeNull();
  expect(activate).toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
});

test("Play without audio starts muted playback", () => {
  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  fireEvent.click(screen.getByRole("button", { name: "Play without audio" }));

  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("button", { name: "Unmute narration" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
});

test("remembers the audio choice when another flow mounts during the visit", () => {
  const first = render(
    <FlowApp flow={flow()} forceSvgFallback requireAudioConsent />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Play without audio" }));
  first.unmount();

  render(<FlowApp flow={flow()} forceSvgFallback requireAudioConsent />);

  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.getByRole("button", { name: "Unmute narration" })).toBeTruthy();
});
