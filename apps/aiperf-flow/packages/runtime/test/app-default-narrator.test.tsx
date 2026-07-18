// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

const backend = {
  available: true,
  voices: () => [],
  speak: () => undefined,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
};

vi.mock("../src/narrative/kokoro-narrator.js", () => ({
  createKokoroNarratorBackend: vi.fn(() => backend),
}));

import { FlowApp } from "../src/app.js";
import { createKokoroNarratorBackend } from "../src/narrative/kokoro-narrator.js";

const sourceMap = {
  source: "default-narrator.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function flow(): FlowIr {
  const scene: SceneIr = {
    id: "scene",
    title: "Scene",
    summary: "Narrated scene",
    roots: [],
    camera: [],
    timeline: [],
    narration: "Kokoro narrates this scene by default.",
    interactions: [],
    responsive: [],
    accessibility: { label: "Scene", readingOrder: [] },
    fallback: "Scene",
    sourceMap,
  };
  return {
    irVersion: 2,
    id: "flow",
    title: "Flow",
    capabilities: [],
    tokens: {},
    themes: [],
    scenes: [scene],
    sourceMap,
  };
}

afterEach(cleanup);

test("uses Kokoro with browser speech fallback when no backend is supplied", () => {
  render(<FlowApp flow={flow()} forceSvgFallback />);

  expect(createKokoroNarratorBackend).toHaveBeenCalledOnce();
  expect(createKokoroNarratorBackend).toHaveBeenCalledWith({
    fallback: expect.objectContaining({ available: expect.any(Boolean) }),
  });
});
