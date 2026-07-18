// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  beginExploration,
  type ExplorationSnapshot,
  resumeLesson,
  updateExploration,
} from "../src/exploration.js";
import {
  createInitialSceneState,
  type SceneState,
} from "../src/store.js";

function playingState(): SceneState {
  return {
    ...createInitialSceneState("request-flow"),
    selectedNodeId: "router",
    inspector: { open: true, nodeId: "router" },
    playbackTimeMs: 625,
    playbackStatus: "playing",
    activeResponsiveVariant: "desktop",
  };
}

describe("exploration lifecycle", () => {
  test("beginning exploration pauses the authored lesson at its current beat", () => {
    const snapshot = beginExploration(playingState());

    expect(snapshot.authored.playbackStatus).toBe("playing");
    expect(snapshot.exploration).toMatchObject({
      playbackTimeMs: 625,
      playbackStatus: "paused",
      temporaryCameraTakeover: true,
    });
  });

  test("updating exploration preserves its paused authored timestamp", () => {
    const snapshot = beginExploration(playingState());
    const exploredState: SceneState = {
      ...snapshot.exploration,
      selectedNodeId: "worker",
      inspector: { open: true, nodeId: "worker" },
      playbackTimeMs: 900,
      playbackStatus: "playing",
    };

    const updated = updateExploration(snapshot, exploredState);

    expect(updated.exploration).toMatchObject({
      selectedNodeId: "worker",
      playbackTimeMs: 625,
      playbackStatus: "paused",
      temporaryCameraTakeover: true,
    });
  });

  test("resuming restores the authored state at the exact paused beat", () => {
    const snapshot = updateExploration(beginExploration(playingState()), {
      ...playingState(),
      selectedNodeId: "worker",
      playbackTimeMs: 1_200,
    });

    expect(resumeLesson(snapshot)).toEqual(playingState());
  });

  test("a serialized snapshot can resume without skipping the beat", () => {
    const snapshot = beginExploration(playingState());
    const serialized = JSON.stringify(snapshot);
    const restored = JSON.parse(serialized) as ExplorationSnapshot;

    expect(resumeLesson(restored)).toEqual(playingState());
    expect(restored.authored.playbackTimeMs).toBe(625);
  });
});
