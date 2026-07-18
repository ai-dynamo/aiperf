// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneState } from "./store.js";

/** Paused exploration overlay anchored to an authored lesson beat. */
export type ExplorationSnapshot = Readonly<{
  authored: SceneState;
  exploration: SceneState;
}>;

/** Captures the current authored beat and opens a paused exploration overlay. */
export function beginExploration(state: SceneState): ExplorationSnapshot {
  const authored = { ...state };
  const exploration: SceneState = {
    ...state,
    playbackStatus: "paused",
    playbackTimeMs: state.playbackTimeMs,
    temporaryCameraTakeover: true,
  };
  return { authored, exploration };
}

/** Applies interaction changes while preserving the authored pause timestamp. */
export function updateExploration(
  snapshot: ExplorationSnapshot,
  exploredState: SceneState,
): ExplorationSnapshot {
  return {
    authored: snapshot.authored,
    exploration: {
      ...exploredState,
      playbackStatus: "paused",
      playbackTimeMs: snapshot.authored.playbackTimeMs,
      temporaryCameraTakeover: true,
    },
  };
}

/** Restores the authored lesson at the exact beat captured before exploration. */
export function resumeLesson(snapshot: ExplorationSnapshot): SceneState {
  return { ...snapshot.authored };
}
