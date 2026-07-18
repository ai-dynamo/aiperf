// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ImmersiveState } from "./immersive-state.js";
import type { SceneState } from "./store.js";

/** Paused exploration overlay anchored to an authored lesson beat. */
export type ExplorationSnapshot = Readonly<{
  authored: SceneState;
  exploration: SceneState;
  immersive?: ImmersiveState;
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
  return state.immersive === undefined
    ? { authored, exploration }
    : { authored, exploration, immersive: state.immersive };
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
    ...(snapshot.immersive === undefined
      ? {}
      : { immersive: snapshot.immersive }),
  };
}

/** Restores the authored lesson at the exact beat captured before exploration. */
export function resumeLesson(snapshot: ExplorationSnapshot): SceneState {
  const immersive = snapshot.immersive ?? snapshot.authored.immersive;
  return immersive === undefined
    ? { ...snapshot.authored }
    : { ...snapshot.authored, immersive };
}
