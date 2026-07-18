// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { normalizeSceneTimeMs } from "./player.js";
import type { ImmersiveState } from "./immersive-state.js";

export type PlaybackStatus = "idle" | "playing" | "paused" | "complete";

export type InspectorState = Readonly<{
  open: boolean;
  nodeId: string | null;
}>;

export type SceneState = Readonly<{
  currentSceneId: string;
  selectedNodeId: string | null;
  inspector: InspectorState;
  playbackTimeMs: number;
  playbackStatus: PlaybackStatus;
  activeResponsiveVariant: string | null;
  temporaryCameraTakeover: boolean;
  immersive?: ImmersiveState;
}>;

export type SceneAction =
  | Readonly<{ type: "change-scene"; sceneId: string }>
  | Readonly<{ type: "select-node"; nodeId: string | null }>
  | Readonly<{ type: "open-inspector"; nodeId: string }>
  | Readonly<{ type: "close-inspector" }>
  | Readonly<{
      type: "set-playback";
      timeMs: number;
      status: PlaybackStatus;
    }>
  | Readonly<{ type: "set-responsive-variant"; variantId: string | null }>
  | Readonly<{ type: "set-camera-takeover"; active: boolean }>;

export function createInitialSceneState(sceneId: string): SceneState {
  return {
    currentSceneId: sceneId,
    selectedNodeId: null,
    inspector: { open: false, nodeId: null },
    playbackTimeMs: 0,
    playbackStatus: "idle",
    activeResponsiveVariant: null,
    temporaryCameraTakeover: false,
  };
}

export function sceneReducer(
  state: SceneState,
  action: SceneAction,
): SceneState {
  switch (action.type) {
    case "change-scene":
      return createInitialSceneState(action.sceneId);
    case "select-node":
      return { ...state, selectedNodeId: action.nodeId };
    case "open-inspector":
      return {
        ...state,
        selectedNodeId: action.nodeId,
        inspector: { open: true, nodeId: action.nodeId },
      };
    case "close-inspector":
      return { ...state, inspector: { open: false, nodeId: null } };
    case "set-playback":
      return {
        ...state,
        playbackTimeMs: normalizeSceneTimeMs(action.timeMs),
        playbackStatus: action.status,
      };
    case "set-responsive-variant":
      return { ...state, activeResponsiveVariant: action.variantId };
    case "set-camera-takeover":
      return { ...state, temporaryCameraTakeover: action.active };
  }
}
