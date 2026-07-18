// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";

/** Serializable camera transform in scene coordinates. */
export type CameraTransform = Readonly<{
  x: number;
  y: number;
  zoom: number;
}>;

/** Serializable bounds to fit within a viewport. */
export type CameraBounds = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** Serializable viewport dimensions in logical pixels. */
export type CameraViewport = Readonly<{
  width: number;
  height: number;
}>;

/** Temporary camera takeover anchored to one authored timeline beat. */
export type CameraTakeover = Readonly<{
  pausedAtMs: number;
  authored: CameraTransform;
  temporary: CameraTransform;
  takeover: true;
}>;

export type CameraResumePlan = Readonly<{
  resumedAtMs: number;
  target: CameraTransform;
  mode: "smooth" | "cut";
  takeover: false;
}>;

type CameraTrack = SceneIr["camera"];

function integerTime(timeMs: number): number {
  if (!Number.isFinite(timeMs)) {
    return 0;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.trunc(timeMs)));
}

function transform(
  value: Pick<CameraTrack[number], "x" | "y" | "zoom">,
): CameraTransform {
  return { x: value.x, y: value.y, zoom: value.zoom };
}

/** Samples the authored camera track at a canonical integer timeline beat. */
export function authoredCameraAt(
  camera: CameraTrack,
  timeMs: number,
): CameraTransform {
  if (camera.length === 0) {
    return { x: 0, y: 0, zoom: 1 };
  }

  const time = integerTime(timeMs);
  const keyframes = camera
    .map((keyframe, index) => ({ keyframe, index }))
    .sort(
      (left, right) =>
        left.keyframe.at - right.keyframe.at || left.index - right.index,
    )
    .map(({ keyframe }) => keyframe);
  const first = keyframes[0]!;
  const last = keyframes[keyframes.length - 1]!;

  if (time <= first.at) {
    return transform(first);
  }
  if (time >= last.at) {
    return transform(last);
  }

  const endIndex = keyframes.findIndex(({ at }) => at > time);
  const start = keyframes[endIndex - 1]!;
  const end = keyframes[endIndex]!;
  const progress = (time - start.at) / (end.at - start.at);
  return {
    x: start.x + (end.x - start.x) * progress,
    y: start.y + (end.y - start.y) * progress,
    zoom: start.zoom + (end.zoom - start.zoom) * progress,
  };
}

/** Freezes an authored camera beat and starts temporary camera takeover. */
export function beginCameraTakeover(
  camera: CameraTrack,
  pausedAtMs: number,
): CameraTakeover {
  const pausedAt = integerTime(pausedAtMs);
  const authored = authoredCameraAt(camera, pausedAt);
  return {
    pausedAtMs: pausedAt,
    authored,
    temporary: authored,
    takeover: true,
  };
}

/** Applies a temporary scene-coordinate pan without advancing the frozen beat. */
export function panCameraTakeover(
  state: CameraTakeover,
  delta: Readonly<{ x: number; y: number }>,
): CameraTakeover {
  return {
    ...state,
    temporary: {
      ...state.temporary,
      x: state.temporary.x + delta.x,
      y: state.temporary.y + delta.y,
    },
  };
}

/** Applies an absolute temporary zoom without advancing the frozen beat. */
export function zoomCameraTakeover(
  state: CameraTakeover,
  zoom: number,
): CameraTakeover {
  if (!Number.isFinite(zoom) || zoom <= 0) {
    throw new RangeError("Camera zoom must be a positive finite number.");
  }
  return { ...state, temporary: { ...state.temporary, zoom } };
}

/** Fits scene bounds into a logical viewport as a temporary camera takeover. */
export function fitCameraTakeover(
  state: CameraTakeover,
  bounds: CameraBounds,
  viewport: CameraViewport,
  padding = 0,
): CameraTakeover {
  const availableWidth = viewport.width - padding * 2;
  const availableHeight = viewport.height - padding * 2;
  if (
    ![
      bounds.x,
      bounds.y,
      bounds.width,
      bounds.height,
      viewport.width,
      viewport.height,
      padding,
    ].every(Number.isFinite) ||
    padding < 0 ||
    bounds.width <= 0 ||
    bounds.height <= 0 ||
    availableWidth <= 0 ||
    availableHeight <= 0
  ) {
    throw new RangeError("Camera fit requires positive drawable dimensions.");
  }
  return {
    ...state,
    temporary: {
      x: bounds.x + bounds.width / 2,
      y: bounds.y + bounds.height / 2,
      zoom: Math.min(
        availableWidth / bounds.width,
        availableHeight / bounds.height,
      ),
    },
  };
}

/** Produces an exact authored-camera restoration plan at the paused beat. */
export function resumeAuthoredCamera(
  state: CameraTakeover,
  options: Readonly<{ reducedMotion?: boolean }> = {},
): CameraResumePlan {
  return {
    resumedAtMs: state.pausedAtMs,
    target: state.authored,
    mode: options.reducedMotion === true ? "cut" : "smooth",
    takeover: false,
  };
}
