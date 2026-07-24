/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public surface of the `src/interactive/` shared primitives: domain-agnostic semantic-zoom,
//! pipeline canvas, play/scrub, and seam-toggle building blocks any deck can compose. Additive to
//! the app's shared library (parallel to `src/nodes`/`src/edges`/`src/shell`).

export type { FlowStep, ZoomTree, ZoomTreeNode } from "./types.js";
export { PipelineCanvas, type PipelineCanvasProps } from "./PipelineCanvas.js";
export { ZoomStage, type ZoomStageProps, type ZoomStageContext } from "./ZoomStage.js";
export { useFlowPlayer, type FlowPlayer } from "./useFlowPlayer.js";
export { RequestParticle, type RequestParticleProps } from "./RequestParticle.js";
export { SeamToggle, type SeamToggleProps, type SeamToggleOption } from "./SeamToggle.js";

// Swimlane-timeline renderer + its domain-agnostic data model and pure x-mapping helpers. The
// `Lane`/`StageRegion`/`SeamFrame` *data* types are the public contract (a deck supplies them); the
// same-named presentational subcomponents stay internal to `TimelineTrack` (they would collide with
// these type names in this barrel). `TimeAxis`/`RequestLine`/`EventMarker` are exported for reuse.
export {
  type LaneId,
  type Lane,
  type StageRegion,
  type TimelineEvent,
  type SeamFrame,
  type RequestPath,
  type TimelineScale,
  type TimelineBounds,
  eventOffsetMs,
  timelineBounds,
  buildOffsetForOrder,
  fractionForEvent,
  fractionForOrder,
} from "./timeline.js";
export { TimelineTrack, type TimelineTrackProps } from "./TimelineTrack.js";
export { TimeAxis, type TimeAxisProps, type TimeAxisTick } from "./TimeAxis.js";
export { RequestLine, type RequestLineProps, type LinePoint } from "./RequestLine.js";
export { EventMarker, type EventMarkerProps } from "./EventMarker.js";
