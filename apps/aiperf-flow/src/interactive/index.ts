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
