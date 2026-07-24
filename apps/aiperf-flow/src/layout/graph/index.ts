/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public surface of the `apps/aiperf-flow` ELK auto-layout engine. Diagrams adopt it by passing
//! `layout` to `PipelineCanvas`, using `AutoLayoutFlow` in place of a raw `<ReactFlow>`, or calling
//! `useElkLayout` directly inside their own `ReactFlowProvider`.

export { layoutGraph, fallbackLayout, DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT } from "./elkEngine.js";
export type { ElkOptions } from "./elkEngine.js";
export { useElkLayout } from "./useElkLayout.js";
export type { UseElkLayoutResult } from "./useElkLayout.js";
export { AutoLayoutFlow } from "./AutoLayoutFlow.js";
export type { AutoLayoutFlowProps } from "./AutoLayoutFlow.js";
