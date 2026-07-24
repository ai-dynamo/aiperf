/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Domain-agnostic data shapes for the `src/interactive/` semantic-zoom + play primitives.
//! Nothing here is AIPerf-specific: `ZoomStage`/`PipelineCanvas`/`useFlowPlayer` are all driven
//! by these generics so any future deck can animate "a thing moving through a zoomable graph".

import type { Edge, Node } from "@xyflow/react";

/**
 * One node in a {@link ZoomTree}: the React Flow subgraph shown when this node is the active
 * (expanded) level, its breadcrumb label, and the ids of any children it can drill into.
 *
 * @typeParam T - optional per-node payload a deck can attach (a source anchor, a tone, whatever
 *   its render-prop needs); defaults to `unknown` so the common id→subgraph case needs no arg.
 */
export interface ZoomTreeNode<T = unknown> {
  /** Human-readable label, used for the breadcrumb trail and the level title. */
  label: string;
  /** The React Flow nodes rendered when this tree node is the active level. */
  nodes: Node[];
  /** The React Flow edges rendered when this tree node is the active level. */
  edges: Edge[];
  /** Ids of child nodes (keys into the same {@link ZoomTree}) this node can drill into. */
  children?: string[];
  /** Optional deck-supplied payload for this node. */
  data?: T;
}

/**
 * A semantic-zoom tree: a flat map from node id to its {@link ZoomTreeNode}. Flat (not nested) so
 * a node can be looked up by id in O(1) during drill/pop, and so the same child id can be
 * referenced from more than one parent if a deck wants a shared leaf.
 *
 * @typeParam T - per-node payload type threaded through to every {@link ZoomTreeNode}.
 */
export type ZoomTree<T = unknown> = Record<string, ZoomTreeNode<T>>;

/**
 * One step of a {@link useFlowPlayer} playback: the node highlighted while the step is active,
 * the "what's happening now" caption, and optional timing/variant hints. Generic in intent — a
 * step is just "highlight node X, say Y" — with no assumptions about the graph's domain.
 */
export interface FlowStep {
  /** Id of the node highlighted while this step is active (matches a React Flow node id). */
  nodeId: string;
  /** The "what's happening now" caption shown for this step. */
  caption: string;
  /** Optional relative dwell time (ms) for wall-paced playback; players may ignore it. */
  timingMs?: number;
  /** Optional variant tag this step belongs to (e.g. a selected transport/clock mode). */
  variant?: string;
}
