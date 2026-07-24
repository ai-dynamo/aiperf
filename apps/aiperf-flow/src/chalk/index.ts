/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! "Systems Chalk" presentational primitives — hub-and-spoke scenes and in-card mini-diagrams,
//! ported from the approved brainstorm mockup (`systems-chalk-hub-spoke.html`). Use these to build
//! chalk-style deck pages (a center hub ringed by numbered accent cards, each with a tiny diagram).

export { HubSpoke, type HubSpokeProps } from "./HubSpoke.js";
export { ChalkCard, type ChalkCardProps } from "./ChalkCard.js";
export {
  Diagram,
  NodeChip,
  RoundNode,
  DbNode,
  MiniArrow,
  MiniBars,
} from "./MiniDiagram.js";
