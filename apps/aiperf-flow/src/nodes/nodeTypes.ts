/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeTypes } from "@xyflow/react";
import { HeaderNode } from "./Header.js";
import { PanelNode } from "./Panel.js";
import { ChipNode } from "./Chip.js";
import { CardNode } from "./Card.js";
import { TimelineNode } from "./Timeline.js";
import { IntervalsNode } from "./Intervals.js";
import { BlocksNode } from "./Blocks.js";

/** Registered once, passed to every `<ReactFlow nodeTypes={nodeTypes}>` in the app. */
export const nodeTypes: NodeTypes = {
  header: HeaderNode,
  panel: PanelNode,
  chip: ChipNode,
  card: CardNode,
  timeline: TimelineNode,
  intervals: IntervalsNode,
  blocks: BlocksNode,
};
