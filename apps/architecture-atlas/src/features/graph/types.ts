// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AudienceLevel,
  GraphEdge,
  GraphNode,
} from "../../domain/architecture";
import type { DirectedNeighborhood } from "../../domain/graph-derivation";
import type {
  LayoutRequest,
  LayoutResult,
} from "../atlas/layout";

export type GraphPathState = "focused" | "upstream" | "downstream" | "default";

export interface GraphCanvasLayoutService {
  layout(request: LayoutRequest): Promise<LayoutResult>;
}

export interface GraphCanvasProps {
  audience: AudienceLevel;
  focusedEntityId: string | null;
  layoutRequest: LayoutRequest;
  layoutService: GraphCanvasLayoutService;
  neighborhood: DirectedNeighborhood;
  onFocusEntity(entityId: string): void;
  visibleEdges: readonly GraphEdge[];
  visibleNodes: readonly GraphNode[];
}

export interface GraphNodePortView {
  channel: GraphNode["seamPorts"][number]["channel"];
  direction: "source" | "target" | "source+target" | "none";
  id: string;
  name: string;
}
