// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AudienceLevel,
  FlowChannel,
  GraphEdge,
  GraphNode,
} from "../../domain/architecture";
import type {
  DirectedNeighborhood,
  FlavorOverlay,
} from "../../domain/graph-derivation";
import type {
  LayoutRequest,
  LayoutResult,
} from "../atlas/layout";

export type GraphPathState = "focused" | "upstream" | "downstream" | "default";
export type GraphFlavorClass = "shared" | "primary-only" | "compare-only";
export type GraphTraceMode = "none" | "upstream" | "downstream" | "isolate";
export type GraphPulseState = "idle" | "active" | "completed";
export type GraphPulseChannelState = "idle" | "active" | "completed";
export type GraphRelayoutState = "canonical" | "preserved" | "relaid-out";

export interface GraphPulseEdges {
  activeChannels: readonly FlowChannel[];
  activeEdgeIds: readonly string[];
  completedChannels: readonly FlowChannel[];
  completedEdgeIds: readonly string[];
  reducedMotion: boolean;
}

export interface GraphManualNodePosition {
  nodeId: string;
  x: number;
  y: number;
}

export interface GraphFitViewCommand {
  requestId: number;
  padding?: number;
}

export interface GraphCanvasLayoutService {
  layout(request: LayoutRequest): Promise<LayoutResult>;
}

export interface GraphCanvasProps {
  activePulseNodeIds?: readonly string[];
  audience: AudienceLevel;
  breadcrumbNodeIds?: readonly string[];
  completedPulseNodeIds?: readonly string[];
  expandedNodeIds?: readonly string[];
  edgeWaypoints?: ReadonlyMap<string, { x: number; y: number }[]>;
  focusedEntityId: string | null;
  layoutRequest: LayoutRequest;
  layoutService: GraphCanvasLayoutService;
  neighborhood: DirectedNeighborhood;
  fitViewCommand?: GraphFitViewCommand;
  onCollapseNode?(nodeId: string): void;
  onExpandNode?(nodeId: string): void;
  onFitViewComplete?(requestId: number): void;
  onFocusBreadcrumb?(nodeId: string): void;
  onFocusEntity(entityId: string): void;
  onNodeDragComplete?(position: GraphManualNodePosition): void;
  onTraceModeChange?(nodeId: string, mode: GraphTraceMode): void;
  onWaypointsChange?(update: { edgeId: string; points: { x: number; y: number }[] }): void;
  onWaypointsReset?(edgeId: string): void;
  overlay: FlavorOverlay;
  pulseEdges?: GraphPulseEdges;
  traceMode?: GraphTraceMode;
  visibleEdges: readonly GraphEdge[];
  visibleNodes: readonly GraphNode[];
}

export interface GraphNodePortView {
  channel: GraphNode["seamPorts"][number]["channel"];
  direction: "source" | "target" | "source+target" | "none";
  id: string;
  name: string;
}
