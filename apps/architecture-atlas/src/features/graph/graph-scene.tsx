// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useEffect,
  useMemo,
  useState,
  useRef,
} from "react";

import { architectureCatalog } from "../../content";
import type { Audience } from "../../domain/audience";
import type { ExecutionFlavor, GraphEdge, GraphNode } from "../../domain/architecture";
import {
  collapseExpandedNode,
  deriveGraphDerivation,
  toggleExpandedNode,
} from "../../domain/graph-derivation";
import { canonicalGraphState, type GraphState } from "../../domain/graph-state";
import {
  DEFAULT_TIMELINE_PLAYBACK,
  buildFlowTimeline,
  pauseTimeline,
  playTimeline,
  scrubTimeline,
  resolveTimelineSemanticState,
} from "../../domain/flow-timeline";
import {
  layoutAtlas,
  type LayoutRequest,
} from "../atlas/layout";
import { AccessibilityOutline } from "./accessibility-outline";
import { EvidenceDrawer, type EvidenceDrawerEntity } from "./evidence-drawer";
import { GraphCanvas } from "./graph-canvas";
import { PulseControls } from "./pulse-controls";
import { derivePulseEdgeOverlayState, PulseLayer } from "./pulse-layer";
import type { GraphFitViewCommand } from "./types";

interface GraphSceneProps {
  audience: Audience;
  compareFlavor: ExecutionFlavor | null;
  fallbackFocusElementId: string;
  fitViewCommand: GraphFitViewCommand | null;
  primaryFlavor: ExecutionFlavor;
  sceneId: string;
  searchQuery: string;
  state: GraphState;
  onFitViewComplete(requestId: number): void;
  onGraphStateChange(nextState: GraphState): void;
}

function buildGraphLayoutRequest(input: {
  sceneId: string;
  nodes: readonly GraphNode[];
  edges: readonly GraphEdge[];
  expandedNodeIds: readonly string[];
  nodePositions: GraphState["nodePositions"];
}): LayoutRequest {
  const tiers = [...new Set(input.nodes.map((node) => node.tier))].sort(
    (left, right) => left - right,
  );
  const bandByTier = new Map(tiers.map((tier, index) => [tier, `tier.${index}`]));
  const nodeIds = new Set(input.nodes.map((node) => node.id));
  const nodeById = new Map(input.nodes.map((node) => [node.id, node]));

  const nodes = [...input.nodes]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((node) => ({
      bandId: bandByTier.get(node.tier) ?? "tier.0",
      id: node.id,
      parentId:
        node.parentId &&
        nodeIds.has(node.parentId) &&
        nodeById.get(node.parentId)?.tier === node.tier
          ? node.parentId
          : undefined,
    }));

  const edges = [...input.edges]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((edge) => ({ from: edge.source.nodeId, id: edge.id, to: edge.target.nodeId }));

  const manualNodeIds = new Set(input.nodePositions.map(({ nodeId }) => nodeId));
  const relayoutNodeIds = nodes
    .map(({ id }) => id)
    .filter((nodeId) => !manualNodeIds.has(nodeId));
  const manualPositions = input.nodePositions
    .filter(({ nodeId }) => nodeIds.has(nodeId))
    .map(({ nodeId, x, y }) => ({ id: nodeId, x, y }));
  const expandedSubgraphs = input.expandedNodeIds
    .filter((nodeId) => nodeIds.has(nodeId))
    .map((rootId) => ({
      rootId,
      nodeIds: [rootId],
    }));

  return {
    bands: tiers.map((tier, index) => ({
      id: bandByTier.get(tier) ?? `tier.${index}`,
      label: `Tier ${tier}`,
      order: index,
    })),
    edges,
    key: [
      input.sceneId,
      nodes.map(({ id }) => id).join(","),
      edges.map(({ id }) => id).join(","),
      input.expandedNodeIds.join(","),
      input.nodePositions.map(({ nodeId, x, y }) => `${nodeId}:${x}:${y}`).join(","),
    ].join("|"),
    nodes,
    partialRelayout: {
      expandedSubgraphs,
      manualPositions,
      relayoutNodeIds,
    },
    perspective: "ownership",
    version: 1,
  };
}

function replaceNodePosition(
  nodePositions: GraphState["nodePositions"],
  nextPosition: { nodeId: string; x: number; y: number },
): GraphState["nodePositions"] {
  const byId = new Map(nodePositions.map((position) => [position.nodeId, position]));
  byId.set(nextPosition.nodeId, nextPosition);
  return [...byId.values()];
}

function replaceEdgeWaypoints(
  edgeWaypoints: GraphState["edgeWaypoints"],
  nextWaypoints: { edgeId: string; points: { x: number; y: number }[] },
): GraphState["edgeWaypoints"] {
  if (nextWaypoints.points.length === 0) {
    return edgeWaypoints.filter(({ edgeId }) => edgeId !== nextWaypoints.edgeId);
  }
  const byId = new Map(edgeWaypoints.map((waypoint) => [waypoint.edgeId, waypoint]));
  byId.set(nextWaypoints.edgeId, nextWaypoints);
  return [...byId.values()];
}

function collectDescendantNodeIds(
  rootNodeId: string,
  nodes: readonly GraphNode[],
): Set<string> {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const descendants = new Set<string>();
  const pending = [rootNodeId];
  while (pending.length > 0) {
    const nodeId = pending.pop();
    if (!nodeId) {
      continue;
    }
    const node = nodeById.get(nodeId);
    if (!node) {
      continue;
    }
    for (const childId of node.childIds) {
      if (!descendants.has(childId)) {
        descendants.add(childId);
        pending.push(childId);
      }
    }
  }
  return descendants;
}

function reducedMotionPreferred(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function createEvidenceEntity(
  focusedEntityId: string | null,
  visibleNodes: readonly GraphNode[],
  visibleEdges: readonly GraphEdge[],
): EvidenceDrawerEntity | null {
  if (!focusedEntityId) {
    return null;
  }

  const nodeById = new Map(visibleNodes.map((node) => [node.id, node]));
  const edgeById = new Map(visibleEdges.map((edge) => [edge.id, edge]));

  const focusedNode = nodeById.get(focusedEntityId);
  if (focusedNode) {
    return {
      kind: "node",
      node: focusedNode,
      relatedEdges: visibleEdges.filter(
        (edge) =>
          edge.source.nodeId === focusedNode.id || edge.target.nodeId === focusedNode.id,
      ),
    };
  }

  const focusedEdge = edgeById.get(focusedEntityId);
  if (!focusedEdge) {
    return null;
  }

  return {
    kind: "edge",
    edge: focusedEdge,
    sourceNode: nodeById.get(focusedEdge.source.nodeId),
    targetNode: nodeById.get(focusedEdge.target.nodeId),
  };
}

export function GraphScene({
  audience,
  compareFlavor,
  fallbackFocusElementId,
  fitViewCommand,
  primaryFlavor,
  sceneId,
  searchQuery,
  state,
  onFitViewComplete,
  onGraphStateChange,
}: GraphSceneProps) {
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const [timelinePlayback, setTimelinePlayback] = useState(DEFAULT_TIMELINE_PLAYBACK);

  useEffect(() => {
    searchInputRef.current = document.getElementById(
      fallbackFocusElementId,
    ) as HTMLInputElement | null;
  }, [fallbackFocusElementId]);

  const derivation = useMemo(
    () =>
      deriveGraphDerivation(architectureCatalog, {
        audience,
        compareFlavor: compareFlavor ?? undefined,
        expandedNodeIds: state.expandedNodeIds,
        focusedEntityId: state.focusedEntityId ?? undefined,
        primaryFlavor,
        sceneId,
        searchQuery,
      }),
    [
      audience,
      compareFlavor,
      primaryFlavor,
      sceneId,
      searchQuery,
      state.expandedNodeIds,
      state.focusedEntityId,
    ],
  );

  const timeline = useMemo(
    () => buildFlowTimeline(architectureCatalog, primaryFlavor),
    [primaryFlavor],
  );
  const timelineState = useMemo(
    () => resolveTimelineSemanticState(timeline, state.timelinePosition),
    [state.timelinePosition, timeline],
  );
  const reducedMotion = useMemo(() => reducedMotionPreferred(), []);
  const activePulseNodeIds = useMemo(() => {
    const activeEvent = timelineState.activeEvent.reference;
    return activeEvent.kind === "node" ? [activeEvent.nodeId] : [];
  }, [timelineState.activeEvent.reference]);
  const completedPulseNodeIds = useMemo(
    () =>
      timelineState.completedEvents
        .map((event) => event.reference)
        .filter((reference): reference is { kind: "node"; nodeId: string; portId: string } =>
          reference.kind === "node"
        )
        .map((reference) => reference.nodeId),
    [timelineState.completedEvents],
  );
  const pulseEdges = useMemo(
    () =>
      derivePulseEdgeOverlayState({
        reducedMotion,
        semanticState: timelineState,
        visibleEdges: derivation.visibleEdges,
      }),
    [derivation.visibleEdges, reducedMotion, timelineState],
  );

  const evidenceEntity = useMemo(
    () =>
      createEvidenceEntity(
        state.focusedEntityId,
        derivation.visibleNodes,
        derivation.visibleEdges,
      ),
    [derivation.visibleEdges, derivation.visibleNodes, state.focusedEntityId],
  );

  const layoutRequest = useMemo(
    () =>
      buildGraphLayoutRequest({
        edges: derivation.visibleEdges,
        expandedNodeIds: derivation.expandedNodeIds,
        nodePositions: state.nodePositions,
        nodes: derivation.visibleNodes,
        sceneId,
      }),
    [
      derivation.expandedNodeIds,
      derivation.visibleEdges,
      derivation.visibleNodes,
      sceneId,
      state.nodePositions,
    ],
  );

  const updateState = (next: Partial<GraphState>) => {
    onGraphStateChange(
      canonicalGraphState({
        ...state,
        ...next,
        audience,
        compareFlavor,
        primaryFlavor,
        sceneId,
      }),
    );
  };
  const edgeWaypointsById = useMemo(
    () => new Map(state.edgeWaypoints.map((waypoint) => [waypoint.edgeId, waypoint.points])),
    [state.edgeWaypoints],
  );
  const traceNodeId =
    state.traceMode !== "none" && state.focusedEntityId?.startsWith("node.")
      ? state.focusedEntityId
      : null;
  const traceNeighborhood = useMemo(() => {
    if (!traceNodeId || state.traceMode === "none") {
      return derivation.neighborhood;
    }
    if (state.traceMode === "upstream") {
      return {
        downstreamNodeIds: [] as string[],
        upstreamNodeIds: derivation.neighborhood.upstreamNodeIds,
      };
    }
    if (state.traceMode === "downstream") {
      return {
        downstreamNodeIds: derivation.neighborhood.downstreamNodeIds,
        upstreamNodeIds: [] as string[],
      };
    }
    return {
      downstreamNodeIds: [] as string[],
      upstreamNodeIds: [] as string[],
    };
  }, [derivation.neighborhood.downstreamNodeIds, derivation.neighborhood.upstreamNodeIds, state.traceMode, traceNodeId]);

  const handleCollapseNode = (nodeId: string) => {
    const descendantNodeIds = collectDescendantNodeIds(
      nodeId,
      architectureCatalog.graphNodes,
    );
    const collapsedExpandedNodeIds = collapseExpandedNode(
      architectureCatalog,
      state.expandedNodeIds,
      nodeId,
    );
    const descendantEdgeIds = new Set(
      architectureCatalog.graphEdges
        .filter(
          (edge) =>
            descendantNodeIds.has(edge.source.nodeId) || descendantNodeIds.has(edge.target.nodeId),
        )
        .map((edge) => edge.id),
    );
    const hiddenSubtreeEntityIds = new Set([
      ...descendantNodeIds,
      ...descendantEdgeIds,
    ]);
    const focusIsHidden =
      state.focusedEntityId !== null &&
      hiddenSubtreeEntityIds.has(state.focusedEntityId);
    updateState({
      edgeWaypoints: state.edgeWaypoints.filter(
        ({ edgeId }) => !descendantEdgeIds.has(edgeId),
      ),
      expandedNodeIds: collapsedExpandedNodeIds,
      focusedEntityId: focusIsHidden ? nodeId : state.focusedEntityId,
      nodePositions: state.nodePositions.filter(
        ({ nodeId: positionNodeId }) => !descendantNodeIds.has(positionNodeId),
      ),
      traceMode: focusIsHidden ? "none" : state.traceMode,
    });
  };

  return (
    <section aria-label={`${derivation.scene.title} scene`} className="graph-scene-route flight-deck-scene">
      <header className="scene-status-hud">
        <h1>{derivation.scene.title}</h1>
        <p aria-label="Derived graph topology summary" role="status">
          {derivation.visibleNodes.length} nodes, {derivation.visibleEdges.length} edges, timeline step{" "}
          {timelineState.eventIndex + 1}: {timelineState.activeEvent.label}
        </p>
      </header>
      <div className="scene-graph-stage">
        <GraphCanvas
        activePulseNodeIds={activePulseNodeIds}
        audience={audience}
        breadcrumbNodeIds={derivation.breadcrumbNodeIds}
        completedPulseNodeIds={completedPulseNodeIds}
        expandedNodeIds={state.expandedNodeIds}
        edgeWaypoints={edgeWaypointsById}
        fitViewCommand={fitViewCommand ?? undefined}
        focusedEntityId={state.focusedEntityId}
        layoutRequest={layoutRequest}
        layoutService={{ layout: layoutAtlas }}
        neighborhood={traceNeighborhood}
        onCollapseNode={handleCollapseNode}
        onExpandNode={(nodeId) =>
          updateState({
            expandedNodeIds: toggleExpandedNode(state.expandedNodeIds, nodeId),
          })
        }
        onFitViewComplete={onFitViewComplete}
        onFocusBreadcrumb={(nodeId) => updateState({ focusedEntityId: nodeId, traceMode: "none" })}
        onFocusEntity={(entityId) => updateState({ focusedEntityId: entityId, traceMode: "none" })}
        onNodeDragComplete={(position) =>
          updateState({
            nodePositions: replaceNodePosition(state.nodePositions, position),
          })
        }
        onTraceModeChange={(nodeId, mode) =>
          updateState({
            focusedEntityId: nodeId,
            traceMode: mode,
          })
        }
        onWaypointsChange={(update) =>
          updateState({
            edgeWaypoints: replaceEdgeWaypoints(state.edgeWaypoints, update),
          })
        }
        onWaypointsReset={(edgeId) =>
          updateState({
            edgeWaypoints: state.edgeWaypoints.filter((waypoint) => waypoint.edgeId !== edgeId),
          })
        }
        overlay={derivation.overlay}
        pulseEdges={pulseEdges}
        traceMode={state.traceMode}
        visibleEdges={derivation.visibleEdges}
        visibleNodes={derivation.visibleNodes}
        />
        <PulseLayer
          reducedMotion={reducedMotion}
          semanticState={timelineState}
          visibleEdges={derivation.visibleEdges}
        />
        <AccessibilityOutline
          audience={audience}
          expandedNodeIds={derivation.expandedNodeIds}
          onCollapseNode={(nodeId) =>
            handleCollapseNode(nodeId)
          }
          onExpandNode={(nodeId) =>
            updateState({
              expandedNodeIds: toggleExpandedNode(state.expandedNodeIds, nodeId),
            })
          }
          onInspectEntity={(entityId) =>
            updateState({ focusedEntityId: entityId, traceMode: "none" })
          }
          onIsolateEntity={(entityId) =>
            updateState({
              expandedNodeIds: [],
              focusedEntityId: entityId,
              traceMode: entityId.startsWith("node.") ? "isolate" : "none",
            })
          }
          onSelectEntity={(entityId) =>
            updateState({ focusedEntityId: entityId, traceMode: "none" })
          }
          visibleEdges={derivation.visibleEdges}
          visibleNodes={derivation.visibleNodes}
        />
      </div>
      <PulseControls
        isPlaying={timelinePlayback.isPlaying}
        onPause={() => setTimelinePlayback(pauseTimeline)}
        onPlay={() => setTimelinePlayback(playTimeline)}
        onRestart={() => {
          setTimelinePlayback(pauseTimeline);
          updateState({ timelinePosition: 0 });
        }}
        onScrub={(position) => {
          setTimelinePlayback((current) => scrubTimeline(current, position));
          updateState({ timelinePosition: position });
        }}
        reducedMotion={reducedMotion}
        semanticState={timelineState}
        timeline={timeline}
      />

      <EvidenceDrawer
        audience={audience}
        entity={evidenceEntity}
        fallbackFocusRef={searchInputRef}
        getTriggerElement={(entityId) =>
          document.querySelector<HTMLElement>(
            `[data-graph-entity-trigger="true"][data-graph-entity-id="${entityId}"]`,
          )
        }
        onClose={() => updateState({ focusedEntityId: null })}
      />

    </section>
  );
}
