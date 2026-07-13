// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useEffect,
  useMemo,
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
  buildFlowTimeline,
  resolveTimelineSemanticState,
} from "../../domain/flow-timeline";
import {
  layoutAtlas,
  type LayoutRequest,
} from "../atlas/layout";
import { AccessibilityOutline } from "./accessibility-outline";
import { EvidenceDrawer, type EvidenceDrawerEntity } from "./evidence-drawer";
import { GraphCanvas } from "./graph-canvas";
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

  const nodes = [...input.nodes]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((node) => ({
      bandId: bandByTier.get(node.tier) ?? "tier.0",
      id: node.id,
      parentId: node.parentId && nodeIds.has(node.parentId) ? node.parentId : undefined,
    }));

  const edges = [...input.edges]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((edge) => ({ from: edge.source.nodeId, id: edge.id, to: edge.target.nodeId }));

  const relayoutNodeIds = nodes.map(({ id }) => id);
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
  onGraphStateChange,
}: GraphSceneProps) {
  const searchInputRef = useRef<HTMLInputElement | null>(null);

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

  return (
    <section aria-label={`${derivation.scene.title} scene`} className="graph-scene-route">
      <h1>{derivation.scene.title}</h1>
      <p aria-label="Derived graph topology summary" role="status">
        {derivation.visibleNodes.length} nodes, {derivation.visibleEdges.length} edges, timeline step{" "}
        {timelineState.eventIndex + 1}: {timelineState.activeEvent.label}
      </p>
      <label>
        <span>Timeline position</span>
        <input
          aria-label="Timeline position"
          max={1}
          min={0}
          onChange={(event) =>
            updateState({ timelinePosition: Number(event.currentTarget.value) })
          }
          step={0.01}
          type="range"
          value={state.timelinePosition}
        />
      </label>
      <GraphCanvas
        audience={audience}
        fitViewCommand={fitViewCommand ?? undefined}
        focusedEntityId={state.focusedEntityId}
        layoutRequest={layoutRequest}
        layoutService={{ layout: layoutAtlas }}
        neighborhood={derivation.neighborhood}
        onFocusEntity={(entityId) => updateState({ focusedEntityId: entityId })}
        overlay={derivation.overlay}
        visibleEdges={derivation.visibleEdges}
        visibleNodes={derivation.visibleNodes}
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

      <AccessibilityOutline
        audience={audience}
        expandedNodeIds={derivation.expandedNodeIds}
        onCollapseNode={(nodeId) =>
          updateState({
            expandedNodeIds: collapseExpandedNode(
              architectureCatalog,
              state.expandedNodeIds,
              nodeId,
            ),
          })
        }
        onExpandNode={(nodeId) =>
          updateState({
            expandedNodeIds: toggleExpandedNode(state.expandedNodeIds, nodeId),
          })
        }
        onInspectEntity={(entityId) => updateState({ focusedEntityId: entityId })}
        onIsolateEntity={(entityId) =>
          updateState({
            expandedNodeIds: [],
            focusedEntityId: entityId,
          })
        }
        onSelectEntity={(entityId) => updateState({ focusedEntityId: entityId })}
        visibleEdges={derivation.visibleEdges}
        visibleNodes={derivation.visibleNodes}
      />
    </section>
  );
}
