// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureCatalog,
  AudienceLevel,
  ExecutionFlavor,
  GraphEdge,
  GraphNode,
  GraphScene,
} from "./architecture";

export interface GraphDerivationInput {
  sceneId: string;
  audience: AudienceLevel;
  primaryFlavor: ExecutionFlavor;
  compareFlavor?: ExecutionFlavor;
  expandedNodeIds?: readonly string[];
  searchQuery?: string;
  focusedEntityId?: string;
}

export interface DirectedNeighborhood {
  upstreamNodeIds: string[];
  downstreamNodeIds: string[];
}

export interface FlavorOverlay {
  sharedNodeIds: string[];
  primaryOnlyNodeIds: string[];
  compareOnlyNodeIds: string[];
  sharedEdgeIds: string[];
  primaryOnlyEdgeIds: string[];
  compareOnlyEdgeIds: string[];
}

export interface GraphDerivationResult {
  scene: GraphScene;
  visibleNodes: GraphNode[];
  visibleEdges: GraphEdge[];
  visibleNodeIds: string[];
  visibleEdgeIds: string[];
  expandedNodeIds: string[];
  revealedAncestorNodeIds: string[];
  breadcrumbNodeIds: string[];
  overlay: FlavorOverlay;
  neighborhood: DirectedNeighborhood;
}

function normalizeQuery(query: string | undefined): string {
  return query?.trim().toLocaleLowerCase() ?? "";
}

function includesText(values: readonly string[], query: string): boolean {
  return values.some((value) => value.toLocaleLowerCase().includes(query));
}

function orderedIds(ids: Iterable<string>, referenceOrder: readonly string[]): string[] {
  const wanted = new Set(ids);
  return referenceOrder.filter((id) => wanted.has(id));
}

function includeForAudience(node: GraphNode, audience: AudienceLevel): boolean {
  return node.audience.visibility.includes(audience);
}

function includeForFlavor(
  flavors: readonly ExecutionFlavor[],
  selected: ReadonlySet<ExecutionFlavor>,
): boolean {
  return flavors.some((flavor) => selected.has(flavor));
}

function collectDescendantIds(
  nodeId: string,
  nodesById: ReadonlyMap<string, GraphNode>,
): Set<string> {
  const descendants = new Set<string>();
  const pending = [nodeId];
  while (pending.length > 0) {
    const currentId = pending.pop();
    if (!currentId) {
      continue;
    }
    const current = nodesById.get(currentId);
    if (!current) {
      continue;
    }
    for (const childId of current.childIds) {
      if (!descendants.has(childId)) {
        descendants.add(childId);
        pending.push(childId);
      }
    }
  }
  return descendants;
}

function focusBreadcrumbs(
  focusedEntityId: string | undefined,
  nodesById: ReadonlyMap<string, GraphNode>,
  visibleNodeIds: ReadonlySet<string>,
): string[] {
  if (!focusedEntityId || !visibleNodeIds.has(focusedEntityId)) {
    return [];
  }
  const crumbs: string[] = [];
  let current = nodesById.get(focusedEntityId);
  while (current) {
    crumbs.push(current.id);
    if (!current.parentId) {
      break;
    }
    current = nodesById.get(current.parentId);
  }
  return crumbs.reverse();
}

function deriveDirectedNeighborhood(
  focusedEntityId: string | undefined,
  visibleNodeIds: ReadonlySet<string>,
  edges: readonly GraphEdge[],
  nodeOrder: readonly string[],
): DirectedNeighborhood {
  if (!focusedEntityId || !visibleNodeIds.has(focusedEntityId)) {
    return { upstreamNodeIds: [], downstreamNodeIds: [] };
  }

  const trace = (direction: "upstream" | "downstream"): string[] => {
    const seen = new Set<string>();
    const pending = [focusedEntityId];
    while (pending.length > 0) {
      const current = pending.shift();
      if (!current) {
        continue;
      }
      for (const edge of edges) {
        const next =
          direction === "upstream" && edge.target.nodeId === current
            ? edge.source.nodeId
            : direction === "downstream" && edge.source.nodeId === current
              ? edge.target.nodeId
              : undefined;
        if (
          next &&
          next !== focusedEntityId &&
          visibleNodeIds.has(next) &&
          !seen.has(next)
        ) {
          seen.add(next);
          pending.push(next);
        }
      }
    }
    return orderedIds(seen, nodeOrder);
  };

  return {
    upstreamNodeIds: trace("upstream"),
    downstreamNodeIds: trace("downstream"),
  };
}

function buildOverlay(
  visibleNodes: readonly GraphNode[],
  visibleEdges: readonly GraphEdge[],
  primaryFlavor: ExecutionFlavor,
  compareFlavor: ExecutionFlavor | undefined,
  nodeOrder: readonly string[],
  edgeOrder: readonly string[],
): FlavorOverlay {
  if (!compareFlavor) {
    return {
      sharedNodeIds: [],
      primaryOnlyNodeIds: visibleNodes.map(({ id }) => id),
      compareOnlyNodeIds: [],
      sharedEdgeIds: [],
      primaryOnlyEdgeIds: visibleEdges.map(({ id }) => id),
      compareOnlyEdgeIds: [],
    };
  }

  const sharedNodeIds = new Set<string>();
  const primaryOnlyNodeIds = new Set<string>();
  const compareOnlyNodeIds = new Set<string>();

  for (const node of visibleNodes) {
    const inPrimary = node.flavors.includes(primaryFlavor);
    const inCompare = node.flavors.includes(compareFlavor);
    if (inPrimary && inCompare) {
      sharedNodeIds.add(node.id);
    } else if (inPrimary) {
      primaryOnlyNodeIds.add(node.id);
    } else if (inCompare) {
      compareOnlyNodeIds.add(node.id);
    }
  }

  const sharedEdgeIds = new Set<string>();
  const primaryOnlyEdgeIds = new Set<string>();
  const compareOnlyEdgeIds = new Set<string>();

  for (const edge of visibleEdges) {
    const inPrimary = edge.flavors.includes(primaryFlavor);
    const inCompare = edge.flavors.includes(compareFlavor);
    if (inPrimary && inCompare) {
      sharedEdgeIds.add(edge.id);
    } else if (inPrimary) {
      primaryOnlyEdgeIds.add(edge.id);
    } else if (inCompare) {
      compareOnlyEdgeIds.add(edge.id);
    }
  }

  return {
    sharedNodeIds: orderedIds(sharedNodeIds, nodeOrder),
    primaryOnlyNodeIds: orderedIds(primaryOnlyNodeIds, nodeOrder),
    compareOnlyNodeIds: orderedIds(compareOnlyNodeIds, nodeOrder),
    sharedEdgeIds: orderedIds(sharedEdgeIds, edgeOrder),
    primaryOnlyEdgeIds: orderedIds(primaryOnlyEdgeIds, edgeOrder),
    compareOnlyEdgeIds: orderedIds(compareOnlyEdgeIds, edgeOrder),
  };
}

export function selectSceneById(
  catalog: ArchitectureCatalog,
  sceneId: string,
): GraphScene {
  const scene = catalog.graphScenes.find((candidate) => candidate.id === sceneId);
  if (!scene) {
    throw new Error(`Unknown scene: ${sceneId}`);
  }
  return scene;
}

export function toggleExpandedNode(
  expandedNodeIds: readonly string[],
  nodeId: string,
): string[] {
  const expanded = new Set(expandedNodeIds);
  if (expanded.has(nodeId)) {
    expanded.delete(nodeId);
  } else {
    expanded.add(nodeId);
  }
  return [...expanded].sort();
}

export function collapseExpandedNode(
  catalog: ArchitectureCatalog,
  expandedNodeIds: readonly string[],
  nodeId: string,
): string[] {
  const nodesById = new Map(catalog.graphNodes.map((node) => [node.id, node]));
  const idsToRemove = collectDescendantIds(nodeId, nodesById);
  idsToRemove.add(nodeId);
  return expandedNodeIds.filter((expandedId) => !idsToRemove.has(expandedId));
}

export function deriveGraphDerivation(
  catalog: ArchitectureCatalog,
  input: GraphDerivationInput,
): GraphDerivationResult {
  const scene = selectSceneById(catalog, input.sceneId);
  const nodeOrder = scene.nodeIds;
  const edgeOrder = scene.edgeIds;

  const nodesById = new Map(catalog.graphNodes.map((node) => [node.id, node]));
  const edgesById = new Map(catalog.graphEdges.map((edge) => [edge.id, edge]));
  const selectedFlavors = new Set<ExecutionFlavor>([input.primaryFlavor]);
  if (input.compareFlavor) {
    selectedFlavors.add(input.compareFlavor);
  }

  const audienceDepth = scene.audience.defaultDepth[input.audience];
  const sceneNodes = nodeOrder
    .map((id) => nodesById.get(id))
    .filter((node): node is GraphNode => node !== undefined)
    .filter(
      (node) =>
        includeForAudience(node, input.audience) &&
        includeForFlavor(node.flavors, selectedFlavors),
    );
  const sceneNodeIds = new Set(sceneNodes.map(({ id }) => id));

  const manuallyExpanded = new Set(
    (input.expandedNodeIds ?? []).filter((nodeId) => sceneNodeIds.has(nodeId)),
  );
  const automaticallyExpanded = new Set(
    sceneNodes
      .filter(({ tier }) => tier < audienceDepth)
      .map(({ id }) => id),
  );

  const query = normalizeQuery(input.searchQuery);
  const searchMatchedNodeIds =
    query.length === 0
      ? []
      : sceneNodes
          .filter((node) =>
            includesText(
              [
                ...Object.values(node.title),
                ...Object.values(node.summary),
                node.id,
              ],
              query,
            ),
          )
          .map(({ id }) => id);

  const revealedAncestorNodeIds = new Set<string>();
  for (const nodeId of searchMatchedNodeIds) {
    let current = nodesById.get(nodeId);
    while (current?.parentId && sceneNodeIds.has(current.parentId)) {
      if (
        !manuallyExpanded.has(current.parentId) &&
        !automaticallyExpanded.has(current.parentId)
      ) {
        revealedAncestorNodeIds.add(current.parentId);
      }
      current = nodesById.get(current.parentId);
    }
  }

  const effectiveExpanded = new Set<string>([
    ...manuallyExpanded,
    ...automaticallyExpanded,
    ...revealedAncestorNodeIds,
  ]);

  const visibleNodeIds = new Set<string>();
  const roots = sceneNodes.filter(
    (node) => !node.parentId || !sceneNodeIds.has(node.parentId),
  );
  const pending = [...roots];
  while (pending.length > 0) {
    const current = pending.shift();
    if (!current || visibleNodeIds.has(current.id)) {
      continue;
    }
    visibleNodeIds.add(current.id);
    if (!effectiveExpanded.has(current.id)) {
      continue;
    }
    for (const childId of current.childIds) {
      if (!sceneNodeIds.has(childId)) {
        continue;
      }
      const child = nodesById.get(childId);
      if (child) {
        pending.push(child);
      }
    }
  }

  const visibleNodes = nodeOrder
    .filter((nodeId) => visibleNodeIds.has(nodeId))
    .map((nodeId) => nodesById.get(nodeId))
    .filter((node): node is GraphNode => node !== undefined);
  const visibleEdges = edgeOrder
    .map((edgeId) => edgesById.get(edgeId))
    .filter((edge): edge is GraphEdge => edge !== undefined)
    .filter(
      (edge) =>
        includeForFlavor(edge.flavors, selectedFlavors) &&
        visibleNodeIds.has(edge.source.nodeId) &&
        visibleNodeIds.has(edge.target.nodeId),
    );

  const visibleNodeIdList = visibleNodes.map(({ id }) => id);
  const visibleEdgeIdList = visibleEdges.map(({ id }) => id);
  return {
    scene,
    visibleNodes,
    visibleEdges,
    visibleNodeIds: visibleNodeIdList,
    visibleEdgeIds: visibleEdgeIdList,
    expandedNodeIds: orderedIds(effectiveExpanded, nodeOrder),
    revealedAncestorNodeIds: orderedIds(revealedAncestorNodeIds, nodeOrder),
    breadcrumbNodeIds: focusBreadcrumbs(
      input.focusedEntityId,
      nodesById,
      visibleNodeIds,
    ),
    overlay: buildOverlay(
      visibleNodes,
      visibleEdges,
      input.primaryFlavor,
      input.compareFlavor,
      nodeOrder,
      edgeOrder,
    ),
    neighborhood: deriveDirectedNeighborhood(
      input.focusedEntityId,
      visibleNodeIds,
      visibleEdges,
      nodeOrder,
    ),
  };
}
