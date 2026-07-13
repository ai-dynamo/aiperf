// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import { useEffect, useMemo, useRef, useState } from "react";

import { deterministicFallbackLayout, type LayoutResult } from "../atlas/layout";
import { RuntimeGraphEdge, edgeMarker, type RuntimeGraphEdgeData } from "./graph-edges";
import { RuntimeGraphNode, type RuntimeGraphNodeData } from "./graph-nodes";
import type {
  GraphCanvasProps,
  GraphFlavorClass,
  GraphNodePortView,
  GraphPathState,
} from "./types";
import "@xyflow/react/dist/style.css";

interface GraphLayoutState {
  result: LayoutResult;
  status: "ready";
}

interface GraphLayoutLoadingState {
  status: "loading";
}

type CanvasLayoutState = GraphLayoutState | GraphLayoutLoadingState;

export async function fitGraphView(
  instance: Pick<ReactFlowInstance, "fitView">,
  nodeIds: readonly string[],
  padding = 0.14,
): Promise<void> {
  await instance.fitView({
    nodes: nodeIds.map((id) => ({ id })),
    padding,
  });
}

function classifyFlavor(
  entityId: string,
  sharedIds: readonly string[],
  primaryOnlyIds: readonly string[],
  compareOnlyIds: readonly string[],
): GraphFlavorClass {
  if (sharedIds.includes(entityId)) {
    return "shared";
  }
  if (compareOnlyIds.includes(entityId)) {
    return "compare-only";
  }
  if (primaryOnlyIds.includes(entityId)) {
    return "primary-only";
  }
  throw new Error(`flavor overlay does not classify ${entityId}`);
}

function classifyNodePathState(
  nodeId: string,
  focusedEntityId: string | null,
  upstreamNodeIds: ReadonlySet<string>,
  downstreamNodeIds: ReadonlySet<string>,
): GraphPathState {
  if (nodeId === focusedEntityId) {
    return "focused";
  }
  if (upstreamNodeIds.has(nodeId)) {
    return "upstream";
  }
  if (downstreamNodeIds.has(nodeId)) {
    return "downstream";
  }
  return "default";
}

function classifyPortDirections(input: GraphCanvasProps): Map<string, GraphNodePortView[]> {
  const sourcePorts = new Set(input.visibleEdges.map((edge) => edge.source.portId));
  const targetPorts = new Set(input.visibleEdges.map((edge) => edge.target.portId));

  return new Map(
    input.visibleNodes.map((node) => [
      node.id,
      node.seamPorts.map((port) => {
        const source = sourcePorts.has(port.id);
        const target = targetPorts.has(port.id);
        const direction: GraphNodePortView["direction"] = source && target
          ? "source+target"
          : source
            ? "source"
            : target
              ? "target"
              : "none";
        return {
          channel: port.channel,
          direction,
          id: port.id,
          name: port.name,
        };
      }),
    ]),
  );
}

function classifyEdgePathState(
  edge: GraphCanvasProps["visibleEdges"][number],
  focusedEntityId: string | null,
  upstreamNodeIds: ReadonlySet<string>,
  downstreamNodeIds: ReadonlySet<string>,
): GraphPathState {
  if (focusedEntityId === edge.id) {
    return "focused";
  }
  if (!focusedEntityId) {
    return "default";
  }

  const sourceId = edge.source.nodeId;
  const targetId = edge.target.nodeId;

  if (
    (targetId === focusedEntityId && upstreamNodeIds.has(sourceId)) ||
    (upstreamNodeIds.has(sourceId) && upstreamNodeIds.has(targetId))
  ) {
    return "upstream";
  }

  if (
    (sourceId === focusedEntityId && downstreamNodeIds.has(targetId)) ||
    (downstreamNodeIds.has(sourceId) && downstreamNodeIds.has(targetId))
  ) {
    return "downstream";
  }

  return "default";
}

const nodeTypes = {
  runtimeNode: RuntimeGraphNode,
};

const edgeTypes = {
  runtimeEdge: RuntimeGraphEdge,
};

export function GraphCanvas(props: GraphCanvasProps) {
  const [layoutState, setLayoutState] = useState<CanvasLayoutState>({
    status: "loading",
  });
  const [instance, setInstance] = useState<ReactFlowInstance | null>(null);
  const handledFitRequestIds = useRef(new Set<number>());

  useEffect(() => {
    let active = true;
    setLayoutState({ status: "loading" });
    void props.layoutService
      .layout(props.layoutRequest)
      .catch((error: unknown) =>
        deterministicFallbackLayout(
          props.layoutRequest,
          error instanceof Error ? error.message : String(error),
        ),
      )
      .then((result) => {
        if (!active) {
          return;
        }
        setLayoutState({ result, status: "ready" });
      });
    return () => {
      active = false;
    };
  }, [props.layoutRequest, props.layoutService]);

  const upstreamNodeIds = useMemo(
    () => new Set(props.neighborhood.upstreamNodeIds),
    [props.neighborhood.upstreamNodeIds],
  );
  const downstreamNodeIds = useMemo(
    () => new Set(props.neighborhood.downstreamNodeIds),
    [props.neighborhood.downstreamNodeIds],
  );
  const portViewsByNode = useMemo(() => classifyPortDirections(props), [props]);

  useEffect(() => {
    if (
      !instance ||
      layoutState.status !== "ready" ||
      !props.fitViewCommand ||
      handledFitRequestIds.current.has(props.fitViewCommand.requestId)
    ) {
      return;
    }
    const requestId = props.fitViewCommand.requestId;
    handledFitRequestIds.current.add(requestId);
    let active = true;
    void fitGraphView(
      instance,
      layoutState.result.positions.map(({ id }) => id),
      props.fitViewCommand.padding,
    ).then(() => {
      if (active) {
        props.onFitViewComplete?.(requestId);
      }
    });
    return () => {
      active = false;
    };
  }, [
    instance,
    layoutState,
    props.fitViewCommand,
    props.onFitViewComplete,
  ]);

  const positionsByNodeId = useMemo(
    () =>
      layoutState.status === "ready"
        ? new Map(
            layoutState.result.positions.map((position) => [
              position.id,
              { x: position.x, y: position.y },
            ]),
          )
        : new Map<string, { x: number; y: number }>(),
    [layoutState],
  );

  const nodes: Node<RuntimeGraphNodeData>[] = useMemo(
    () =>
      layoutState.status !== "ready"
        ? []
        : props.visibleNodes.map((node) => ({
            data: {
              audience: props.audience,
              flavorClass: classifyFlavor(
                node.id,
                props.overlay.sharedNodeIds,
                props.overlay.primaryOnlyNodeIds,
                props.overlay.compareOnlyNodeIds,
              ),
              node,
              onSelect: props.onFocusEntity,
              pathState: classifyNodePathState(
                node.id,
                props.focusedEntityId,
                upstreamNodeIds,
                downstreamNodeIds,
              ),
              ports: portViewsByNode.get(node.id) ?? [],
            },
            draggable: false,
            id: node.id,
            position: positionsByNodeId.get(node.id) ?? { x: 0, y: 0 },
            style: { width: 280 },
            type: "runtimeNode",
          })),
    [
      downstreamNodeIds,
      layoutState.status,
      portViewsByNode,
      positionsByNodeId,
      props.audience,
      props.focusedEntityId,
      props.onFocusEntity,
      props.overlay,
      props.visibleNodes,
      upstreamNodeIds,
    ],
  );

  const edges: Edge<RuntimeGraphEdgeData>[] = useMemo(
    () =>
      props.visibleEdges.map((edge) => {
        const pathState = classifyEdgePathState(
          edge,
          props.focusedEntityId,
          upstreamNodeIds,
          downstreamNodeIds,
        );
        const flavorClass = classifyFlavor(
          edge.id,
          props.overlay.sharedEdgeIds,
          props.overlay.primaryOnlyEdgeIds,
          props.overlay.compareOnlyEdgeIds,
        );
        return {
          className: `graph-edge-path-${pathState}`,
          data: {
            edge,
            flavorClass,
            onSelect: props.onFocusEntity,
            pathState,
          },
          id: edge.id,
          markerEnd: edgeMarker(
            flavorClass,
            edge.status.state === "planned",
          ),
          source: edge.source.nodeId,
          target: edge.target.nodeId,
          type: "runtimeEdge",
        };
      }),
    [
      downstreamNodeIds,
      props.focusedEntityId,
      props.onFocusEntity,
      props.overlay,
      props.visibleEdges,
      upstreamNodeIds,
    ],
  );

  if (layoutState.status === "loading") {
    return (
      <p aria-label="Graph layout status" role="status">
        Positioning graph layout...
      </p>
    );
  }

  return (
    <section aria-label="Graph canvas">
      <p aria-label="Graph layout status" role="status">
        {layoutState.result.degraded
          ? `Graph layout degraded; deterministic fallback in use. ${layoutState.result.reason ?? ""}`.trim()
          : "Graph layout ready."}
      </p>
      <div style={{ height: 620, width: "100%" }}>
        <ReactFlow
          colorMode="dark"
          edgeTypes={edgeTypes}
          edges={edges}
          minZoom={0.2}
          nodeTypes={nodeTypes}
          nodes={nodes}
          nodesDraggable={false}
          onInit={setInstance}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} size={1} />
          <div aria-label="Graph viewport controls" role="group">
            <Controls showInteractive={false} />
          </div>
          <div aria-label="Graph canvas minimap" role="region">
            <MiniMap
              ariaLabel="Graph canvas minimap"
              maskColor="rgba(16, 18, 20, 0.76)"
              nodeColor="#6f7882"
            />
          </div>
        </ReactFlow>
      </div>
    </section>
  );
}
