// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Background,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  useReactFlow,
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
  GraphPulseChannelState,
  GraphFlavorClass,
  GraphManualNodePosition,
  GraphNodePortView,
  GraphPathState,
  GraphPulseState,
  GraphRelayoutState,
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

export function completeNodeDrag(
  node: Pick<Node, "id" | "position">,
  complete: ((position: GraphManualNodePosition) => void) | undefined,
): void {
  complete?.({
    nodeId: node.id,
    x: node.position.x,
    y: node.position.y,
  });
}

/**
 * Frames the graph in the viewport using the live React Flow store (via
 * `useReactFlow`), which is the same source the working zoom controls use.
 * The `onInit`-captured instance's `fitView` silently no-ops here, so all
 * fitting must go through this in-tree controller instead.
 *
 * Auto-fits when the layout is ready and whenever the node set changes
 * (scene switch), and honours the manual "Fit graph" command. `useNodesInitialized`
 * never flips true for these nodes, so readiness is keyed off the layout instead
 * and explicit node width/height give `fitView` real bounds to frame.
 */
function GraphFitController({
  layoutKey,
  fitCommand,
  onFitComplete,
  handledRequestIds,
}: {
  layoutKey: string | null;
  fitCommand?: { requestId: number; padding?: number };
  onFitComplete?(requestId: number): void;
  handledRequestIds: { current: Set<number> };
}): null {
  const { fitView } = useReactFlow();
  const ready = layoutKey !== null;

  useEffect(() => {
    if (!ready) {
      return;
    }
    // Defer a frame so the freshly laid-out nodes are in the store before we
    // frame them (explicit node width/height give fitView real bounds).
    const frame = requestAnimationFrame(() => {
      void fitView({ padding: 0.16, maxZoom: 1.15, duration: 250 });
    });
    return () => cancelAnimationFrame(frame);
  }, [ready, layoutKey, fitView]);

  useEffect(() => {
    if (!fitCommand || handledRequestIds.current.has(fitCommand.requestId)) {
      return;
    }
    handledRequestIds.current.add(fitCommand.requestId);
    const requestId = fitCommand.requestId;
    void fitView({
      padding: fitCommand.padding ?? 0.16,
      maxZoom: 1.15,
      duration: 250,
    }).then(() => onFitComplete?.(requestId));
  }, [fitCommand, fitView, onFitComplete, handledRequestIds]);

  return null;
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
  traceNodeId: string | null,
  traceMode: GraphCanvasProps["traceMode"],
  focusedEntityId: string | null,
  upstreamNodeIds: ReadonlySet<string>,
  downstreamNodeIds: ReadonlySet<string>,
): GraphPathState {
  if (nodeId === traceNodeId || nodeId === focusedEntityId) {
    return "focused";
  }
  if (traceMode === "upstream" && upstreamNodeIds.has(nodeId)) {
    return "upstream";
  }
  if (traceMode === "downstream" && downstreamNodeIds.has(nodeId)) {
    return "downstream";
  }
  if (traceMode === "none" && upstreamNodeIds.has(nodeId)) {
    return "upstream";
  }
  if (traceMode === "none" && downstreamNodeIds.has(nodeId)) {
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
  traceNodeId: string | null,
  traceMode: GraphCanvasProps["traceMode"],
  focusedEntityId: string | null,
  upstreamNodeIds: ReadonlySet<string>,
  downstreamNodeIds: ReadonlySet<string>,
): GraphPathState {
  if (focusedEntityId === edge.id) {
    return "focused";
  }
  if (!focusedEntityId && !traceNodeId) {
    return "default";
  }

  const sourceId = edge.source.nodeId;
  const targetId = edge.target.nodeId;
  if (traceMode === "isolate" && traceNodeId) {
    return sourceId === traceNodeId || targetId === traceNodeId ? "focused" : "default";
  }

  if (
    ((targetId === focusedEntityId || targetId === traceNodeId) && upstreamNodeIds.has(sourceId)) ||
    (upstreamNodeIds.has(sourceId) && upstreamNodeIds.has(targetId))
  ) {
    return "upstream";
  }

  if (
    ((sourceId === focusedEntityId || sourceId === traceNodeId) && downstreamNodeIds.has(targetId)) ||
    (downstreamNodeIds.has(sourceId) && downstreamNodeIds.has(targetId))
  ) {
    return "downstream";
  }

  return "default";
}

function classifyEdgePulseState(
  edgeId: string,
  activeEdgeIds: ReadonlySet<string>,
  completedEdgeIds: ReadonlySet<string>,
): GraphPulseState {
  if (activeEdgeIds.has(edgeId)) {
    return "active";
  }
  if (completedEdgeIds.has(edgeId)) {
    return "completed";
  }
  return "idle";
}

function classifyEdgePulseChannelState(
  channel: GraphCanvasProps["visibleEdges"][number]["channel"],
  activeChannels: ReadonlySet<GraphCanvasProps["visibleEdges"][number]["channel"]>,
  completedChannels: ReadonlySet<GraphCanvasProps["visibleEdges"][number]["channel"]>,
): GraphPulseChannelState {
  if (activeChannels.has(channel)) {
    return "active";
  }
  if (completedChannels.has(channel)) {
    return "completed";
  }
  return "idle";
}

export function resolveCanvasPulseEdgeState(
  edge: GraphCanvasProps["visibleEdges"][number],
  pulseEdges: GraphCanvasProps["pulseEdges"],
): {
  channelState: GraphPulseChannelState;
  phase: GraphPulseState;
  reducedMotion: boolean;
} {
  const activeEdgeIds = new Set(pulseEdges?.activeEdgeIds ?? []);
  const completedEdgeIds = new Set(pulseEdges?.completedEdgeIds ?? []);
  const activeChannels = new Set(pulseEdges?.activeChannels ?? []);
  const completedChannels = new Set(pulseEdges?.completedChannels ?? []);
  return {
    channelState: classifyEdgePulseChannelState(edge.channel, activeChannels, completedChannels),
    phase: classifyEdgePulseState(edge.id, activeEdgeIds, completedEdgeIds),
    reducedMotion: pulseEdges?.reducedMotion ?? false,
  };
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
  // Owned here (a stable component) so fit-command de-duplication survives the
  // fit controller remounting when the graph re-lays-out.
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
  const expandedNodeIds = useMemo(
    () => new Set(props.expandedNodeIds ?? []),
    [props.expandedNodeIds],
  );
  const activePulseNodeIds = useMemo(
    () => new Set(props.activePulseNodeIds ?? []),
    [props.activePulseNodeIds],
  );
  const completedPulseNodeIds = useMemo(
    () => new Set(props.completedPulseNodeIds ?? []),
    [props.completedPulseNodeIds],
  );
  const traceMode = props.traceMode ?? "none";
  const traceNodeId =
    props.focusedEntityId?.startsWith("node.") && traceMode !== "none"
      ? props.focusedEntityId
      : null;

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
  const preservedNodeIds = useMemo(
    () =>
      new Set(
        layoutState.status === "ready"
          ? layoutState.result.partialRelayout?.preservedManualNodeIds ?? []
          : [],
      ),
    [layoutState],
  );
  const relaidOutNodeIds = useMemo(
    () =>
      new Set(
        layoutState.status === "ready"
          ? layoutState.result.partialRelayout?.relaidOutNodeIds ?? []
          : [],
      ),
    [layoutState],
  );

  const nodes: Node<RuntimeGraphNodeData>[] = useMemo(
    () =>
      layoutState.status !== "ready"
        ? []
        : props.visibleNodes.map((node) => {
            const pulseState: GraphPulseState = activePulseNodeIds.has(node.id)
              ? "active"
              : completedPulseNodeIds.has(node.id)
                ? "completed"
                : "idle";
            const relayoutState: GraphRelayoutState = preservedNodeIds.has(node.id)
              ? "preserved"
              : relaidOutNodeIds.has(node.id)
                ? "relaid-out"
                : "canonical";
            return {
              data: {
              audience: props.audience,
              expanded: expandedNodeIds.has(node.id),
              flavorClass: classifyFlavor(
                node.id,
                props.overlay.sharedNodeIds,
                props.overlay.primaryOnlyNodeIds,
                props.overlay.compareOnlyNodeIds,
              ),
              node,
              onCollapse: props.onCollapseNode,
              onExpand: props.onExpandNode,
              onSelect: props.onFocusEntity,
              onTraceModeChange: props.onTraceModeChange,
              pathState: classifyNodePathState(
                node.id,
                traceNodeId,
                traceMode,
                props.focusedEntityId,
                upstreamNodeIds,
                downstreamNodeIds,
              ),
              ports: portViewsByNode.get(node.id) ?? [],
              pulseState,
              relayoutState,
              traceMode,
            },
            draggable: true,
            dragHandle: ".graph-node-drag-handle",
            id: node.id,
            // Explicit width/height give React Flow real bounds so fitView()
            // can frame the graph; without them getNodesBounds is empty and
            // fitView (and the Fit graph button) silently no-op.
            width: 320,
            height: 156,
            position: positionsByNodeId.get(node.id) ?? { x: 0, y: 0 },
            style: { width: 320 },
            type: "runtimeNode",
            };
          }),
    [
      activePulseNodeIds,
      completedPulseNodeIds,
      downstreamNodeIds,
      expandedNodeIds,
      layoutState.status,
      portViewsByNode,
      positionsByNodeId,
      preservedNodeIds,
      props.audience,
      props.focusedEntityId,
      props.onCollapseNode,
      props.onExpandNode,
      props.onFocusEntity,
      props.onTraceModeChange,
      props.overlay,
      traceMode,
      traceNodeId,
      props.visibleNodes,
      relaidOutNodeIds,
      upstreamNodeIds,
    ],
  );

  const edges: Edge<RuntimeGraphEdgeData>[] = useMemo(
    () =>
      props.visibleEdges.map((edge) => {
        const pathState = classifyEdgePathState(
          edge,
          traceNodeId,
          traceMode,
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
        const pulseEdgeState = resolveCanvasPulseEdgeState(edge, props.pulseEdges);
        return {
          className: `graph-edge-path-${pathState}`,
          data: {
            edge,
            flavorClass,
            onSelect: props.onFocusEntity,
            onWaypointsChange: props.onWaypointsChange,
            onWaypointsReset: props.onWaypointsReset,
            pathState,
            pulseEdgeState,
            waypoints: props.edgeWaypoints?.get(edge.id),
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
      props.onWaypointsChange,
      props.onWaypointsReset,
      props.onFocusEntity,
      props.overlay,
      props.pulseEdges,
      traceMode,
      props.visibleEdges,
      props.edgeWaypoints,
      traceNodeId,
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
    <section
      aria-label="Graph canvas"
      className="graph-canvas-shell"
      data-active-pulse-channels={props.pulseEdges?.activeChannels.join(",") ?? ""}
      data-active-pulse-edge-ids={props.pulseEdges?.activeEdgeIds.join(",") ?? ""}
      data-reduced-motion={String(props.pulseEdges?.reducedMotion ?? false)}
    >
      {(props.breadcrumbNodeIds?.length ?? 0) > 0 ? (
        <nav aria-label="Graph focus context" className="graph-focus-breadcrumbs">
          <ol>
            {props.breadcrumbNodeIds?.map((nodeId) => {
              const node = props.visibleNodes.find(({ id }) => id === nodeId);
              const label = node?.title[props.audience] ?? nodeId;
              return (
                <li key={nodeId}>
                  <button
                    aria-current={
                      nodeId === props.focusedEntityId ? "location" : undefined
                    }
                    onClick={() => props.onFocusBreadcrumb?.(nodeId)}
                    type="button"
                  >
                    {label}
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
      ) : null}
      <p aria-label="Graph layout status" className="canvas-status-chip" role="status">
        {layoutState.result.degraded
          ? `Graph layout degraded; deterministic fallback in use. ${layoutState.result.reason ?? ""}`.trim()
          : "Graph layout ready."}
      </p>
      <div className="graph-canvas-stage">
        <ReactFlow
          colorMode="dark"
          edgeTypes={edgeTypes}
          edges={edges}
          minZoom={0.4}
          nodeTypes={nodeTypes}
          nodes={nodes}
          nodesDraggable
          onNodeDragStop={(_event, node) =>
            completeNodeDrag(node, props.onNodeDragComplete)
          }
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#22222c" gap={26} size={1} />
          <GraphFitController
            fitCommand={props.fitViewCommand}
            handledRequestIds={handledFitRequestIds}
            layoutKey={
              layoutState.status === "ready"
                ? layoutState.result.positions.map(({ id }) => id).join(",")
                : null
            }
            onFitComplete={props.onFitViewComplete}
          />
          <Panel className="graph-canvas-controls-panel" position="bottom-left">
            <div aria-label="Graph viewport controls" role="group">
              <Controls showInteractive={false} />
            </div>
          </Panel>
          <Panel className="graph-canvas-minimap-panel" position="bottom-right">
            <div aria-label="Graph canvas minimap" role="region">
              <MiniMap
                ariaLabel="Graph canvas minimap"
                className="graph-canvas-minimap"
                maskColor="rgba(16, 18, 20, 0.36)"
                nodeColor="#6f7882"
                pannable={false}
                zoomable={false}
              />
            </div>
          </Panel>
        </ReactFlow>
      </div>
    </section>
  );
}
