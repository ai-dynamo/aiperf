// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureCatalog,
  ExecutionFlavor,
  FlowChannel,
  GraphEdge,
  GraphNode,
} from "./architecture";

export interface FlowTimelineEvent {
  id: string;
  step: number;
  channel: FlowChannel;
  flavor: ExecutionFlavor | "shared";
  sceneId: string;
  reference: FlowTimelineReference;
  label: string;
}

export type FlowTimelineReference =
  | { kind: "node"; nodeId: string; portId: string }
  | { kind: "edge"; edgeId: string };

export interface TimelinePlaybackState {
  isPlaying: boolean;
  position: number;
}

export interface TimelineSemanticState {
  position: number;
  eventIndex: number;
  activeEvent: FlowTimelineEvent;
  completedEvents: FlowTimelineEvent[];
}

export interface TimelineRenderingState extends TimelineSemanticState {
  presentation: {
    motion: "animated" | "reduced";
    animateTransitions: boolean;
  };
}

type TimelineEventDefinition =
  | {
      sceneId: string;
      reference: Extract<FlowTimelineReference, { kind: "node" }>;
    }
  | {
      sceneId: string;
      reference: Extract<FlowTimelineReference, { kind: "edge" }>;
    };

const SHARED_TIMELINE_PREFIX: readonly TimelineEventDefinition[] = [
  {
    sceneId: "scene.runner-protocol-registries",
    reference: {
      kind: "node",
      nodeId: "node.runner-protocol-registries",
      portId: "port.runner.in",
    },
  },
  {
    sceneId: "scene.scheduling-phase-lifecycle",
    reference: {
      kind: "node",
      nodeId: "node.scheduling-phase-lifecycle",
      portId: "port.schedule.in",
    },
  },
  {
    sceneId: "scene.dataset-segment-pipeline",
    reference: {
      kind: "node",
      nodeId: "node.dataset-segment-pipeline",
      portId: "port.dataset.out",
    },
  },
];

const SHARED_TIMELINE_SUFFIX: readonly TimelineEventDefinition[] = [
  {
    sceneId: "scene.runtime-composition",
    reference: {
      kind: "edge",
      edgeId: "edge.request-sink.token.metrics",
    },
  },
  {
    sceneId: "scene.metrics-telemetry",
    reference: {
      kind: "edge",
      edgeId: "edge.runtime.dispatch.metrics",
    },
  },
  {
    sceneId: "scene.metrics-telemetry",
    reference: {
      kind: "edge",
      edgeId: "edge.metrics.to.result",
    },
  },
];

const FLAVOR_TIMELINE_DEFINITIONS: Record<
  ExecutionFlavor,
  readonly TimelineEventDefinition[]
> = {
  native_http: [
    {
      sceneId: "scene.endpoint-bindings-transports",
      reference: {
        kind: "edge",
        edgeId: "edge.dataset.to.endpoint",
      },
    },
  ],
  native_grpc: [
    {
      sceneId: "scene.endpoint-bindings-transports",
      reference: {
        kind: "edge",
        edgeId: "edge.dataset.to.endpoint",
      },
    },
  ],
  online_mock: [
    {
      sceneId: "scene.endpoint-bindings-transports",
      reference: {
        kind: "edge",
        edgeId: "edge.dataset.to.endpoint",
      },
    },
  ],
  dynamo_offline: [
    {
      sceneId: "scene.runtime-composition",
      reference: {
        kind: "edge",
        edgeId: "edge.dynamo.offline.sim-clock.replay",
      },
    },
  ],
  dynamo_online: [
    {
      sceneId: "scene.runtime-composition",
      reference: {
        kind: "edge",
        edgeId: "edge.dynamo.online.replay-mode",
      },
    },
  ],
};

export const DEFAULT_TIMELINE_PLAYBACK: TimelinePlaybackState = {
  isPlaying: false,
  position: 0,
};

function resolveTimelineEvent(
  catalog: ArchitectureCatalog,
  definition: TimelineEventDefinition,
  flavor: ExecutionFlavor | "shared",
): Omit<FlowTimelineEvent, "step"> {
  const scene = catalog.graphScenes.find(({ id }) => id === definition.sceneId);
  if (!scene) {
    throw new Error(`timeline references missing scene ${definition.sceneId}`);
  }

  let channel: FlowChannel;
  let label: string;
  let referenceId: string;
  const reference = definition.reference;
  if (reference.kind === "node") {
    const node = catalog.graphNodes.find(
      ({ id }) => id === reference.nodeId,
    );
    assertNodeInScene(scene.nodeIds, reference.nodeId, node);
    const port = resolveNodePort(catalog, node, reference.portId);
    channel = port.channel;
    label = node.title.developer;
    referenceId = `${node.id}.${port.id}`;
  } else {
    const edge = catalog.graphEdges.find(
      ({ id }) => id === reference.edgeId,
    );
    assertEdgeInScene(scene.edgeIds, reference.edgeId, edge);
    channel = edge.channel;
    label = edge.protocol;
    referenceId = edge.id;
  }

  return {
    id: `${flavor}.${definition.reference.kind}.${referenceId}`,
    channel,
    flavor,
    sceneId: scene.id,
    reference: definition.reference,
    label,
  };
}

function resolveNodePort(
  catalog: ArchitectureCatalog,
  node: GraphNode,
  portId: string,
): GraphNode["seamPorts"][number] {
  const port = node.seamPorts.find(({ id }) => id === portId);
  if (port) {
    return port;
  }
  const owningNode = catalog.graphNodes.find((candidate) =>
    candidate.seamPorts.some(({ id }) => id === portId),
  );
  if (owningNode) {
    throw new Error(
      `timeline port ${portId} belongs to node ${owningNode.id}, not ${node.id}`,
    );
  }
  throw new Error(`timeline references missing port ${portId} on node ${node.id}`);
}

function assertNodeInScene(
  sceneNodeIds: readonly string[],
  nodeId: string,
  node: GraphNode | undefined,
): asserts node is GraphNode {
  if (!node) {
    throw new Error(`timeline references missing node ${nodeId}`);
  }
  if (!sceneNodeIds.includes(nodeId)) {
    throw new Error(`timeline scene does not contain node ${nodeId}`);
  }
}

function assertEdgeInScene(
  sceneEdgeIds: readonly string[],
  edgeId: string,
  edge: GraphEdge | undefined,
): asserts edge is GraphEdge {
  if (!edge) {
    throw new Error(`timeline references missing edge ${edgeId}`);
  }
  if (!sceneEdgeIds.includes(edgeId)) {
    throw new Error(`timeline scene does not contain edge ${edgeId}`);
  }
}

function toTimelineEvents(
  catalog: ArchitectureCatalog,
  definitions: readonly TimelineEventDefinition[],
  flavor: ExecutionFlavor | "shared",
): FlowTimelineEvent[] {
  return definitions.map((definition, step) => ({
    ...resolveTimelineEvent(catalog, definition, flavor),
    step,
  }));
}

export function buildFlowTimeline(
  catalog: ArchitectureCatalog,
  flavor: ExecutionFlavor,
): FlowTimelineEvent[] {
  const prefix = toTimelineEvents(catalog, SHARED_TIMELINE_PREFIX, "shared");
  const branch = toTimelineEvents(
    catalog,
    FLAVOR_TIMELINE_DEFINITIONS[flavor],
    flavor,
  );
  const suffix = toTimelineEvents(catalog, SHARED_TIMELINE_SUFFIX, "shared");
  return [...prefix, ...branch, ...suffix].map((event, step) => ({
    ...event,
    step,
  }));
}

export function clampTimelinePosition(position: number): number {
  if (!Number.isFinite(position)) {
    return 0;
  }
  return Math.max(0, Math.min(1, position));
}

export function playTimeline(
  state: TimelinePlaybackState,
): TimelinePlaybackState {
  return { ...state, isPlaying: true };
}

export function pauseTimeline(
  state: TimelinePlaybackState,
): TimelinePlaybackState {
  return { ...state, isPlaying: false };
}

export function scrubTimeline(
  state: TimelinePlaybackState,
  position: number,
): TimelinePlaybackState {
  return { ...state, isPlaying: false, position: clampTimelinePosition(position) };
}

export function resolveTimelineSemanticState(
  timeline: readonly FlowTimelineEvent[],
  position: number,
): TimelineSemanticState {
  if (timeline.length === 0) {
    throw new Error("timeline requires at least one event");
  }
  const clampedPosition = clampTimelinePosition(position);
  const lastIndex = timeline.length - 1;
  const eventProgress = clampedPosition * lastIndex;
  const nextIndex = Math.floor(eventProgress);
  const eventIndex = Math.max(0, Math.min(lastIndex, nextIndex));
  return {
    position: clampedPosition,
    eventIndex,
    activeEvent: timeline[eventIndex],
    completedEvents: timeline.slice(0, eventIndex + 1),
  };
}

export function resolveTimelineRenderingState(
  semanticState: TimelineSemanticState,
  reducedMotion: boolean,
): TimelineRenderingState {
  return {
    ...semanticState,
    presentation: reducedMotion
      ? { motion: "reduced", animateTransitions: false }
      : { motion: "animated", animateTransitions: true },
  };
}
