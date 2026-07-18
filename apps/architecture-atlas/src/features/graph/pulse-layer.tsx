// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import type { ExecutionFlavor, GraphEdge } from "../../domain/architecture";
import type { FlowTimelineEvent, TimelineSemanticState } from "../../domain/flow-timeline";
import type { GraphPulseEdges } from "./types";

export interface PulseLayerProps {
  visibleEdges: readonly GraphEdge[];
  semanticState: TimelineSemanticState;
  reducedMotion: boolean;
}

type PulsePhase = "active" | "completed" | "idle";
type FlavorBranch = "shared" | "active-flavor" | "other-flavor";

function resolveActiveEdgeId(
  edges: readonly GraphEdge[],
  event: FlowTimelineEvent,
): string | null {
  const reference = event.reference;
  if (reference.kind === "edge") {
    return edges.some(({ id }) => id === reference.edgeId) ? reference.edgeId : null;
  }
  const sortedEdges = [...edges].sort((left, right) => left.id.localeCompare(right.id));
  const matching = sortedEdges.find((edge) => {
    const touchesReferencedPort =
      (edge.source.nodeId === reference.nodeId &&
        edge.source.portId === reference.portId) ||
      (edge.target.nodeId === reference.nodeId &&
        edge.target.portId === reference.portId);
    if (!touchesReferencedPort) {
      return false;
    }
    if (edge.channel !== event.channel) {
      return false;
    }
    if (event.flavor === "shared") {
      return true;
    }
    return edge.flavors.includes(event.flavor);
  });
  return matching?.id ?? null;
}

function resolveCompletedEdgeIds(events: readonly FlowTimelineEvent[]): ReadonlySet<string> {
  const completedEdgeIds: string[] = [];
  for (const event of events) {
    if (event.reference.kind === "edge") {
      completedEdgeIds.push(event.reference.edgeId);
    }
  }
  return new Set(completedEdgeIds);
}

function resolveFlavorBranch(edge: GraphEdge, activeFlavor: FlowTimelineEvent["flavor"]): FlavorBranch {
  if (activeFlavor === "shared") {
    return "shared";
  }
  return edge.flavors.includes(activeFlavor) ? "active-flavor" : "other-flavor";
}

function resolvePulsePhase(
  edgeId: string,
  activeEdgeId: string | null,
  completedEdgeIds: ReadonlySet<string>,
): PulsePhase {
  if (edgeId === activeEdgeId) {
    return "active";
  }
  if (completedEdgeIds.has(edgeId)) {
    return "completed";
  }
  return "idle";
}

function buildNarration(state: TimelineSemanticState): string {
  const step = state.eventIndex + 1;
  const total = state.completedEvents.length;
  return `Pulse step ${step}: ${state.activeEvent.label} on ${state.activeEvent.channel} (${state.activeEvent.flavor}); ${total} semantic events completed.`;
}

function edgeSupportsFlavor(edge: GraphEdge, flavor: ExecutionFlavor | "shared"): boolean {
  if (flavor === "shared") {
    return true;
  }
  return edge.flavors.includes(flavor);
}

interface DerivePulseEdgeOverlayStateInput {
  visibleEdges: readonly GraphEdge[];
  semanticState: TimelineSemanticState;
  reducedMotion: boolean;
}

export function derivePulseEdgeOverlayState({
  reducedMotion,
  semanticState,
  visibleEdges,
}: DerivePulseEdgeOverlayStateInput): GraphPulseEdges {
  const completedEdgeIdSet = resolveCompletedEdgeIds(semanticState.completedEvents);
  const activeEdgeId = resolveActiveEdgeId(visibleEdges, semanticState.activeEvent);
  const edgeById = new Map(visibleEdges.map((edge) => [edge.id, edge]));
  const completedChannels = [...completedEdgeIdSet]
    .map((edgeId) => edgeById.get(edgeId)?.channel)
    .filter((channel): channel is GraphEdge["channel"] => channel !== undefined);
  return {
    activeChannels: [semanticState.activeEvent.channel],
    activeEdgeIds: activeEdgeId === null ? [] : [activeEdgeId],
    completedChannels: [...new Set(completedChannels)],
    completedEdgeIds: [...completedEdgeIdSet],
    reducedMotion,
  };
}

export function PulseLayer({ reducedMotion, semanticState, visibleEdges }: PulseLayerProps) {
  const pulseEdges = useMemo(
    () =>
      derivePulseEdgeOverlayState({
        reducedMotion,
        semanticState,
        visibleEdges,
      }),
    [reducedMotion, semanticState, visibleEdges],
  );
  const completedEdgeIds = useMemo(
    () => new Set(pulseEdges.completedEdgeIds),
    [pulseEdges.completedEdgeIds],
  );
  const activeEdgeId = pulseEdges.activeEdgeIds[0] ?? null;

  const activeEdge =
    activeEdgeId === null ? null : visibleEdges.find((edge) => edge.id === activeEdgeId) ?? null;

  return (
    <section aria-label="Pulse edge overlay" className="pulse-layer-telemetry" role="region">
      <p aria-label="Pulse narration" className="sr-only" role="status">
        {buildNarration(semanticState)}
      </p>
      <ul aria-hidden="true" aria-label="Pulse edge states" className="pulse-edge-state-inventory">
        {visibleEdges.map((edge) => (
          <li
            data-active-channel={semanticState.activeEvent.channel}
            data-edge-channel={edge.channel}
            data-edge-id={edge.id}
            data-flavor-branch={resolveFlavorBranch(edge, semanticState.activeEvent.flavor)}
            data-flavor-support={edgeSupportsFlavor(edge, semanticState.activeEvent.flavor)}
            data-pulse-phase={resolvePulsePhase(edge.id, activeEdgeId, completedEdgeIds)}
            data-testid={`pulse-edge-${edge.id}`}
            key={edge.id}
          >
            {edge.id}
          </li>
        ))}
      </ul>
      <div
        data-active-channel={semanticState.activeEvent.channel}
        data-active-edge-id={activeEdge?.id ?? ""}
        data-active-flavor={semanticState.activeEvent.flavor}
        data-motion={reducedMotion ? "reduced" : "animated"}
        className="pulse-active-particle-state"
        data-testid="pulse-active-particle"
      />
    </section>
  );
}
