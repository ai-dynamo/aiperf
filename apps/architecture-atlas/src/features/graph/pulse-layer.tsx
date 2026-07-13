// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import type { ExecutionFlavor, GraphEdge } from "../../domain/architecture";
import type { FlowTimelineEvent, TimelineSemanticState } from "../../domain/flow-timeline";

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

export function PulseLayer({ reducedMotion, semanticState, visibleEdges }: PulseLayerProps) {
  const completedEdgeIds = useMemo(
    () => resolveCompletedEdgeIds(semanticState.completedEvents),
    [semanticState.completedEvents],
  );
  const activeEdgeId = useMemo(
    () => resolveActiveEdgeId(visibleEdges, semanticState.activeEvent),
    [semanticState.activeEvent, visibleEdges],
  );

  const activeEdge =
    activeEdgeId === null ? null : visibleEdges.find((edge) => edge.id === activeEdgeId) ?? null;

  return (
    <section aria-label="Pulse edge overlay" role="region">
      <p aria-label="Pulse narration" role="status">
        {buildNarration(semanticState)}
      </p>
      <ul aria-label="Pulse edge states">
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
        data-testid="pulse-active-particle"
      />
    </section>
  );
}
