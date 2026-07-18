// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { GraphEdge } from "../../domain/architecture";
import type { FlowTimelineEvent, TimelineSemanticState } from "../../domain/flow-timeline";
import { derivePulseEdgeOverlayState, PulseLayer } from "./pulse-layer";

function buildEdge(input: {
  id: string;
  channel: GraphEdge["channel"];
  flavors: GraphEdge["flavors"];
  source?: GraphEdge["source"];
  target?: GraphEdge["target"];
}): GraphEdge {
  return {
    channel: input.channel,
    evidence: [{ path: "AGENTS.md" }],
    flavors: input.flavors,
    footnotes: [],
    id: input.id,
    protocol: `${input.id} protocol`,
    source: input.source ?? { nodeId: "node.a", portId: "port.a" },
    status: { delivery: "unconditional", state: "built" },
    target: input.target ?? { nodeId: "node.b", portId: "port.b" },
  };
}

function buildTimeline(): FlowTimelineEvent[] {
  return [
    {
      channel: "control",
      flavor: "shared",
      id: "shared.edge.runner",
      label: "Runner startup",
      reference: { edgeId: "edge.runner.control", kind: "edge" },
      sceneId: "scene.runner",
      step: 0,
    },
    {
      channel: "request_data",
      flavor: "native_http",
      id: "native_http.node.dispatch",
      label: "HTTP dispatch",
      reference: {
        kind: "node",
        nodeId: "node.dispatch",
        portId: "port.dispatch.out",
      },
      sceneId: "scene.dispatch",
      step: 1,
    },
    {
      channel: "token",
      flavor: "shared",
      id: "shared.edge.tokens",
      label: "Token streaming",
      reference: { edgeId: "edge.tokens.metrics", kind: "edge" },
      sceneId: "scene.metrics",
      step: 2,
    },
  ];
}

function semanticStateAt(
  timeline: readonly FlowTimelineEvent[],
  eventIndex: number,
): TimelineSemanticState {
  const activeEvent = timeline[eventIndex];
  return {
    activeEvent,
    completedEvents: timeline.slice(0, eventIndex + 1),
    eventIndex,
    position: eventIndex / Math.max(1, timeline.length - 1),
  };
}

describe("PulseLayer", () => {
  it("annotates active and completed edge identities from semantic timeline state", () => {
    const timeline = buildTimeline();
    const edges = [
      buildEdge({
        channel: "control",
        flavors: ["native_http", "native_grpc"],
        id: "edge.runner.control",
      }),
      buildEdge({
        channel: "request_data",
        flavors: ["native_http"],
        id: "edge.dispatch.http",
        source: {
          nodeId: "node.dispatch",
          portId: "port.dispatch.out",
        },
      }),
      buildEdge({
        channel: "token",
        flavors: ["native_http"],
        id: "edge.tokens.metrics",
      }),
    ];

    render(
      <PulseLayer
        reducedMotion={false}
        semanticState={semanticStateAt(timeline, 1)}
        visibleEdges={edges}
      />,
    );

    const overlay = screen.getByRole("region", { name: "Pulse edge overlay" });
    expect(
      overlay.querySelector('ul[aria-label="Pulse edge states"]'),
    ).toHaveAttribute("aria-hidden", "true");
    expect(within(overlay).getByTestId("pulse-edge-edge.runner.control")).toHaveAttribute(
      "data-pulse-phase",
      "completed",
    );
    expect(within(overlay).getByTestId("pulse-edge-edge.dispatch.http")).toHaveAttribute(
      "data-pulse-phase",
      "active",
    );
    expect(within(overlay).getByTestId("pulse-edge-edge.tokens.metrics")).toHaveAttribute(
      "data-pulse-phase",
      "idle",
    );

    const particle = screen.getByTestId("pulse-active-particle");
    expect(particle).toHaveAttribute("data-active-edge-id", "edge.dispatch.http");
    expect(particle).toHaveAttribute("data-active-channel", "request_data");
    expect(particle).toHaveAttribute("data-active-flavor", "native_http");
    expect(particle).toHaveAttribute("data-motion", "animated");
  });

  it("anchors node events to the referenced port instead of an unrelated channel match", () => {
    const timeline = buildTimeline();
    const edges = [
      buildEdge({
        channel: "request_data",
        flavors: ["native_http"],
        id: "edge.a-unrelated",
      }),
      buildEdge({
        channel: "request_data",
        flavors: ["native_http"],
        id: "edge.dispatch.http",
        target: {
          nodeId: "node.dispatch",
          portId: "port.dispatch.out",
        },
      }),
    ];

    const overlay = derivePulseEdgeOverlayState({
      reducedMotion: false,
      semanticState: semanticStateAt(timeline, 1),
      visibleEdges: edges,
    });

    expect(overlay.activeEdgeIds).toEqual(["edge.dispatch.http"]);
  });

  it("keeps semantic identities and narration under reduced motion while disabling movement", () => {
    const timeline = buildTimeline();
    const edges = [
      buildEdge({
        channel: "control",
        flavors: ["native_http", "native_grpc"],
        id: "edge.runner.control",
      }),
      buildEdge({
        channel: "request_data",
        flavors: ["native_http"],
        id: "edge.dispatch.http",
      }),
      buildEdge({
        channel: "token",
        flavors: ["native_http"],
        id: "edge.tokens.metrics",
      }),
    ];
    const semanticState = semanticStateAt(timeline, 2);
    const { rerender } = render(
      <PulseLayer reducedMotion={false} semanticState={semanticState} visibleEdges={edges} />,
    );

    const animatedNarration = screen.getByRole("status", { name: "Pulse narration" }).textContent;
    const animatedPhases = edges.map((edge) =>
      screen.getByTestId(`pulse-edge-${edge.id}`).getAttribute("data-pulse-phase"),
    );

    rerender(<PulseLayer reducedMotion semanticState={semanticState} visibleEdges={edges} />);

    const reducedNarration = screen.getByRole("status", { name: "Pulse narration" }).textContent;
    const reducedPhases = edges.map((edge) =>
      screen.getByTestId(`pulse-edge-${edge.id}`).getAttribute("data-pulse-phase"),
    );

    expect(reducedNarration).toBe(animatedNarration);
    expect(reducedPhases).toEqual(animatedPhases);
    expect(screen.getByTestId("pulse-active-particle")).toHaveAttribute("data-motion", "reduced");
  });

  it("derives typed pulse edge ids and channels for graph-edge rendering", () => {
    const timeline = buildTimeline();
    const edges = [
      buildEdge({
        channel: "control",
        flavors: ["native_http", "native_grpc"],
        id: "edge.runner.control",
      }),
      buildEdge({
        channel: "request_data",
        flavors: ["native_http"],
        id: "edge.dispatch.http",
      }),
      buildEdge({
        channel: "token",
        flavors: ["native_http"],
        id: "edge.tokens.metrics",
      }),
    ];

    const overlay = derivePulseEdgeOverlayState({
      reducedMotion: true,
      semanticState: semanticStateAt(timeline, 2),
      visibleEdges: edges,
    });

    expect(overlay.activeEdgeIds).toEqual(["edge.tokens.metrics"]);
    expect(overlay.activeChannels).toEqual(["token"]);
    expect(overlay.completedEdgeIds).toEqual(["edge.runner.control", "edge.tokens.metrics"]);
    expect(overlay.completedChannels).toEqual(["control", "token"]);
    expect(overlay.reducedMotion).toBe(true);
  });
});
