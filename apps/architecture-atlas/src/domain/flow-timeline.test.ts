// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  DEFAULT_TIMELINE_PLAYBACK,
  buildFlowTimeline,
  clampTimelinePosition,
  pauseTimeline,
  playTimeline,
  resolveTimelineSemanticState,
  scrubTimeline,
  type FlowTimelineEvent,
} from "./flow-timeline";

import { architectureCatalog } from "../content";
import type {
  ArchitectureCatalog,
  ExecutionFlavor,
  FlowChannel,
} from "./architecture";

const ALL_FLAVORS: readonly ExecutionFlavor[] = [
  "native_http",
  "native_grpc",
  "online_mock",
  "dynamo_offline",
  "dynamo_online",
];

const ALL_CHANNELS: readonly FlowChannel[] = [
  "control",
  "request_data",
  "token",
  "telemetry",
  "report_result",
];

function ids(events: readonly FlowTimelineEvent[]): string[] {
  return events.map(({ id }) => id);
}

describe("flow timeline", () => {
  it("builds deterministic finite events for every flavor", () => {
    for (const flavor of ALL_FLAVORS) {
      const first = buildFlowTimeline(architectureCatalog, flavor);
      const second = buildFlowTimeline(architectureCatalog, flavor);
      const channels = new Set(first.map(({ channel }) => channel));

      expect(first.length).toBeGreaterThan(0);
      expect(first.every(({ step }, index) => step === index)).toBe(true);
      expect(new Set(ids(first)).size).toBe(first.length);
      expect(ids(first)).toEqual(ids(second));
      expect(first).toEqual(second);
      expect(first.every((event) => Number.isInteger(event.step))).toBe(true);
      expect(first.every((event) => event.step >= 0)).toBe(true);
      expect(first.some((event) => event.flavor === flavor)).toBe(true);
      expect(first.some((event) => event.flavor === "shared")).toBe(true);

      for (const channel of ALL_CHANNELS) {
        expect(channels.has(channel)).toBe(true);
      }
    }
  });

  it("derives every event reference and label from the canonical catalog", () => {
    const scenes = new Map(
      architectureCatalog.graphScenes.map((scene) => [scene.id, scene]),
    );
    const nodes = new Map(
      architectureCatalog.graphNodes.map((node) => [node.id, node]),
    );
    const edges = new Map(
      architectureCatalog.graphEdges.map((edge) => [edge.id, edge]),
    );

    for (const flavor of ALL_FLAVORS) {
      for (const event of buildFlowTimeline(architectureCatalog, flavor)) {
        const scene = scenes.get(event.sceneId);
        expect(scene, `${event.id} scene`).toBeDefined();
        if (event.reference.kind === "node") {
          const node = nodes.get(event.reference.nodeId);
          expect(node, `${event.id} node`).toBeDefined();
          expect(scene?.nodeIds).toContain(event.reference.nodeId);
          expect(event.label).toBe(node?.title.developer);
        } else {
          const edge = edges.get(event.reference.edgeId);
          expect(edge, `${event.id} edge`).toBeDefined();
          expect(scene?.edgeIds).toContain(event.reference.edgeId);
          expect(event.channel).toBe(edge?.channel);
          expect(event.label).toBe(edge?.protocol);
        }
      }
    }
  });

  it("fails closed when a timeline reference is absent", () => {
    const catalog: ArchitectureCatalog = {
      ...architectureCatalog,
      graphNodes: architectureCatalog.graphNodes.filter(
        ({ id }) => id !== "node.request-sink-seam",
      ),
    };

    expect(() => buildFlowTimeline(catalog, "native_http")).toThrow(
      /node\.request-sink-seam/,
    );
  });

  it("emits flavor-specific catalog branches", () => {
    const branchReferences = Object.fromEntries(
      ALL_FLAVORS.map((flavor) => [
        flavor,
        buildFlowTimeline(architectureCatalog, flavor)
          .filter((event) => event.flavor === flavor)
          .map((event) =>
            event.reference.kind === "node"
              ? event.reference.nodeId
              : event.reference.edgeId,
          ),
      ]),
    );

    expect(branchReferences.native_http).toContain("edge.dataset.to.endpoint");
    expect(branchReferences.native_grpc).toContain("edge.dataset.to.endpoint");
    expect(branchReferences.online_mock).toContain("edge.dataset.to.endpoint");
    expect(branchReferences.dynamo_offline).toContain(
      "edge.dynamo.offline.sim-clock.replay",
    );
    expect(branchReferences.dynamo_online).toContain(
      "edge.dynamo.online.replay-mode",
    );
  });

  it("clamps and applies pure playback helpers", () => {
    expect(clampTimelinePosition(-0.25)).toBe(0);
    expect(clampTimelinePosition(0.5)).toBe(0.5);
    expect(clampTimelinePosition(2)).toBe(1);
    expect(clampTimelinePosition(Number.NaN)).toBe(0);

    const playing = playTimeline(DEFAULT_TIMELINE_PLAYBACK);
    expect(playing).toEqual({ isPlaying: true, position: 0 });
    expect(DEFAULT_TIMELINE_PLAYBACK).toEqual({ isPlaying: false, position: 0 });

    const paused = pauseTimeline({ isPlaying: true, position: 0.4 });
    expect(paused).toEqual({ isPlaying: false, position: 0.4 });

    const scrubbed = scrubTimeline({ isPlaying: true, position: 0.1 }, 2);
    expect(scrubbed).toEqual({ isPlaying: false, position: 1 });
  });

  it.each([0.01, 0.17, 0.33, 0.49, 0.67, 0.83, 0.99])(
    "keeps semantic selection motion-independent at position %s",
    (position) => {
      const timeline = buildFlowTimeline(architectureCatalog, "native_http");
      const animatedConsumer = resolveTimelineSemanticState(timeline, position);
      const reducedMotionConsumer = resolveTimelineSemanticState(
        timeline,
        position,
      );

      expect(animatedConsumer.position).toBe(position);
      expect(reducedMotionConsumer.eventIndex).toBe(animatedConsumer.eventIndex);
      expect(reducedMotionConsumer.activeEvent).toEqual(
        animatedConsumer.activeEvent,
      );
      expect(reducedMotionConsumer.completedEvents).toEqual(
        animatedConsumer.completedEvents,
      );
    },
  );
});
