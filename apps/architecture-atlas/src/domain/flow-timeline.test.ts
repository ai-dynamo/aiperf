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

import type { ExecutionFlavor, FlowChannel } from "./architecture";

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
      const first = buildFlowTimeline(flavor);
      const second = buildFlowTimeline(flavor);
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

  it("emits unique flavor-specific branch markers", () => {
    const http = buildFlowTimeline("native_http");
    const grpc = buildFlowTimeline("native_grpc");
    const mock = buildFlowTimeline("online_mock");
    const offline = buildFlowTimeline("dynamo_offline");
    const online = buildFlowTimeline("dynamo_online");

    expect(ids(http)).toContain("branch.native-http.sse-response");
    expect(ids(grpc)).toContain("branch.native-grpc.bidi-stream");
    expect(ids(mock)).toContain("branch.online-mock.synthetic-latency");
    expect(ids(offline)).toContain("branch.dynamo-offline.parity-gate");
    expect(ids(online)).toContain("branch.dynamo-online.replay-online");

    expect(ids(http)).not.toContain("branch.native-grpc.bidi-stream");
    expect(ids(grpc)).not.toContain("branch.online-mock.synthetic-latency");
    expect(ids(mock)).not.toContain("branch.dynamo-offline.parity-gate");
    expect(ids(offline)).not.toContain("branch.dynamo-online.replay-online");
    expect(ids(online)).not.toContain("branch.native-http.sse-response");
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

  it("resolves reduced-motion state from the same semantics", () => {
    const timeline = buildFlowTimeline("native_http");
    const animated = resolveTimelineSemanticState(timeline, 0.4, false);
    const reduced = resolveTimelineSemanticState(timeline, 0.4, true);

    expect(animated.position).toBe(0.4);
    expect(reduced.position).toBe(0.4);
    expect(animated.activeEvent.channel).toBe(reduced.activeEvent.channel);
    expect(animated.completedEvents.length).toBeGreaterThan(0);
    expect(reduced.completedEvents.length).toBeGreaterThan(0);
    expect(
      reduced.completedEvents.every((event) => timeline[event.step]?.id === event.id),
    ).toBe(true);
  });
});
