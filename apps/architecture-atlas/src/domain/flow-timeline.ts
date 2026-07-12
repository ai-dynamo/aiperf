// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExecutionFlavor, FlowChannel } from "./architecture";

export interface FlowTimelineEvent {
  id: string;
  step: number;
  channel: FlowChannel;
  flavor: ExecutionFlavor | "shared";
  title: string;
}

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

interface TimelineEventDefinition {
  id: string;
  channel: FlowChannel;
  title: string;
}

const SHARED_TIMELINE_DEFINITIONS: readonly TimelineEventDefinition[] = [
  {
    id: "shared.control.config-v2-projected",
    channel: "control",
    title: "Config v2 request projected",
  },
  {
    id: "shared.control.runner-validated",
    channel: "control",
    title: "Runner validates authored operation",
  },
  {
    id: "shared.request.workload-materialized",
    channel: "request_data",
    title: "Workload and request materialization",
  },
  {
    id: "shared.token.first-token-observed",
    channel: "token",
    title: "First token observed",
  },
  {
    id: "shared.token.stream-complete",
    channel: "token",
    title: "Token stream completion",
  },
  {
    id: "shared.telemetry.metrics-aggregated",
    channel: "telemetry",
    title: "Metrics and telemetry aggregation",
  },
  {
    id: "shared.result.native-report-emitted",
    channel: "report_result",
    title: "Native report emitted",
  },
];

const FLAVOR_TIMELINE_DEFINITIONS: Record<
  ExecutionFlavor,
  readonly TimelineEventDefinition[]
> = {
  native_http: [
    {
      id: "branch.native-http.transport-selected",
      channel: "control",
      title: "HTTP transport selected",
    },
    {
      id: "branch.native-http.sse-response",
      channel: "request_data",
      title: "SSE response stream dispatched",
    },
  ],
  native_grpc: [
    {
      id: "branch.native-grpc.transport-selected",
      channel: "control",
      title: "gRPC transport selected",
    },
    {
      id: "branch.native-grpc.bidi-stream",
      channel: "request_data",
      title: "Bidirectional gRPC stream dispatched",
    },
  ],
  online_mock: [
    {
      id: "branch.online-mock.target-selected",
      channel: "control",
      title: "Online mock target selected",
    },
    {
      id: "branch.online-mock.synthetic-latency",
      channel: "telemetry",
      title: "Synthetic latency and backend telemetry",
    },
  ],
  dynamo_offline: [
    {
      id: "branch.dynamo-offline.sim-clock",
      channel: "control",
      title: "Virtual SimClock drives replay",
    },
    {
      id: "branch.dynamo-offline.parity-gate",
      channel: "report_result",
      title: "Common-summary parity gate applied",
    },
  ],
  dynamo_online: [
    {
      id: "branch.dynamo-online.replay-online",
      channel: "control",
      title: "Replay mode set to online",
    },
    {
      id: "branch.dynamo-online.wall-clock-dispatch",
      channel: "request_data",
      title: "Wall-clock replay dispatch",
    },
  ],
};

export const DEFAULT_TIMELINE_PLAYBACK: TimelinePlaybackState = {
  isPlaying: false,
  position: 0,
};

function toTimelineEvents(
  definitions: readonly TimelineEventDefinition[],
  flavor: ExecutionFlavor | "shared",
): FlowTimelineEvent[] {
  return definitions.map((definition, step) => ({ ...definition, step, flavor }));
}

export function buildFlowTimeline(flavor: ExecutionFlavor): FlowTimelineEvent[] {
  const shared = toTimelineEvents(SHARED_TIMELINE_DEFINITIONS, "shared");
  const branch = toTimelineEvents(FLAVOR_TIMELINE_DEFINITIONS[flavor], flavor);
  return [...shared, ...branch].map((event, step) => ({ ...event, step }));
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
  reducedMotion: boolean,
): TimelineSemanticState {
  if (timeline.length === 0) {
    throw new Error("timeline requires at least one event");
  }
  const clampedPosition = clampTimelinePosition(position);
  const lastIndex = timeline.length - 1;
  const eventProgress = clampedPosition * lastIndex;
  const nextIndex = reducedMotion
    ? Math.round(eventProgress)
    : Math.floor(eventProgress);
  const eventIndex = Math.max(0, Math.min(lastIndex, nextIndex));
  return {
    position: clampedPosition,
    eventIndex,
    activeEvent: timeline[eventIndex],
    completedEvents: timeline.slice(0, eventIndex + 1),
  };
}
