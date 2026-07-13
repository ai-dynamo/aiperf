// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";

import type { FlowChannel } from "../../domain/architecture";
import type { FlowTimelineEvent, TimelineSemanticState } from "../../domain/flow-timeline";

const CHANNEL_LEGEND: readonly FlowChannel[] = [
  "control",
  "request_data",
  "token",
  "telemetry",
  "report_result",
];

const DEFAULT_TICK_MS = 900;

export interface PulseTimerScheduler {
  setInterval(callback: () => void, ms: number): unknown;
  clearInterval(timerId: unknown): void;
}

const browserTimerScheduler: PulseTimerScheduler = {
  clearInterval: (timerId) => window.clearInterval(timerId as number),
  setInterval: (callback, ms) => window.setInterval(callback, ms),
};

export interface PulseControlsProps {
  isPlaying: boolean;
  timeline: readonly FlowTimelineEvent[];
  semanticState: TimelineSemanticState;
  reducedMotion: boolean;
  onPlay(): void;
  onPause(): void;
  onRestart(): void;
  onScrub(position: number): void;
  scheduler?: PulseTimerScheduler;
  tickMs?: number;
}

interface TickOutcome {
  reachedEnd: boolean;
  position: number;
}

function resolveNextTick(
  timeline: readonly FlowTimelineEvent[],
  semanticState: TimelineSemanticState,
): TickOutcome {
  const lastIndex = Math.max(0, timeline.length - 1);
  if (semanticState.eventIndex >= lastIndex) {
    return { reachedEnd: true, position: 1 };
  }
  const nextIndex = semanticState.eventIndex + 1;
  return {
    reachedEnd: nextIndex >= lastIndex,
    position: lastIndex === 0 ? 1 : nextIndex / lastIndex,
  };
}

function buildNarration(
  timeline: readonly FlowTimelineEvent[],
  semanticState: TimelineSemanticState,
): string {
  const step = semanticState.eventIndex + 1;
  return `Step ${step} of ${timeline.length}: ${semanticState.activeEvent.label} on ${semanticState.activeEvent.channel} (${semanticState.activeEvent.flavor})`;
}

export function PulseControls({
  isPlaying,
  onPause,
  onPlay,
  onRestart,
  onScrub,
  reducedMotion,
  scheduler = browserTimerScheduler,
  semanticState,
  tickMs = DEFAULT_TICK_MS,
  timeline,
}: PulseControlsProps) {
  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    const timerId = scheduler.setInterval(() => {
      const next = resolveNextTick(timeline, semanticState);
      onScrub(next.position);
      if (next.reachedEnd) {
        onPause();
      }
    }, tickMs);
    return () => {
      scheduler.clearInterval(timerId);
    };
  }, [isPlaying, onPause, onScrub, scheduler, semanticState, tickMs, timeline]);

  const sliderStep = timeline.length > 1 ? 1 / (timeline.length - 1) : 1;

  return (
    <section aria-label="Pulse timeline controls">
      <div aria-label="Pulse playback actions" role="group">
        <button
          aria-label={isPlaying ? "Pause pulse timeline" : "Play pulse timeline"}
          onClick={isPlaying ? onPause : onPlay}
          type="button"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button aria-label="Restart pulse timeline" onClick={onRestart} type="button">
          Restart
        </button>
      </div>
      <label>
        <span>Timeline scrubber</span>
        <input
          aria-label="Pulse timeline scrubber"
          max={1}
          min={0}
          onChange={(event) => onScrub(Number(event.currentTarget.value))}
          step={sliderStep}
          type="range"
          value={semanticState.position}
        />
      </label>
      <p aria-label="Active pulse narration" role="status">
        {buildNarration(timeline, semanticState)}
      </p>
      <p>{reducedMotion ? "Motion reduced: semantic playback only." : "Motion enabled."}</p>
      <div aria-label="Pulse channels legend" role="region">
        <h2>Channels</h2>
        <ul>
          {CHANNEL_LEGEND.map((channel) => (
            <li key={channel}>{channel}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
