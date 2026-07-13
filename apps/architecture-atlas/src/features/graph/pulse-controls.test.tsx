// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FlowTimelineEvent, TimelineSemanticState } from "../../domain/flow-timeline";
import { PulseControls, type PulseTimerScheduler } from "./pulse-controls";

function buildTimeline(): FlowTimelineEvent[] {
  return [
    {
      channel: "control",
      flavor: "shared",
      id: "shared.edge.runner",
      label: "Runner protocol",
      reference: { edgeId: "edge.runner.protocol", kind: "edge" },
      sceneId: "scene.runner",
      step: 0,
    },
    {
      channel: "request_data",
      flavor: "native_http",
      id: "native_http.edge.dispatch",
      label: "HTTP dispatch",
      reference: { edgeId: "edge.dispatch.http", kind: "edge" },
      sceneId: "scene.dispatch",
      step: 1,
    },
    {
      channel: "token",
      flavor: "shared",
      id: "shared.edge.tokens",
      label: "Token stream",
      reference: { edgeId: "edge.tokens", kind: "edge" },
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

describe("PulseControls", () => {
  it("renders channel legend and active-event narration", () => {
    const timeline = buildTimeline();
    render(
      <PulseControls
        isPlaying={false}
        onPause={vi.fn()}
        onPlay={vi.fn()}
        onRestart={vi.fn()}
        onScrub={vi.fn()}
        reducedMotion={false}
        semanticState={semanticStateAt(timeline, 1)}
        timeline={timeline}
      />,
    );

    expect(screen.getByLabelText("Pulse timeline controls")).toHaveClass("pulse-dock");
    expect(screen.getByRole("region", { name: "Pulse channels legend" })).toHaveTextContent(
      "Control",
    );
    expect(screen.getByRole("region", { name: "Pulse channels legend" })).toHaveTextContent(
      "Request data",
    );
    expect(screen.getByRole("region", { name: "Pulse channels legend" })).toHaveTextContent(
      "Token",
    );
    expect(screen.getByRole("status", { name: "Active pulse narration" })).toHaveTextContent(
      "Step 2 of 3: HTTP dispatch",
    );
    expect(screen.getByRole("status", { name: "Active pulse narration" })).toHaveTextContent(
      "request_data",
    );
  });

  it("invokes play, pause, scrub, and restart callbacks", () => {
    const timeline = buildTimeline();
    const onPlay = vi.fn();
    const onPause = vi.fn();
    const onScrub = vi.fn();
    const onRestart = vi.fn();
    const { rerender } = render(
      <PulseControls
        isPlaying={false}
        onPause={onPause}
        onPlay={onPlay}
        onRestart={onRestart}
        onScrub={onScrub}
        reducedMotion={false}
        semanticState={semanticStateAt(timeline, 0)}
        timeline={timeline}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play pulse timeline" }));
    expect(onPlay).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByRole("slider", { name: "Pulse timeline scrubber" }), {
      target: { value: "0.5" },
    });
    expect(onScrub).toHaveBeenCalledWith(0.5);

    fireEvent.click(screen.getByRole("button", { name: "Restart pulse timeline" }));
    expect(onRestart).toHaveBeenCalledTimes(1);

    rerender(
      <PulseControls
        isPlaying
        onPause={onPause}
        onPlay={onPlay}
        onRestart={onRestart}
        onScrub={onScrub}
        reducedMotion={false}
        semanticState={semanticStateAt(timeline, 1)}
        timeline={timeline}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Pause pulse timeline" }));
    expect(onPause).toHaveBeenCalledTimes(1);
  });

  it("advances deterministically while playing and cleans timer lifecycle", () => {
    const timeline = buildTimeline();
    const onScrub = vi.fn();
    const onPause = vi.fn();
    const scheduler: PulseTimerScheduler = {
      clearInterval: vi.fn(),
      setInterval: vi.fn((_callback: () => void) => 42),
    };

    const { unmount, rerender } = render(
      <PulseControls
        isPlaying
        onPause={onPause}
        onPlay={vi.fn()}
        onRestart={vi.fn()}
        onScrub={onScrub}
        reducedMotion={false}
        scheduler={scheduler}
        semanticState={semanticStateAt(timeline, 1)}
        timeline={timeline}
      />,
    );

    expect(scheduler.setInterval).toHaveBeenCalledTimes(1);
    const firstSetIntervalCall = (
      scheduler.setInterval as unknown as { mock: { calls: Array<[() => void, number]> } }
    ).mock.calls[0];
    const tick = firstSetIntervalCall?.[0];
    if (typeof tick !== "function") {
      throw new Error("expected interval callback");
    }
    tick();
    expect(onScrub).toHaveBeenCalledWith(1);
    expect(onPause).toHaveBeenCalledTimes(1);

    rerender(
      <PulseControls
        isPlaying={false}
        onPause={onPause}
        onPlay={vi.fn()}
        onRestart={vi.fn()}
        onScrub={onScrub}
        reducedMotion={false}
        scheduler={scheduler}
        semanticState={semanticStateAt(timeline, 2)}
        timeline={timeline}
      />,
    );

    expect(scheduler.clearInterval).toHaveBeenCalledWith(42);
    unmount();
    expect(scheduler.clearInterval).toHaveBeenCalledWith(42);
  });
});
