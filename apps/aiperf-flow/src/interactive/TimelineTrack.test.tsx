/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TimelineTrack } from "./TimelineTrack.js";
import type { Lane, SeamFrame, StageRegion, TimelineEvent } from "./timeline.js";

const LANES: Lane[] = [
  { id: "dataset", label: "Dataset" },
  { id: "scheduler", label: "Scheduler" },
  { id: "server", label: "Server" },
];

const EVENTS: TimelineEvent[] = [
  { id: "freeze", label: "freeze", laneId: "dataset", atOrder: 0, realOffsetMs: 0 },
  { id: "issue", label: "issue", laneId: "scheduler", atOrder: 1, realOffsetMs: 10 },
  { id: "ttft", label: "TTFT", laneId: "server", atOrder: 2, realOffsetMs: 200 },
];

const REGIONS: StageRegion[] = [
  { id: "dataset", laneId: "dataset", label: "Dataset loading", startOrder: 0, endOrder: 0 },
  { id: "sched", laneId: "scheduler", label: "Workload", startOrder: 1, endOrder: 1 },
  { id: "hotpath", laneId: "server", label: "Request hot-path", startOrder: 2, endOrder: 2 },
];

const SEAMS: SeamFrame[] = [
  { id: "clock", label: "Clock" },
  { id: "transport", label: "Transport", spanLaneIds: ["server"], spanOrder: [1, 2] },
];

function renderTrack(props?: Partial<React.ComponentProps<typeof TimelineTrack>>): ReturnType<typeof render> {
  return render(
    <TimelineTrack
      lanes={LANES}
      regions={REGIONS}
      events={EVENTS}
      seamFrames={SEAMS}
      requestPath={["freeze", "issue", "ttft"]}
      scale="virtual"
      {...props}
    />,
  );
}

describe("TimelineTrack", () => {
  it("renders every lane label, region label, event marker, and seam frame", () => {
    const { container, getByText } = renderTrack();
    for (const lane of LANES) {
      expect(getByText(lane.label)).toBeInTheDocument();
    }
    expect(getByText("Dataset loading")).toBeInTheDocument();
    expect(getByText("Request hot-path")).toBeInTheDocument();
    // No connecting request line, and no stray event markers when idle (no active play head).
    expect(container.querySelectorAll('[data-testid="request-line"]').length).toBe(0);
    expect(container.querySelectorAll('[data-testid="event-marker"]').length).toBe(0);
    // One seam frame per SeamFrame.
    expect(container.querySelectorAll('[data-testid="seam-frame"]').length).toBe(SEAMS.length);
  });

  it("shows only the active event's marker (the play head), no line", () => {
    const { container } = renderTrack({ activeEventId: "issue" });
    expect(container.querySelectorAll('[data-testid="request-line"]').length).toBe(0);
    const markers = container.querySelectorAll('[data-testid="event-marker"]');
    expect(markers.length).toBe(1);
    expect(markers[0]!.getAttribute("data-active")).toBe("true");
  });

  it("switches the axis unit label with the scale", () => {
    const virtual = renderTrack({ scale: "virtual" });
    expect(virtual.getByText("SimClock · virtual ticks")).toBeInTheDocument();
    virtual.unmount();
    const real = renderTrack({ scale: "real" });
    expect(real.getByText("RealClock · wall-ms")).toBeInTheDocument();
  });

  it("uses even (order-based) x-positions independent of the Clock scale (even-spaced stage flow)", () => {
    // The overview is an even-spaced stage flow: block positions come from event ORDER only, so
    // toggling the Clock scale reformats tick labels but does NOT move the blocks.
    const regionXs = (c: HTMLElement): number[] =>
      Array.from(c.querySelectorAll('[data-testid="stage-region"] rect'))
        .map((r) => Number(r.getAttribute("x")))
        .sort((a, b) => a - b);
    const virtual = renderTrack({ scale: "virtual" });
    const vx = regionXs(virtual.container);
    virtual.unmount();
    const real = renderTrack({ scale: "real" });
    expect(regionXs(real.container)).toEqual(vx);
  });

  it("drills into a stage when its region block is clicked", () => {
    const onRegionClick = vi.fn();
    const { getByRole } = renderTrack({ onRegionClick });
    fireEvent.click(getByRole("button", { name: "Drill into Request hot-path" }));
    expect(onRegionClick).toHaveBeenCalledWith("hotpath");
  });

  it("highlights the region + marker owning the active event", () => {
    const { container } = renderTrack({ activeEventId: "ttft", onRegionClick: () => {} });
    const activeMarkers = container.querySelectorAll('[data-testid="event-marker"][data-active="true"]');
    expect(activeMarkers.length).toBe(1);
    const activeRegion = container.querySelector('[data-testid="stage-region"][data-active="true"]');
    // The active region owns the ttft event (order 2 → the hotpath region).
    expect(activeRegion?.getAttribute("aria-label")).toBe("Drill into Request hot-path");
  });
});
