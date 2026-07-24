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
    // One marker per event.
    expect(container.querySelectorAll('[data-testid="event-marker"]').length).toBe(EVENTS.length);
    // One seam frame per SeamFrame.
    expect(container.querySelectorAll('[data-testid="seam-frame"]').length).toBe(SEAMS.length);
    // Exactly one weaving request line.
    expect(container.querySelectorAll('[data-testid="request-line"]').length).toBe(1);
  });

  it("weaves ONE request line through the events in path order", () => {
    const { container } = renderTrack();
    const line = container.querySelector('[data-testid="request-line"]');
    const pts = (line?.getAttribute("points") ?? "").split(" ");
    expect(pts).toHaveLength(3);
  });

  it("switches the axis unit label with the scale", () => {
    const virtual = renderTrack({ scale: "virtual" });
    expect(virtual.getByText("SimClock · virtual ticks")).toBeInTheDocument();
    virtual.unmount();
    const real = renderTrack({ scale: "real" });
    expect(real.getByText("RealClock · wall-ms")).toBeInTheDocument();
  });

  it("places the middle event closer to the start on the real scale than on the virtual scale", () => {
    // On the virtual scale the middle event sits at the midpoint; on the real scale its small wall-ms
    // offset (10 of 200) pulls it far left — the x-mapping actually changes with the scale.
    const virtual = renderTrack({ scale: "virtual" });
    const vx = Number(
      virtual.container.querySelector('[data-testid="request-line"]')?.getAttribute("points")?.split(" ")[1]?.split(",")[0],
    );
    virtual.unmount();
    const real = renderTrack({ scale: "real" });
    const rx = Number(
      real.container.querySelector('[data-testid="request-line"]')?.getAttribute("points")?.split(" ")[1]?.split(",")[0],
    );
    expect(rx).toBeLessThan(vx);
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
