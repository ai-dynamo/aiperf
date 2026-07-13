// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  EdgeWaypointControls,
  appendWaypoint,
  createWaypointPath,
  removeWaypointByIndex,
  type EdgeWaypoint,
} from "./edge-waypoints";

describe("edge waypoints", () => {
  it("builds a polyline path that traverses every waypoint", () => {
    const path = createWaypointPath({
      points: [
        { x: 60, y: 40 },
        { x: 120, y: 30 },
      ],
      source: { x: 0, y: 0 },
      target: { x: 200, y: 0 },
    });

    expect(path).toContain("L 60 40");
    expect(path).toContain("L 120 30");
  });

  it("appends waypoints using visual-only payload shape", () => {
    const result = appendWaypoint({
      edgeId: "edge.runner.transport",
      points: [{ x: 40, y: 10 }],
      source: { x: 0, y: 0 },
      target: { x: 200, y: 0 },
    });

    expect(result).toEqual({
      edgeId: "edge.runner.transport",
      points: [
        { x: 40, y: 10 },
        { x: 100, y: 0 },
      ],
    });
    expect(result).not.toHaveProperty("source");
    expect(result).not.toHaveProperty("target");
  });

  it("removes waypoints by index", () => {
    const points: EdgeWaypoint[] = [
      { x: 40, y: 10 },
      { x: 80, y: 20 },
    ];

    expect(removeWaypointByIndex(points, 0)).toEqual([{ x: 80, y: 20 }]);
  });

  it("supports keyboard controls and reset action", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onReset = vi.fn();
    render(
      <EdgeWaypointControls
        edgeId="edge.runner.transport"
        onChange={onChange}
        onReset={onReset}
        points={[{ x: 70, y: 12 }]}
        source={{ x: 0, y: 0 }}
        target={{ x: 200, y: 0 }}
        visible={true}
      />,
    );

    const handle = screen.getByRole("button", { name: "Move waypoint 1" });
    handle.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenCalledWith({
      edgeId: "edge.runner.transport",
      points: [{ x: 82, y: 12 }],
    });

    await user.keyboard("{Delete}");
    expect(onChange).toHaveBeenLastCalledWith({
      edgeId: "edge.runner.transport",
      points: [],
    });

    await user.click(screen.getByRole("button", { name: "Reset waypoints" }));
    expect(onReset).toHaveBeenCalledWith("edge.runner.transport");
  });

  it("converts pointer coordinates to flow coordinates", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onReset = vi.fn();
    render(
      <EdgeWaypointControls
        edgeId="edge.runner.transport"
        onChange={onChange}
        onReset={onReset}
        points={[{ x: 70, y: 12 }]}
        source={{ x: 0, y: 0 }}
        target={{ x: 200, y: 0 }}
        toFlowPosition={({ x, y }) => ({ x: x - 10, y: y - 20 })}
        visible={true}
      />,
    );
    const handle = screen.getByRole("button", { name: "Move waypoint 1" });

    await user.pointer([
      {
        keys: "[MouseLeft>]",
        target: handle,
        coords: { x: 110, y: 220 },
      },
      {
        target: handle,
        coords: { x: 140, y: 260 },
      },
      {
        keys: "[/MouseLeft]",
        target: handle,
      },
    ]);

    expect(onChange).toHaveBeenCalledWith({
      edgeId: "edge.runner.transport",
      points: [{ x: 130, y: 240 }],
    });
  });
});
