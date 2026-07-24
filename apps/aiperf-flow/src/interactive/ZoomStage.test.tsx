/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ZoomStage } from "./ZoomStage.js";
import type { ZoomTree } from "./types.js";

// A 3-level tree so the parametric-depth claim is actually exercised (root -> stage -> leaf).
const TREE: ZoomTree = {
  overview: { label: "Overview", nodes: [], edges: [], children: ["clock", "transport"] },
  clock: { label: "Clock seam", nodes: [], edges: [], children: [] },
  transport: { label: "Transport seam", nodes: [], edges: [], children: ["http"] },
  http: { label: "HTTP sink", nodes: [], edges: [] },
};

function Harness(): React.JSX.Element {
  return (
    <ZoomStage tree={TREE} rootId="overview">
      {(ctx) => (
        <div>
          <p>active: {ctx.node.label}</p>
          <p>level: {ctx.level}</p>
          <p>siblings: {ctx.siblings.join(",")}</p>
          {(ctx.node.children ?? []).map((childId) => (
            <button key={childId} type="button" onClick={() => ctx.drill(childId)}>
              drill {childId}
            </button>
          ))}
        </div>
      )}
    </ZoomStage>
  );
}

describe("ZoomStage", () => {
  it("starts at the root node (level 0)", () => {
    render(<Harness />);
    expect(screen.getByText("active: Overview")).toBeInTheDocument();
    expect(screen.getByText("level: 0")).toBeInTheDocument();
  });

  it("drills into a child, growing the breadcrumb and level", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "drill transport" }));
    expect(screen.getByText("active: Transport seam")).toBeInTheDocument();
    expect(screen.getByText("level: 1")).toBeInTheDocument();
    // Breadcrumb now shows both the root and the active node.
    expect(screen.getByRole("button", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Transport seam" })).toBeInTheDocument();
  });

  it("supports parametric depth: drills a second level into a leaf", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "drill transport" }));
    fireEvent.click(screen.getByRole("button", { name: "drill http" }));
    expect(screen.getByText("active: HTTP sink")).toBeInTheDocument();
    expect(screen.getByText("level: 2")).toBeInTheDocument();
  });

  it("pops a level when a breadcrumb ancestor is clicked", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "drill transport" }));
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(screen.getByText("active: Overview")).toBeInTheDocument();
    expect(screen.getByText("level: 0")).toBeInTheDocument();
  });

  it("pops a level on Escape", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "drill clock" }));
    expect(screen.getByText("active: Clock seam")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByText("active: Overview")).toBeInTheDocument();
  });

  it("moves between siblings with the arrow keys", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "drill clock" }));
    expect(screen.getByText("active: Clock seam")).toBeInTheDocument();
    expect(screen.getByText("siblings: clock,transport")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText("active: Transport seam")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(screen.getByText("active: Clock seam")).toBeInTheDocument();
  });
});
