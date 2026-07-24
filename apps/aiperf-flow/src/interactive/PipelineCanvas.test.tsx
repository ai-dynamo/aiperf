/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PipelineCanvas } from "./PipelineCanvas.js";

const NODES: Node[] = [
  { id: "alpha", type: "panel", position: { x: 0, y: 0 }, data: { title: "Alpha stage" } },
  { id: "beta", type: "panel", position: { x: 240, y: 0 }, data: { title: "Beta stage" } },
];
const EDGES: Edge[] = [{ id: "e-alpha-beta", source: "alpha", target: "beta", type: "flow" }];

describe("PipelineCanvas", () => {
  it("renders every node's label", () => {
    render(<PipelineCanvas nodes={NODES} edges={EDGES} />);
    expect(screen.getByText("Alpha stage")).toBeInTheDocument();
    expect(screen.getByText("Beta stage")).toBeInTheDocument();
  });

  it("calls onNodeClick with the clicked node's id", () => {
    const onNodeClick = vi.fn();
    render(<PipelineCanvas nodes={NODES} edges={EDGES} onNodeClick={onNodeClick} />);
    fireEvent.click(screen.getByText("Beta stage"));
    expect(onNodeClick).toHaveBeenCalledWith("beta");
  });

  it("does not throw when no onNodeClick is supplied and a node is clicked", () => {
    render(<PipelineCanvas nodes={NODES} edges={EDGES} />);
    expect(() => fireEvent.click(screen.getByText("Alpha stage"))).not.toThrow();
  });

  it("renders every node in ELK layout mode", () => {
    render(<PipelineCanvas nodes={NODES} edges={EDGES} layout={{ direction: "RIGHT" }} />);
    expect(screen.getByText("Alpha stage")).toBeInTheDocument();
    expect(screen.getByText("Beta stage")).toBeInTheDocument();
  });
});
