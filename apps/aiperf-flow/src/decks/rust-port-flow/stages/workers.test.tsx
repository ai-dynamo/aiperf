/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the Stage 4 "Workers sync & connect" module. They assert the real StageDef contract
//! (order/tone/drill wiring/evidence anchors) AND that the level-1 and level-2 subgraphs render
//! their real code-name labels through the shared PipelineCanvas — not "renders without crashing".

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineCanvas } from "../../../interactive/PipelineCanvas.js";
import { workersStage } from "./workers.js";

describe("workersStage StageDef", () => {
  it("is the stage-4 purple spine slot with id 'workers'", () => {
    expect(workersStage.id).toBe("workers");
    expect(workersStage.order).toBe(4);
    expect(workersStage.tone).toBe("purple");
    expect(workersStage.label).toBe("Workers sync & connect");
  });

  it("drills from the level-1 sub-cell node into the per-thread leaf", () => {
    const leafId = "workers-thread";
    // The subgraph's declared drill child must be an actual leaf key AND an actual level-1 node id,
    // or ZoomStage.drill(clickedNodeId) can never navigate into it.
    expect(workersStage.subgraph?.children).toEqual([leafId]);
    expect(Object.keys(workersStage.leaves ?? {})).toContain(leafId);
    const nodeIds = (workersStage.subgraph?.nodes ?? []).map((n) => n.id);
    expect(nodeIds).toContain(leafId);
  });

  it("pins the verified rust/ source anchors", () => {
    const paths = (workersStage.evidence ?? []).map((e) => e.path);
    expect(paths).toContain("runtime/src/engine/sharded_scheduled.rs:245");
    expect(paths).toContain("runtime/src/engine/sharded_scheduled.rs:358");
    expect(paths).toContain("runtime/src/engine/execute/sharding.rs:25");
    expect(paths).toContain("runtime/src/clock/real_clock.rs:27");
    expect(paths).toContain("runtime/src/engine/turn_execution.rs:214");
  });
});

describe("workersStage level-1 subgraph", () => {
  it("renders the coordinator, shared inputs, sub-cell, and merge by their real names", () => {
    const sub = workersStage.subgraph!;
    render(<PipelineCanvas nodes={sub.nodes} edges={sub.edges} />);
    expect(screen.getByText("run_sharded_scheduled")).toBeInTheDocument();
    expect(screen.getByText("RealClockAnchor")).toBeInTheDocument();
    expect(screen.getByText("GlobalAdmission")).toBeInTheDocument();
    expect(screen.getByText("sub-cell worker × W")).toBeInTheDocument();
    expect(screen.getByText("merge_shards")).toBeInTheDocument();
  });
});

describe("workersStage per-thread leaf subgraph", () => {
  it("renders the !Send per-thread stack by its real names", () => {
    const leaf = workersStage.leaves!["workers-thread"]!;
    render(<PipelineCanvas nodes={leaf.nodes} edges={leaf.edges} />);
    expect(screen.getByText("current_thread runtime")).toBeInTheDocument();
    expect(screen.getByText("LocalSet")).toBeInTheDocument();
    expect(screen.getByText("reactor-local RealClock")).toBeInTheDocument();
    expect(screen.getByText("co-located transport sink")).toBeInTheDocument();
    expect(screen.getByText("execute_scheduled_shard")).toBeInTheDocument();
  });
});
