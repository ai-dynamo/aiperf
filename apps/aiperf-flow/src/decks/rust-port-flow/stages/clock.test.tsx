/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineCanvas } from "../../../interactive/index.js";
import { clockStage, clockFlowSteps } from "./clock.js";

describe("clockStage", () => {
  it("keeps the spine identity the deck shell wires against", () => {
    expect(clockStage.id).toBe("clock");
    expect(clockStage.order).toBe(5);
    expect(clockStage.label).toBe("Clock seam");
    expect(clockStage.tone).toBe("orange");
  });

  it("defines the RealClock-vs-SimClock level-1 seam subgraph", () => {
    const sub = clockStage.subgraph;
    expect(sub).toBeDefined();
    const ids = sub!.nodes.map((n) => n.id);
    expect(ids).toContain("clock__trait");
    expect(ids).toContain("clockReal");
    expect(ids).toContain("clockSim");
    // Both backends fan in to a single transport-timing consumer node.
    expect(ids).toContain("clock__transport");
    // The two backend nodes are the drillable children (their ids key the leaves).
    expect(sub!.children).toEqual(["clockReal", "clockSim"]);
    // Every edge connects two declared nodes.
    for (const edge of sub!.edges) {
      expect(ids).toContain(edge.source);
      expect(ids).toContain(edge.target);
    }
  });

  it("provides RealClock and SimClock level-2 leaves keyed by their drillable node ids", () => {
    expect(clockStage.leaves).toBeDefined();
    expect(Object.keys(clockStage.leaves!).sort()).toEqual(["clockReal", "clockSim"]);
    const real = clockStage.leaves!.clockReal!;
    const realTitles = real.nodes.map((n) => (n.data as { title?: string }).title);
    expect(realTitles).toContain("RealClockAnchor");
    expect(realTitles).toContain("timerfd_sleep_ns");
    const sim = clockStage.leaves!.clockSim!;
    const simTitles = sim.nodes.map((n) => (n.data as { title?: string }).title);
    expect(simTitles).toContain("advance_to(ns)");
    expect(simTitles.some((t) => t?.includes("at_ns, seq_no"))).toBe(true);
  });

  it("pins verified, correctly-formatted source anchors", () => {
    const paths = (clockStage.evidence ?? []).map((e) => e.path);
    expect(paths).toContain("runtime/src/clock/clock.rs:12");
    expect(paths).toContain("runtime/src/clock/real_clock.rs:52");
    expect(paths).toContain("runtime/src/clock/sim_clock.rs:48");
    // Every anchor is a real file:line reference.
    for (const p of paths) {
      expect(p).toMatch(/^[\w./-]+\.rs:\d+$/);
    }
  });

  it("supplies FlowStep captions that traverse the seam's own nodes", () => {
    expect(clockFlowSteps.length).toBeGreaterThanOrEqual(3);
    const stepIds = clockFlowSteps.map((s) => s.nodeId);
    const subIds = clockStage.subgraph!.nodes.map((n) => n.id);
    for (const id of stepIds) {
      expect(subIds).toContain(id);
    }
    expect(clockFlowSteps.some((s) => /SimClock/.test(s.caption))).toBe(true);
    expect(clockFlowSteps.some((s) => /is_virtual/.test(s.caption))).toBe(true);
  });

  it("renders the seam node titles on a real React Flow canvas", () => {
    const sub = clockStage.subgraph!;
    render(<PipelineCanvas nodes={sub.nodes} edges={sub.edges} height={320} />);
    expect(screen.getByText("trait Clock")).toBeInTheDocument();
    expect(screen.getByText("RealClock")).toBeInTheDocument();
    expect(screen.getByText("SimClock")).toBeInTheDocument();
  });
});
