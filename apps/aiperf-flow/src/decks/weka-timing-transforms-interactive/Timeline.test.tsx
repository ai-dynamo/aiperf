/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Timeline } from "./Timeline.js";
import { SCENARIOS, derive, idleGaps, lanesOf } from "./logic.js";

describe("Timeline", () => {
  it("labels the raw and warped clock blocks and every node id and lane", () => {
    const reqs = SCENARIOS.agent!.reqs;
    const lanes = lanesOf(reqs);
    const nodes = derive(reqs, 60);
    const gaps = idleGaps(reqs, 60);
    render(<Timeline nodes={nodes} gaps={gaps} lanes={lanes} warpOn />);

    expect(screen.getByText("raw clock")).toBeInTheDocument();
    expect(screen.getByText("warped clock")).toBeInTheDocument();
    for (const lane of lanes) {
      expect(screen.getAllByText(lane).length).toBe(2); // once per block
    }
    for (const n of nodes) {
      expect(screen.getAllByText(n.id).length).toBe(2); // raw bar + warped bar
    }
  });

  it("flags the capped idle gap distinctly from an uncapped one", () => {
    const reqs = SCENARIOS.agent!.reqs;
    const lanes = lanesOf(reqs);
    const nodes = derive(reqs, 60);
    const gaps = idleGaps(reqs, 60);
    render(<Timeline nodes={nodes} gaps={gaps} lanes={lanes} warpOn />);

    expect(screen.getByText("idle 86s > cap")).toBeInTheDocument();
    expect(screen.getByText("idle 2s")).toBeInTheDocument();
  });

  it("labels the warped clock as 'no cap' when warp is off", () => {
    const reqs = SCENARIOS.dense!.reqs;
    const lanes = lanesOf(reqs);
    const nodes = derive(reqs, null);
    render(<Timeline nodes={nodes} gaps={[]} lanes={lanes} warpOn={false} />);
    expect(screen.getByText("warped clock (no cap)")).toBeInTheDocument();
  });
});
