/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TStarChop } from "./TStarChop.js";
import { MINI_TRACES, computeEdges, derive, lanesOf } from "./logic.js";

describe("TStarChop", () => {
  it("renders both before/after titles and the S* re-root box by default", () => {
    const reqs = MINI_TRACES[1]!.reqs; // one-sub
    const lanes = lanesOf(reqs);
    const nodes = derive(reqs, 60);
    const edges = computeEdges(nodes);
    render(<TStarChop nodes={nodes} edges={edges} lanes={lanes} tStar={9} />);

    expect(screen.getByText("before")).toBeInTheDocument();
    expect(screen.getByText("after")).toBeInTheDocument();
    expect(screen.getByText("S*")).toBeInTheDocument();
    expect(screen.getByText("t* = 9s")).toBeInTheDocument();
  });

  it("omits the after block and S* box in beforeOnly mode", () => {
    const reqs = MINI_TRACES[0]!.reqs; // linear
    const lanes = lanesOf(reqs);
    const nodes = derive(reqs, 60);
    const edges = computeEdges(nodes);
    render(<TStarChop nodes={nodes} edges={edges} lanes={lanes} tStar={4} beforeOnly />);

    expect(screen.queryByText("after")).not.toBeInTheDocument();
    expect(screen.queryByText("S*")).not.toBeInTheDocument();
    // every node still renders once in the before block
    for (const n of nodes) {
      expect(screen.getByText(n.id)).toBeInTheDocument();
    }
  });
});
