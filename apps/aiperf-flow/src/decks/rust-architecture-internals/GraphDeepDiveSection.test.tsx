/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GraphDeepDiveSection } from "./GraphDeepDiveSection.js";

describe("GraphDeepDiveSection", () => {
  it("renders the trace formats and compile chain", () => {
    render(<GraphDeepDiveSection detail="engineering" />);
    expect(screen.getByText("A graph is compiled once, then cut differently by phase")).toBeInTheDocument();
    for (const fmt of ["dag_jsonl", "aiperf_trace", "weka_trace", "dynamo_trace"]) {
      expect(screen.getByText(fmt)).toBeInTheDocument();
    }
    expect(screen.getByText("SegmentPool::freeze")).toBeInTheDocument();
    expect(screen.getByText("firing gate max()")).toBeInTheDocument();
  });

  it("defaults to profiling and swaps the transform label by focus", () => {
    render(<GraphDeepDiveSection detail="engineering" />);
    expect(screen.getByText("chop_trie_at_tstar(t*) · drop pre-t* and re-root")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Warmup" }));
    expect(screen.getByText("rewrite_for_warmup(t*) · boundary nodes only")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Handoff" }));
    expect(screen.getByText("chop_trie_at_frontier · drop executed nodes using lane handoff")).toBeInTheDocument();
  });
});
