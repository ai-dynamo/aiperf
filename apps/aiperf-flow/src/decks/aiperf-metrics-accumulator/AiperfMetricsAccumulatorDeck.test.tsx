/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { AiperfMetricsAccumulatorDeck } from "./AiperfMetricsAccumulatorDeck.js";

describe("AiperfMetricsAccumulatorDeck", () => {
  it("renders the title and intro copy", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("AIPerf Metrics Accumulator")).toBeInTheDocument();
    expect(screen.getByText("Conceptual illustration")).toBeInTheDocument();
  });

  it("shows the default pipeline stage (Sweep Engine) and switches on click", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("4 · Sweep Engine")).toBeInTheDocument();
    expect(screen.getByText("SweepLineCurves")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Ingress"));
    expect(screen.getByText("1 · Ingress")).toBeInTheDocument();
    expect(screen.getByText("RequestRecord")).toBeInTheDocument();
  });

  it("renders the ColumnStore table with real column headers and NaN/ragged cells", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("inter_chunk_latency")).toBeInTheDocument();
    expect(screen.getByText("benchmark_phase")).toBeInTheDocument();
    expect(screen.getAllByText("NaN").length).toBeGreaterThan(0);
    expect(screen.getAllByText("→ragged").length).toBe(6);
    expect(screen.getAllByText("WARMUP").length).toBe(1);
  });

  it("selecting a ragged-series record updates the offset readout", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    // Default selection is r2 ([8, 10, 9, 13, 7]).
    expect(screen.getByText(/starts at offset/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "r1" }));
    expect(screen.getByText(/−1 \(absent\)/)).toBeInTheDocument();
  });

  it("switching the sweep curve updates the active-requests stat and legend", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    const label = screen.getByText("active requests");
    expect(label.parentElement).toHaveTextContent("6");

    fireEvent.click(screen.getByRole("button", { name: "Tokens in flight" }));
    expect(screen.getAllByText(/tokens in flight/i).length).toBeGreaterThan(0);
  });

  it("toggling a request off in the sweep view reduces the active-requests count", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    const label = screen.getByText("active requests");
    fireEvent.click(screen.getByRole("button", { name: "F" }));
    expect(label.parentElement).toHaveTextContent("5");
  });

  it("expanding the cumsum collapsible section reveals the running-sum table", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    fireEvent.click(screen.getByText("Events → running sum (concurrency, all requests)"));
    expect(screen.getByText("running sum (concurrency)")).toBeInTheDocument();
  });

  it("shows Effective vs Active section copy", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("Effective")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
    expect(screen.getByText(/sustain over wall clock/)).toBeInTheDocument();
  });

  it("changing slice_duration recomputes slice count and flags an incomplete trailing slice", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    // Default slice_duration=15 over span [0,50] -> 4 slices, last one incomplete.
    const label = screen.getByText("slices");
    expect(label.parentElement).toHaveTextContent("4");
    expect(screen.getByText("Incomplete trailing slice")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "25 ns" }));
    expect(screen.getByText("Evenly divisible")).toBeInTheDocument();
  });

  it("renders the metric taxonomy with real tags and examples", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("RECORD")).toBeInTheDocument();
    expect(screen.getByText("AGGREGATE")).toBeInTheDocument();
    expect(screen.getByText("DERIVED")).toBeInTheDocument();
    expect(screen.getByText(/output_token_throughput/)).toBeInTheDocument();
  });

  it("renders the egress chain", () => {
    render(<AiperfMetricsAccumulatorDeck />);
    expect(screen.getByText("AccumulatorMetricsSummary")).toBeInTheDocument();
    expect(screen.getByText("ExporterManager")).toBeInTheDocument();
    expect(screen.getByText("per-record JSONL")).toBeInTheDocument();
  });
});
