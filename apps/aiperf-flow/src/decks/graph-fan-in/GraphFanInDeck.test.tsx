/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GraphFanInDeck } from "./GraphFanInDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <GraphFanInDeck />
    </ReactFlowProvider>,
  );
}

describe("GraphFanInDeck", () => {
  it("renders the header title, pill, and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Fan-in in the graph dataflow runtime")).toBeInTheDocument();
    expect(screen.getByText("VersionedChannelStore")).toBeInTheDocument();
    expect(screen.getByText(/There is no central join node/)).toBeInTheDocument();
  });

  it("renders the header stat tiles", () => {
    renderDeck();
    expect(screen.getByText("append-log")).toBeInTheDocument();
    expect(screen.getByText("count=N | all")).toBeInTheDocument();
    expect(screen.getByText("(seq, writer_id)")).toBeInTheDocument();
    expect(screen.getByText("orphan")).toBeInTheDocument();
  });

  it("renders the fan-in graph nodes for producers, channel, gate, reduce, and fire", () => {
    renderDeck();
    expect(screen.getByText("Producer A")).toBeInTheDocument();
    expect(screen.getByText("write(messages) -> seq 1")).toBeInTheDocument();
    expect(screen.getByText("Producer B")).toBeInTheDocument();
    expect(screen.getByText("Producer C")).toBeInTheDocument();
    expect(screen.getByText("skipped branch · wrote=False")).toBeInTheDocument();
    expect(screen.getByText("channel: messages")).toBeInTheDocument();
    expect(screen.getByText("reducer=add_messages · declared=3")).toBeInTheDocument();
    expect(screen.getByText("Consumer.await_inputs")).toBeInTheDocument();
    expect(screen.getByText("read + reduce")).toBeInTheDocument();
    expect(screen.getByText("Consumer fires")).toBeInTheDocument();
  });

  it("renders the legend and firing count pill", () => {
    renderDeck();
    expect(screen.getByText("channel log")).toBeInTheDocument();
    expect(screen.getByText("gate")).toBeInTheDocument();
    expect(screen.getByText("skipped producer (wrote=False)")).toBeInTheDocument();
    expect(screen.getByText("count=2 of 3")).toBeInTheDocument();
  });

  it("renders all six gate lifecycle steps", () => {
    renderDeck();
    expect(screen.getByText("Resolve target count")).toBeInTheDocument();
    expect(screen.getByText("Reachability check")).toBeInTheDocument();
    expect(screen.getByText("Register waiter, await event")).toBeInTheDocument();
    expect(screen.getByText("Producer writes wake waiters")).toBeInTheDocument();
    expect(screen.getByText("Capture")).toBeInTheDocument();
    expect(screen.getByText("Read + reduce")).toBeInTheDocument();
  });

  it("renders the gate modes table rows", () => {
    renderDeck();
    expect(screen.getByText("count = N")).toBeInTheDocument();
    expect(screen.getByText('count = "all"')).toBeInTheDocument();
    expect(screen.getByText("streaming close")).toBeInTheDocument();
    expect(screen.getByText("relaxed barrier: any")).toBeInTheDocument();
    expect(screen.getByText("relaxed barrier: quorum")).toBeInTheDocument();
  });

  it("renders the producer resolution table and callouts", () => {
    renderDeck();
    expect(screen.getByText("A real write already landed")).toBeInTheDocument();
    expect(screen.getByText("Ran, no write (skipped conditional branch)")).toBeInTheDocument();
    expect(screen.getByText("Cancelled / failed producer")).toBeInTheDocument();
    expect(screen.getByText("Last producer, 0 arrivals, no init seed")).toBeInTheDocument();
    expect(screen.getByText("insufficient_producers_remaining")).toBeInTheDocument();
    expect(screen.getByText("all_producers_cancelled")).toBeInTheDocument();
  });

  it("renders the determinism cards and Step/Emit callout", () => {
    renderDeck();
    expect(screen.getAllByText("Reduce order").length).toBeGreaterThan(0);
    expect(screen.getByText("Init seed vs. arrival")).toBeInTheDocument();
    expect(screen.getByText("Under Step/Emit (M-1)")).toBeInTheDocument();
  });
});
