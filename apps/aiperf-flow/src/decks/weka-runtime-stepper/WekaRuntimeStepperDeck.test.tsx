/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { WekaRuntimeStepperDeck } from "./WekaRuntimeStepperDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <WekaRuntimeStepperDeck />
    </ReactFlowProvider>,
  );
}

describe("WekaRuntimeStepperDeck", () => {
  it("renders the header title, pill, and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Async dataflow frontier — interactive stepper")).toBeInTheDocument();
    expect(screen.getByText("weka replay")).toBeInTheDocument();
    expect(
      screen.getByText(/Step through one weka trie trace as the TraceExecutor drives it/),
    ).toBeInTheDocument();
  });

  it("starts on step 1 of 8 with A ready and no channel writes yet", () => {
    renderDeck();
    expect(screen.getByText("step 1 / 8")).toBeInTheDocument();
    expect(screen.getByText(/Scheduler seeds the frontier: A is the entry Step/)).toBeInTheDocument();
    expect(screen.getByText("no writes yet")).toBeInTheDocument();
    expect(screen.getByText("ready · Dispatch")).toBeInTheDocument();
    expect(screen.getByText("0 / 2 arrived")).toBeInTheDocument();
    expect(screen.getByText("Gate is waiting; D stays parked on its asyncio.Event.")).toBeInTheDocument();
  });

  it("renders the trace graph nodes for START, A, B, C, D", () => {
    renderDeck();
    expect(screen.getByText("START")).toBeInTheDocument();
    expect(screen.getByText(/^A\s+\(LlmNode\)$/)).toBeInTheDocument();
    expect(screen.getByText(/^B\s+\(LlmNode\)$/)).toBeInTheDocument();
    expect(screen.getByText(/^C\s+\(LlmNode\)$/)).toBeInTheDocument();
    expect(screen.getByText(/^D\s+\(LlmNode\)$/)).toBeInTheDocument();
  });

  it("advances through Next and shows the channel write and gate progress", async () => {
    renderDeck();

    // Step 1 -> 2: A fires.
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("step 2 / 8")).toBeInTheDocument();
    expect(screen.getByText(/A fires — effect: Dispatch/)).toBeInTheDocument();

    // Step 2 -> 3: A writes A_out (seq 1).
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("step 3 / 8")).toBeInTheDocument();
    expect(screen.getByText("A_out")).toBeInTheDocument();
    expect(screen.getByText("write_seq 1")).toBeInTheDocument();
  });

  it("reaches the satisfied gate and completed trace on the final step via a pill jump", async () => {
    renderDeck();

    fireEvent.click(screen.getByRole("button", { name: "8" }));
    expect(screen.getByText("step 8 / 8")).toBeInTheDocument();
    expect(screen.getByText(/D writes D_out \(seq 4\) and marks done/)).toBeInTheDocument();
    expect(screen.getByText("2 / 2 arrived")).toBeInTheDocument();
    expect(screen.getByText("Gate satisfied — D is released to fire.")).toBeInTheDocument();
    expect(screen.getByText("D_out")).toBeInTheDocument();
    expect(screen.getByText("write_seq 4")).toBeInTheDocument();

    const nextButton = screen.getByRole("button", { name: "Next" });
    expect(nextButton).toBeDisabled();
  });

  it("resets back to step 1 after Reset", async () => {
    renderDeck();

    fireEvent.click(screen.getByRole("button", { name: "4" }));
    expect(screen.getByText("step 4 / 8")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("step 1 / 8")).toBeInTheDocument();
  });

  it("renders the state legend", () => {
    renderDeck();
    expect(screen.getByText("firing")).toBeInTheDocument();
    expect(screen.getAllByText("ready").length).toBeGreaterThan(0);
    expect(screen.getAllByText("pending").length).toBeGreaterThan(0);
  });
});
