/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  ClaudeCodeSubagentStepperDeck,
  M2_INPUTS,
  M3_INPUTS,
  TRACE_FRAMES,
  arrivedCount,
  doneCountByKind,
  nodeStateOf,
} from "./ClaudeCodeSubagentStepperDeck.js";

describe("nodeStateOf / arrivedCount / doneCountByKind (pure trace logic)", () => {
  it("treats START as always done", () => {
    expect(nodeStateOf(0, "START")).toBe("done");
  });

  it("marks M1 as firing mid-turn and done once its window closes", () => {
    expect(nodeStateOf(900, "M1")).toBe("firing");
    expect(nodeStateOf(1600, "M1")).toBe("done");
  });

  it("keeps a node pending until every predecessor has completed", () => {
    // S1a depends on M1 (ends at 1600); before that it has no ready inputs yet.
    expect(nodeStateOf(1000, "S1a")).toBe("pending");
    expect(nodeStateOf(1700, "S1a")).toBe("ready");
  });

  it("counts M2's AND-fan-in inputs as they arrive, independent of sub3's slow WebFetch", () => {
    // At t=6050, sub1 (ends 5900) and sub2 (ends 6000) have both resolved.
    expect(arrivedCount(6050, M2_INPUTS)).toBe(2);
    // sub3's WebFetch (S3t) doesn't resolve until 8200, well after M2 fires.
    expect(arrivedCount(6050, M3_INPUTS)).toBe(0);
  });

  it("counts M3's late join once the Edit emit and sub3's late summarize both land", () => {
    // M2t (Edit) ends 10600, S3b (sub3 summarize) ends 10900.
    expect(arrivedCount(10600, M3_INPUTS)).toBe(1);
    expect(arrivedCount(10900, M3_INPUTS)).toBe(2);
  });

  it("counts completed dispatch and emit nodes separately, excluding START", () => {
    // At the final frame (t=15500) every node has completed: 10 dispatch turns
    // (M1,S1a,S1b,S2a,S2b,S3a,S3b,B1a,M2,M3) and 4 emit tool calls (S1t,S2t,S3t,M2t).
    const t = TRACE_FRAMES[TRACE_FRAMES.length - 1].t;
    expect(doneCountByKind(t, "dispatch")).toBe(10);
    expect(doneCountByKind(t, "emit")).toBe(4);
  });
});

describe("ClaudeCodeSubagentStepperDeck", () => {
  it("renders the real title and intro copy from the ported canvas", () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    expect(
      screen.getByText("Claude Code subagent flow — concurrent spawn, staggered join"),
    ).toBeInTheDocument();
    expect(screen.getByText(/sub3 runs a slow WebFetch/)).toBeInTheDocument();
  });

  it("starts at step 1 and shows the entry-step narration", () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    expect(screen.getByText(/step 1 \/ 16/)).toBeInTheDocument();
    expect(
      screen.getByText(/A user task arrives\. The main-agent plan turn \(M1\) is queued/),
    ).toBeInTheDocument();
  });

  it("advances to the next frame's narration when Next is clicked", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText(/step 2 \/ 16/)).toBeInTheDocument();
    expect(screen.getByText(/M1 is in flight/)).toBeInTheDocument();
  });

  it("jumps directly to a clicked step pill", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    fireEvent.click(screen.getByRole("button", { name: "step 6" }));
    expect(screen.getByText(/step 6 \/ 16/)).toBeInTheDocument();
    expect(screen.getByText(/The reason turns returned/)).toBeInTheDocument();
  });

  it("goes back to the previous frame when Prev is clicked", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    fireEvent.click(screen.getByRole("button", { name: "step 3" }));
    fireEvent.click(screen.getByRole("button", { name: "Prev" }));
    expect(screen.getByText(/step 2 \/ 16/)).toBeInTheDocument();
  });

  it("resets back to step 1 when Reset is clicked", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    fireEvent.click(screen.getByRole("button", { name: "step 9" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText(/step 1 \/ 16/)).toBeInTheDocument();
  });

  it("shows the two AND-fan-in gate titles and their await_inputs copy", () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    expect(screen.getByText("M2 · synthesize (early join)")).toBeInTheDocument();
    expect(screen.getByText("await_inputs: sub1_out, sub2_out")).toBeInTheDocument();
    expect(screen.getByText("M3 · report (late join)")).toBeInTheDocument();
    expect(screen.getByText("await_inputs: main_edit_out, sub3_out")).toBeInTheDocument();
  });

  it("shows the gate as parked (0/2) at step 1 and reflects the model name in the request rail", () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    expect(screen.getAllByText("0 / 2 arrived").length).toBeGreaterThan(0);
    expect(screen.getByText("meta-llama/Llama-3.1-70B-Instruct")).toBeInTheDocument();
  });

  it("shows sub3's node label and the concurrent-request request-log entry once fired", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    // Step 4 (t=2000): M1 returned, sub1/sub2/sub3/bg all fired at once.
    fireEvent.click(screen.getByRole("button", { name: "step 4" }));
    expect(screen.getByText("In-flight requests")).toBeInTheDocument();
    expect(screen.getByText("4 live")).toBeInTheDocument();
    expect(screen.getAllByText("bg: scan").length).toBeGreaterThan(0);
  });

  it("shows the M3 gate released copy once both late inputs have arrived", async () => {
    render(<ClaudeCodeSubagentStepperDeck />);
    fireEvent.click(screen.getByRole("button", { name: "step 14" })); // t=11000
    expect(
      screen.getByText(/Edit output \+ sub3's late result both arrived — the report turn fires\./),
    ).toBeInTheDocument();
  });
});
