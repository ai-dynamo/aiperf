/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RequestParticle } from "./RequestParticle.js";
import type { FlowStep } from "./types.js";

const STEP: FlowStep = {
  nodeId: "transport",
  caption: "Dispatcher hands the request to the chosen WorkerSink",
};

describe("RequestParticle", () => {
  it("shows the active node label and the step caption", () => {
    render(<RequestParticle step={STEP} nodeLabel="Transport seam" position={4} total={9} />);
    expect(screen.getByText("Transport seam")).toBeInTheDocument();
    expect(
      screen.getByText("Dispatcher hands the request to the chosen WorkerSink"),
    ).toBeInTheDocument();
    expect(screen.getByText("step 4/9")).toBeInTheDocument();
  });

  it("falls back to the node id when no label is given", () => {
    render(<RequestParticle step={STEP} />);
    expect(screen.getByText("transport")).toBeInTheDocument();
  });

  it("shows an idle prompt when there is no active step", () => {
    render(<RequestParticle step={undefined} />);
    expect(
      screen.getByText("Press Play to send a request through the pipeline."),
    ).toBeInTheDocument();
  });
});
