/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SlurmExplainedStepByStepDeck } from "./SlurmExplainedStepByStepDeck.js";
import { STEPS, SCENE_LABELS, NARRATION } from "./steps-data.js";

function renderDeck() {
  return render(
    <ReactFlowProvider>
      <SlurmExplainedStepByStepDeck />
    </ReactFlowProvider>,
  );
}

describe("SlurmExplainedStepByStepDeck", () => {
  it("renders the first step's ported prose and scene label", () => {
    renderDeck();
    expect(screen.getByText("You want to load-test a big AI server")).toBeInTheDocument();
    expect(screen.getByText(STEPS[0].eyebrow)).toBeInTheDocument();
    expect(screen.getByText(SCENE_LABELS[0])).toBeInTheDocument();
    expect(screen.getByText(NARRATION[0])).toBeInTheDocument();
    expect(screen.getByText(STEPS[0].caption)).toBeInTheDocument();
  });

  it("renders the first step's diagram node labels", () => {
    renderDeck();
    expect(screen.getByText("Many load generators")).toBeInTheDocument();
    // "Inference server" is both a diagram node and the term title, so it appears more than once.
    expect(screen.getAllByText("Inference server").length).toBeGreaterThan(0);
  });

  it("renders the term definition callout for a step that has one", () => {
    renderDeck();
    expect(screen.getAllByText("Inference server").length).toBeGreaterThanOrEqual(2);
    expect(
      screen.getByText(/The service that runs an AI model and answers requests/),
    ).toBeInTheDocument();
  });

  it("advances to the next step and shows its title, scene label, and diagram", () => {
    renderDeck();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("SLURM hands you a cluster of machines")).toBeInTheDocument();
    expect(screen.getByText(SCENE_LABELS[1])).toBeInTheDocument();
    expect(screen.getByText("SLURM scheduler")).toBeInTheDocument();
  });

  it("jumps directly to the last step via the step rail and shows the commands", () => {
    renderDeck();
    fireEvent.click(screen.getByRole("button", { name: String(STEPS.length) }));
    expect(screen.getByText("The two commands you actually type")).toBeInTheDocument();
    expect(screen.getByText("The commands")).toBeInTheDocument();
    expect(
      screen.getByText(
        "aiperf slurm generate --config benchmark.yaml --cells 4 --output job.sbatch",
      ),
    ).toBeInTheDocument();
    // Also appears as the "Submit" diagram node detail on the final scene.
    expect(screen.getAllByText("sbatch job.sbatch").length).toBeGreaterThanOrEqual(1);
  });
});
