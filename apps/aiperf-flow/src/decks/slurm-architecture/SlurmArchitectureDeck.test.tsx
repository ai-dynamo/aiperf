/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SlurmArchitectureDeck } from "./SlurmArchitectureDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <SlurmArchitectureDeck />
    </ReactFlowProvider>,
  );
}

describe("SlurmArchitectureDeck", () => {
  it("renders the title and framing copy", () => {
    renderDeck();
    expect(screen.getByText("One allocation. One controller. Autonomous load cells.")).toBeInTheDocument();
    expect(
      screen.getByText(/Every SLURM task runs the same native command/),
    ).toBeInTheDocument();
    expect(screen.getByText("Source: current Rust implementation + docs/velo.md")).toBeInTheDocument();
    expect(screen.getByText("Audited 2026-07-17")).toBeInTheDocument();
  });

  it("renders the rank ribbon with all four SLURM task rows", () => {
    renderDeck();
    expect(screen.getByText("SLURM TASK SPACE")).toBeInTheDocument();
    expect(screen.getByText("cell_count = SLURM_NTASKS − 1")).toBeInTheDocument();

    expect(screen.getByText("RANK 0")).toBeInTheDocument();
    expect(screen.getByText("Controller")).toBeInTheDocument();
    expect(screen.getByText("reads Config v2")).toBeInTheDocument();

    expect(screen.getByText("RANK 1")).toBeInTheDocument();
    expect(screen.getByText("RANK 2")).toBeInTheDocument();
    expect(screen.getByText("RANK N−1")).toBeInTheDocument();
    expect(screen.getAllByText("Cell 0").length).toBeGreaterThan(0);
    expect(screen.getByText("Cell 1")).toBeInTheDocument();
    expect(screen.getByText("Cell N−2")).toBeInTheDocument();
    expect(screen.getAllByText("cell_id = rank − 1").length).toBeGreaterThan(0);
  });

  it("defaults the inspector to the controller source", () => {
    renderDeck();
    expect(screen.getByText("Selected architecture unit")).toBeInTheDocument();
    expect(screen.getByText("Cellular controller")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Binds Velo, slices the benchmark envelope, waits for registrations, releases START, and merges terminal results.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("rust/runtime/src/engine/cellular_controller.rs")).toBeInTheDocument();
  });

  it("switches the inspector when a diagram node is clicked", () => {
    renderDeck();
    // The "Velo cellular transport" panel is represented by the CELL k / N-1 diagram node.
    within(screen.getByText("CELL k / N−1").closest(".react-flow__node")!)
      .getAllByText("CELL k / N−1")[0]!;
    fireEvent.click(screen.getAllByText("CELL k / N−1")[0]!);
    expect(screen.getByText("Velo cellular transport")).toBeInTheDocument();
    expect(screen.getByText("rust/runtime/src/cellular/transport/velo_transport.rs")).toBeInTheDocument();
  });

  it("switches the inspector when the rank ribbon is clicked", () => {
    renderDeck();
    fireEvent.click(screen.getByText("RANK 1").closest("button")!);
    expect(screen.getByText("Autonomous cells")).toBeInTheDocument();
    expect(screen.getByText("rust/runtime/src/engine/cellular_cell.rs")).toBeInTheDocument();
  });

  it("renders the architecture diagram bands and key nodes", () => {
    renderDeck();
    expect(screen.getByText("ALLOCATION + ROLE DISPATCH")).toBeInTheDocument();
    expect(screen.getByText("CELLULAR CONTROL + EXECUTION")).toBeInTheDocument();
    expect(screen.getByText("TERMINAL RESULTS")).toBeInTheDocument();

    expect(screen.getByText("sbatch / srun")).toBeInTheDocument();
    expect(screen.getByText("N identical tasks")).toBeInTheDocument();
    expect(screen.getByText("SLURM_* topology")).toBeInTheDocument();
    expect(screen.getByText("aiperf slurm run")).toBeInTheDocument();
    expect(screen.getByText("native rank dispatch")).toBeInTheDocument();
    expect(screen.getByText("rank 0 → controller")).toBeInTheDocument();
    expect(screen.getByText("ranks 1…N−1 → cells")).toBeInTheDocument();
    expect(screen.getAllByText("CONTROLLER").length).toBeGreaterThan(0);
    expect(screen.getByText("Inference servers")).toBeInTheDocument();
    expect(screen.getByText("Velo is never on this path")).toBeInTheDocument();
    expect(screen.getByText("Controller merge")).toBeInTheDocument();
    expect(screen.getByText("Authoritative outputs")).toBeInTheDocument();
    expect(screen.getByText("Bulk artifact upload")).toBeInTheDocument();
  });

  it("renders the trace focus pills", () => {
    renderDeck();
    expect(screen.getByText("TRACE")).toBeInTheDocument();
    for (const label of ["End to end", "Rank dispatch", "Control + execution", "Results"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: "Rank dispatch" }));
    expect(screen.getByRole("button", { name: "Rank dispatch" })).toHaveAttribute("aria-pressed", "true");
  });

  it("renders the lifecycle strip", () => {
    renderDeck();
    for (const [title, detail] of [
      ["Resolve", "SLURM_* → role + coordinate"],
      ["Register", "cells connect over Velo"],
      ["Release", "START after readiness barrier"],
      ["Execute", "cells drive HTTP / gRPC load"],
      ["Reduce", "partitions merge into one report"],
    ]) {
      expect(screen.getByText(title!)).toBeInTheDocument();
      expect(screen.getByText(detail!)).toBeInTheDocument();
    }
  });

  it("renders the three traffic planes table", () => {
    renderDeck();
    expect(screen.getByText("Three traffic planes")).toBeInTheDocument();
    expect(screen.getByText("Velo")).toBeInTheDocument();
    expect(screen.getByText("Cell ↔ controller")).toBeInTheDocument();
    expect(
      screen.getByText("register · envelope · START · heartbeat · partition/store"),
    ).toBeInTheDocument();
    expect(screen.getByText("HTTP / gRPC")).toBeInTheDocument();
    expect(screen.getByText("Cell ↔ inference server")).toBeInTheDocument();
    expect(screen.getByText("HTTP/1 + zstd")).toBeInTheDocument();
    expect(screen.getByText("Cell → controller")).toBeInTheDocument();
  });

  it("renders the deployment invariants", () => {
    renderDeck();
    expect(screen.getByText("Deployment invariants")).toBeInTheDocument();
    expect(screen.getByText("SLURM launches; AIPerf coordinates")).toBeInTheDocument();
    expect(
      screen.getByText(
        "SlurmLauncher creates no processes. It expects sibling srun tasks and uses controller timeouts as a backstop.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("One fact bootstraps every cell")).toBeInTheDocument();
    expect(screen.getByText("Cells never parse the authored config")).toBeInTheDocument();
  });

  it("renders the footer identity codes and allocation note", () => {
    renderDeck();
    expect(screen.getByText("controller = rank 0")).toBeInTheDocument();
    expect(screen.getAllByText("cell_id = rank − 1").length).toBeGreaterThan(0);
    expect(screen.getByText("cell_count = ntasks − 1")).toBeInTheDocument();
    expect(
      screen.getByText("Minimum allocation: 2 tasks · default Velo bootstrap port: 9500"),
    ).toBeInTheDocument();
  });
});
