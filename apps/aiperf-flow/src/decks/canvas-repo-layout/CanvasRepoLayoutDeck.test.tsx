/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CanvasRepoLayoutDeck } from "./CanvasRepoLayoutDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <CanvasRepoLayoutDeck />
    </ReactFlowProvider>,
  );
}

describe("CanvasRepoLayoutDeck", () => {
  it("renders the title and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Canvas repo layout")).toBeInTheDocument();
    expect(
      screen.getByText(/Committed canvases for the AIPerf Rust workspace/),
    ).toBeInTheDocument();
  });

  it("renders the summary pills", () => {
    renderDeck();
    expect(screen.getByText("7 canvases migrated")).toBeInTheDocument();
    expect(screen.getAllByText("docs/canvases/").length).toBeGreaterThan(0);
    expect(screen.getByText("symlink bridge")).toBeInTheDocument();
  });

  it("renders the why-symlinks callout", () => {
    renderDeck();
    expect(screen.getByText("Why not commit only to the repo path?")).toBeInTheDocument();
    expect(
      screen.getByText(/Cursor detects canvases only when they appear as direct children/),
    ).toBeInTheDocument();
  });

  it("renders the edit -> symlink -> sidecar flow diagram nodes", () => {
    renderDeck();
    expect(screen.getAllByText("Edit in repo").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(/Source files live in docs\/canvases\/\*\.canvas\.tsx/).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("IDE bridge").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Each file is symlinked into/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Local runtime state").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(/\*\.canvas\.data\.json and \*\.canvas\.status\.json/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("git")).toBeInTheDocument();
    expect(screen.getByText("built")).toBeInTheDocument();
    expect(screen.getByText("local")).toBeInTheDocument();
  });

  it("renders the directory map code block", () => {
    renderDeck();
    expect(screen.getByText("Directory map")).toBeInTheDocument();
    expect(screen.getByText(/committed source \(edit here\)/)).toBeInTheDocument();
    expect(screen.getAllByText(/home-anthony-nvidia-projects-aiperf-ajc-rust\/canvases\//).length).toBeGreaterThan(
      0,
    );
    expect(screen.getByText(/local UI state \(not committed\)/)).toBeInTheDocument();
  });

  it("renders the table of all seven committed canvases", () => {
    renderDeck();
    expect(screen.getByText("Committed canvases")).toBeInTheDocument();
    expect(screen.getByText("cellular-algorithm-workbook.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Cellular algorithm workbook")).toBeInTheDocument();
    expect(screen.getByText("cellular-architecture.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Cellular controller / cell topology")).toBeInTheDocument();
    expect(screen.getByText("dynosim-offline-flow.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Dynosim offline replay flow")).toBeInTheDocument();
    expect(screen.getByText("mock-server-architecture.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("aiperf-mock-server surface map")).toBeInTheDocument();
    expect(screen.getByText("rust-aiperf-architecture.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Rust product architecture")).toBeInTheDocument();
    expect(screen.getByText("segment-pools-and-body-plans.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Segment pools and body plans")).toBeInTheDocument();
    expect(screen.getByText("velo-in-aiperf.canvas.tsx")).toBeInTheDocument();
    expect(screen.getByText("Velo transport in cellular mode")).toBeInTheDocument();
  });

  it("renders the adding-a-new-canvas steps", () => {
    renderDeck();
    expect(screen.getByText("Adding a new canvas")).toBeInTheDocument();
    expect(screen.getByText("1. Create source in repo")).toBeInTheDocument();
    expect(
      screen.getByText(/Import only from\s*cursor\/canvas and default-export one component\./),
    ).toBeInTheDocument();
    expect(screen.getByText("2. Bridge to Cursor")).toBeInTheDocument();
    expect(screen.getByText(/ln -s "\$PWD\/docs\/canvases\/my-topic\.canvas\.tsx"/)).toBeInTheDocument();
  });

  it("renders the companion planning docs note", () => {
    renderDeck();
    expect(screen.getByText("Companion planning docs")).toBeInTheDocument();
    expect(
      screen.getByText(/Markdown storyboards for some canvases already live under/),
    ).toBeInTheDocument();
  });
});
