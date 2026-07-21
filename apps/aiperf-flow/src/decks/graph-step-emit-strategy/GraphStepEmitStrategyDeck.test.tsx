/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GraphStepEmitStrategyDeck } from "./GraphStepEmitStrategyDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <GraphStepEmitStrategyDeck />
    </ReactFlowProvider>,
  );
}

describe("GraphStepEmitStrategyDeck", () => {
  it("renders the header title, pill, and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Separated Graph Runtime on Step/Emit")).toBeInTheDocument();
    expect(screen.getByText("Strategy spec")).toBeInTheDocument();
    expect(screen.getByText(/North-star strategy for the graph-IR subsystem/)).toBeInTheDocument();
    expect(
      screen.getByText(/consolidates 5\s*\n?\s*prior specs \+ a 2026-06-30 design session/),
    ).toBeInTheDocument();
  });

  it("renders the header stat tiles", () => {
    renderDeck();
    expect(screen.getByText("SOUND")).toBeInTheDocument();
    expect(screen.getByText("0 / 6")).toBeInTheDocument();
    expect(screen.getByText("72")).toBeInTheDocument();
    expect(screen.getByText("Greenfield")).toBeInTheDocument();
  });

  it("renders the governing decision callouts and lane diagram nodes", () => {
    renderDeck();
    expect(screen.getByText("Owner directive (2026-06-30)")).toBeInTheDocument();
    expect(screen.getByText("Accepted tradeoff")).toBeInTheDocument();
    expect(screen.getByText("Graph adapters")).toBeInTheDocument();
    expect(screen.getByText("weka · dynamo · native")).toBeInTheDocument();
    expect(screen.getByText("Graph IR + SegmentPool")).toBeInTheDocument();
    expect(screen.getByText("graph_ir_source")).toBeInTheDocument();
    expect(screen.getByText("GraphIRReplayStrategy")).toBeInTheDocument();
    expect(screen.getByText("Conversation / Turn")).toBeInTheDocument();
    expect(screen.getByText("Linear timing strategies")).toBeInTheDocument();
    expect(screen.getByText("session_manager / worker")).toBeInTheDocument();
  });

  it("renders the Step/Emit IR effect cards and planes", () => {
    renderDeck();
    expect(screen.getByText("effect: Dispatch")).toBeInTheDocument();
    expect(screen.getByText("effect: Emit")).toBeInTheDocument();
    expect(screen.getByText("Plane 1 — dependency / timing")).toBeInTheDocument();
    expect(screen.getByText("Plane 2 — content")).toBeInTheDocument();
  });

  it("renders the adapter lowering table", () => {
    renderDeck();
    expect(screen.getAllByText("weka").length).toBeGreaterThan(0);
    expect(screen.getByText("LlmNode + StaticEdge")).toBeInTheDocument();
    expect(screen.getAllByText("dynamo").length).toBeGreaterThan(0);
    expect(screen.getByText("LlmNode / ReplayNode / SubgraphNode")).toBeInTheDocument();
    expect(screen.getAllByText("native").length).toBeGreaterThan(0);
    expect(screen.getByText(/every NodeKind/)).toBeInTheDocument();
    expect(screen.getByText(/Leverage: weka-interval-order-causality is a down-payment/)).toBeInTheDocument();
  });

  it("renders all eight sequencing steps", () => {
    renderDeck();
    expect(screen.getByText("weka interval-order-causality")).toBeInTheDocument();
    expect(screen.getByText("Phase B — dynamo content emission")).toBeInTheDocument();
    expect(screen.getByText("Define the Step/Emit IR")).toBeInTheDocument();
    expect(screen.getByText("weka → Step/Emit")).toBeInTheDocument();
    expect(screen.getByText("Decouple the graph carrier")).toBeInTheDocument();
    expect(screen.getByText("dynamo → Step/Emit")).toBeInTheDocument();
    expect(screen.getByText("Per-tool-type timing scaling")).toBeInTheDocument();
    expect(screen.getByText("Node-zoo / dead-code collapse")).toBeInTheDocument();
  });

  it("renders the node-zoo collapse cards", () => {
    renderDeck();
    expect(screen.getByText("Correction from prior drafts")).toBeInTheDocument();
    expect(screen.getByText("Deletable (Tier 1)")).toBeInTheDocument();
    expect(screen.getByText("Judgment (Tier 2)")).toBeInTheDocument();
    expect(screen.getByText("Not deletable")).toBeInTheDocument();
  });

  it("renders the adjudicated refinements table", () => {
    renderDeck();
    expect(screen.getByText("F-1")).toBeInTheDocument();
    expect(screen.getByText("F-3")).toBeInTheDocument();
    expect(screen.getByText("M-1")).toBeInTheDocument();
    expect(screen.getByText("M-3")).toBeInTheDocument();
    expect(screen.getByText("M-7")).toBeInTheDocument();
    expect(screen.getByText("F-4")).toBeInTheDocument();
  });

  it("renders the verification & confidence table and closing callout", () => {
    renderDeck();
    expect(screen.getByText("Step/Emit IR sound as a design")).toBeInTheDocument();
    expect(screen.getByText("weka conversion is a mostly-renames reshape")).toBeInTheDocument();
    expect(screen.getByText("One remaining pre-execution action")).toBeInTheDocument();
  });

  it("renders the hard exclusions", () => {
    renderDeck();
    expect(
      screen.getByText(/No merge of graph lane with legacy Conversation\/Turn\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/No branch-on-live-model-output \(validator-enforced\)\./)).toBeInTheDocument();
    expect(screen.getByText(/Multi-root dynamo files remain rejected\./)).toBeInTheDocument();
    expect(screen.getByText(/agentx external_event stays out of scope\./)).toBeInTheDocument();
  });
});
