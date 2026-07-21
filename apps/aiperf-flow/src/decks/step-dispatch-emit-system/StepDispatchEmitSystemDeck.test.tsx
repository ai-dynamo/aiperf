/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { StepDispatchEmitSystemDeck } from "./StepDispatchEmitSystemDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <StepDispatchEmitSystemDeck />
    </ReactFlowProvider>,
  );
}

describe("StepDispatchEmitSystemDeck", () => {
  it("renders the header title, pill, and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Step / Dispatch / Emit IR")).toBeInTheDocument();
    expect(screen.getByText("graph-lane-internal")).toBeInTheDocument();
    expect(screen.getByText(/The greenfield two-effect workload IR/)).toBeInTheDocument();
  });

  it("renders the header stat tiles", () => {
    renderDeck();
    expect(screen.getByText("vertex kind (Step)")).toBeInTheDocument();
    expect(screen.getByText("2 effects (Dispatch / Emit)")).toBeInTheDocument();
    expect(screen.getByText("planes (timing / content)")).toBeInTheDocument();
    expect(screen.getByText("AND")).toBeInTheDocument();
    expect(screen.getByText("fan-in gate semantics")).toBeInTheDocument();
  });

  it("renders the structural invariant callout", () => {
    renderDeck();
    expect(screen.getByText("Structural invariant (spec §0)")).toBeInTheDocument();
    expect(screen.getByText(/Control flow never branches on live model output/)).toBeInTheDocument();
  });

  it("renders the Step field table and the Dispatch/Emit effect cards", () => {
    renderDeck();
    expect(screen.getByText("One vertex, two effects")).toBeInTheDocument();
    expect(screen.getAllByText("effect").length).toBeGreaterThan(0);
    expect(screen.getByText("Dispatch | Emit")).toBeInTheDocument();
    expect(screen.getByText(/effect = "dispatch"/)).toBeInTheDocument();
    expect(screen.getByText(/effect = "emit"/)).toBeInTheDocument();
    expect(
      screen.getByText("Server-hitting: live timing, consumes credit. Service time is measured, never a field."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Canned/replayed: authored/recorded latency, no network, no credit."),
    ).toBeInTheDocument();
    expect(screen.getByText("response_channel")).toBeInTheDocument();
    expect(screen.getByText("duration")).toBeInTheDocument();
  });

  it("renders the object model diagram nodes", () => {
    renderDeck();
    expect(screen.getAllByText("Workload").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Graph").length).toBeGreaterThan(0);
    expect(screen.getByText("Trace[]")).toBeInTheDocument();
    expect(screen.getByText("Step{}")).toBeInTheDocument();
    expect(screen.getByText("Edge[]")).toBeInTheDocument();
    expect(screen.getByText("Channel{}")).toBeInTheDocument();
    expect(screen.getAllByText("Dispatch").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Emit").length).toBeGreaterThan(0);
  });

  it("renders the Edge and Trace field tables", () => {
    renderDeck();
    expect(screen.getByText("source")).toBeInTheDocument();
    expect(screen.getByText("producing Step id")).toBeInTheDocument();
    expect(screen.getByText("arrival_time")).toBeInTheDocument();
    expect(screen.getByText('stable trace id (e.g. "t-1#0")')).toBeInTheDocument();
  });

  it("renders the two-planes cards", () => {
    renderDeck();
    expect(screen.getByText("Plane 1 — dependency / timing")).toBeInTheDocument();
    expect(screen.getByText("Plane 2 — content / cache")).toBeInTheDocument();
    expect(screen.getByText("serialized IR")).toBeInTheDocument();
    expect(screen.getByText("runtime companion")).toBeInTheDocument();
  });

  it("renders the node-kind lowering table rows", () => {
    renderDeck();
    expect(screen.getByText("Node-kind lowering")).toBeInTheDocument();
    expect(screen.getByText("llm")).toBeInTheDocument();
    expect(screen.getByText("replay")).toBeInTheDocument();
    expect(screen.getByText("loop")).toBeInTheDocument();
    expect(screen.getByText("pre-unrolled; not a Step/Emit type at runtime")).toBeInTheDocument();
  });

  it("renders the projection pipeline diagram and byte-parity callout", () => {
    renderDeck();
    expect(screen.getByText("Projection pipeline")).toBeInTheDocument();
    expect(screen.getAllByText("ParsedGraph").length).toBeGreaterThan(0);
    expect(screen.getAllByText("weka_trie_to_workload").length).toBeGreaterThan(0);
    expect(screen.getAllByText("unified store").length).toBeGreaterThan(0);
    expect(screen.getByText("Byte-parity contract")).toBeInTheDocument();
  });

  it("renders all nine runtime firing steps", () => {
    renderDeck();
    expect(screen.getByText("Runtime firing path")).toBeInTheDocument();
    expect(screen.getByText("Gate")).toBeInTheDocument();
    expect(screen.getByText("Snapshot")).toBeInTheDocument();
    expect(screen.getByText("Timing")).toBeInTheDocument();
    expect(screen.getByText("Span")).toBeInTheDocument();
    expect(screen.getByText("Execute")).toBeInTheDocument();
    expect(screen.getByText("Write")).toBeInTheDocument();
    expect(screen.getByText("Producers")).toBeInTheDocument();
    expect(screen.getByText("Successors")).toBeInTheDocument();
    expect(screen.getByText("Orphan")).toBeInTheDocument();
    expect(screen.getByText("Dispatch execution")).toBeInTheDocument();
    expect(screen.getByText("Emit execution")).toBeInTheDocument();
  });

  it("renders the structural validation cards and deferred-by-design callout", () => {
    renderDeck();
    expect(screen.getByText("Structural validation")).toBeInTheDocument();
    expect(screen.getByText("Effect coherence")).toBeInTheDocument();
    expect(screen.getByText("Reference integrity")).toBeInTheDocument();
    expect(screen.getByText("Acyclicity")).toBeInTheDocument();
    expect(screen.getByText("Deferred by design (not gaps)")).toBeInTheDocument();
  });

  it("renders the source files section with file labels", () => {
    renderDeck();
    expect(screen.getByText("Source files")).toBeInTheDocument();
    expect(screen.getByText("step_emit.py")).toBeInTheDocument();
    expect(screen.getByText("step_emit_weka.py")).toBeInTheDocument();
    expect(screen.getByText("step_emit_validate.py")).toBeInTheDocument();
    expect(screen.getByText("graph/executor.py")).toBeInTheDocument();
  });
});
