/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "../../test/router.js";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { UpcomingAsyncDataflowDeck } from "./UpcomingAsyncDataflowDeck.js";

function renderDeck(): void {
  render(
    <ReactFlowProvider>
      <UpcomingAsyncDataflowDeck />
    </ReactFlowProvider>,
  );
}

describe("UpcomingAsyncDataflowDeck", () => {
  it("renders the header title, pill, and framing copy", () => {
    renderDeck();
    expect(screen.getByText("Upcoming Async Dataflow — Step/Emit runtime")).toBeInTheDocument();
    expect(screen.getByText("target")).toBeInTheDocument();
    expect(screen.getAllByText(/node-kind-agnostic/).length).toBeGreaterThan(0);
    expect(screen.getByText(/singledispatch over 13 NodeKinds/)).toBeInTheDocument();
  });

  it("renders the header stat tiles", () => {
    renderDeck();
    expect(screen.getByText("Vertex type (Step)")).toBeInTheDocument();
    expect(screen.getByText("Effects")).toBeInTheDocument();
    expect(screen.getByText("Producer resolutions")).toBeInTheDocument();
    expect(screen.getByText("Firing-loop rewrites")).toBeInTheDocument();
  });

  it("renders the firing-lifecycle graph nodes across all lanes", () => {
    renderDeck();
    expect(screen.getByText("Scheduler: entry Steps")).toBeInTheDocument();
    expect(screen.getByText("Await inputs + timing gate")).toBeInTheDocument();
    expect(screen.getByText("Fire Step")).toBeInTheDocument();
    expect(screen.getAllByText("effect: Dispatch").length).toBeGreaterThan(0);
    expect(screen.getAllByText("effect: Emit").length).toBeGreaterThan(0);
    expect(screen.getAllByText("CreditDispatchAdapter").length).toBeGreaterThan(0);
    expect(screen.getByText("CreditIssuer.issue_graph_credit")).toBeInTheDocument();
    expect(screen.getByText("StickyCreditRouter -> worker")).toBeInTheDocument();
    expect(screen.getByText("Graph return observer")).toBeInTheDocument();
    expect(screen.getByText("sleep(Duration.micros)")).toBeInTheDocument();
    expect(screen.getByText("VersionedChannelStore write")).toBeInTheDocument();
    expect(screen.getByText("Producer resolves once")).toBeInTheDocument();
    expect(screen.getByText("Schedule successors")).toBeInTheDocument();
  });

  it("renders the lane legend and the frontier-graph card chrome", () => {
    renderDeck();
    expect(screen.getByText("Dispatch lane (server, credit)")).toBeInTheDocument();
    expect(screen.getByText("Emit lane (replayed latency)")).toBeInTheDocument();
    expect(screen.getByText("Typed resolution")).toBeInTheDocument();
    expect(screen.getByText("Reused firing-loop core")).toBeInTheDocument();
    expect(screen.getByText("frontier loop (back-edge)")).toBeInTheDocument();
    expect(screen.getByText("TraceExecutor frontier · one Step firing")).toBeInTheDocument();
    expect(screen.getByText("scrolls")).toBeInTheDocument();
  });

  it("renders the Dispatch/Emit effect split cards", () => {
    renderDeck();
    expect(screen.getByText("weka + dynamo")).toBeInTheDocument();
    expect(screen.getByText("dynamo only")).toBeInTheDocument();
    expect(screen.getByText(/Builds a/)).toBeInTheDocument();
    expect(screen.getByText("Everything below the leaf is unchanged")).toBeInTheDocument();
  });

  it("renders the typed producer resolution cards and watchdog callouts", () => {
    renderDeck();
    expect(screen.getByText("Produced a value")).toBeInTheDocument();
    expect(screen.getByText("Ran, produced nothing")).toBeInTheDocument();
    expect(screen.getByText("Never will run")).toBeInTheDocument();
    expect(screen.getByText("real")).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByText("WILL_NOT_PRODUCE")).toBeInTheDocument();
    expect(screen.getByText("F-1 · runtime watchdog is the real backstop")).toBeInTheDocument();
    expect(screen.getByText("F-3 · relaxed gates need an escape")).toBeInTheDocument();
  });

  it("renders the reused-vs-changed table rows", () => {
    renderDeck();
    expect(screen.getByText("Frontier firing loop")).toBeInTheDocument();
    expect(screen.getAllByText("VersionedChannelStore").length).toBeGreaterThan(0);
    expect(screen.getByText("Scheduler adjacency")).toBeInTheDocument();
    expect(screen.getByText("Edge-delay / t-star gate")).toBeInTheDocument();
    expect(screen.getByText("Watchdog")).toBeInTheDocument();
    expect(screen.getByText("Dispatch table")).toBeInTheDocument();
    expect(screen.getByText("Worker materialize / manifest")).toBeInTheDocument();
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.getByText("collapsed")).toBeInTheDocument();
    expect(screen.getByText("re-expressed")).toBeInTheDocument();
  });

  it("renders the concurrency/backpressure layers table and closing callout", () => {
    renderDeck();
    expect(screen.getByText("Node tasks")).toBeInTheDocument();
    expect(screen.getByText("Trace lanes")).toBeInTheDocument();
    expect(screen.getByText("Graph credit issue")).toBeInTheDocument();
    expect(screen.getByText("Prefill slots")).toBeInTheDocument();
    expect(screen.getByText("Adapter waiters")).toBeInTheDocument();
    expect(screen.getByText("Replay barrier")).toBeInTheDocument();
    expect(screen.getByText("Router load")).toBeInTheDocument();
    expect(screen.getByText("Not the same thing")).toBeInTheDocument();
  });
});
