/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { DispatchPage } from "./DispatchPage.js";

describe("DispatchPage", () => {
  it("renders the dispatch precedence heading and framing copy", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Dispatch — one precedence vector, domain-driven")).toBeInTheDocument();
    expect(screen.getByText(/replacing the old five-field precedence/)).toBeInTheDocument();
  });

  it("renders the Turn.body source node", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getAllByText("Turn.body").length).toBeGreaterThan(0);
    expect(screen.getByText("SmallVec<[Handle]>")).toBeInTheDocument();
  });

  it("renders the three fan-out precedence-check nodes", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Raw handle first?")).toBeInTheDocument();
    expect(screen.getByText("→ complete body")).toBeInTheDocument();
    expect(screen.getByText("TokenIds handle?")).toBeInTheDocument();
    expect(screen.getByText("→ token-native")).toBeInTheDocument();
    expect(screen.getByText("Message handles")).toBeInTheDocument();
    expect(screen.getByText("→ format as array")).toBeInTheDocument();
  });

  it("renders the converging BodyPlan and Bytes materializer nodes", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getAllByText("BodyPlan").length).toBeGreaterThan(0);
    expect(screen.getByText("raw · cached · format")).toBeInTheDocument();
    expect(screen.getByText("Bytes")).toBeInTheDocument();
    expect(screen.getByText("→ wire")).toBeInTheDocument();
  });

  it("renders the dispatch_body precedence code callout", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("dispatch_body precedence")).toBeInTheDocument();
    expect(screen.getByText(/pub fn dispatch_body/)).toBeInTheDocument();
  });

  it("renders the two downstream seams callout", () => {
    render(
      <ReactFlowProvider>
        <DispatchPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("The two seams this feeds")).toBeInTheDocument();
    expect(screen.getByText(/RequestSink<R>::dispatch/)).toBeInTheDocument();
    expect(screen.getByText(/Graph HTTP dispatch skips/)).toBeInTheDocument();
  });
});
