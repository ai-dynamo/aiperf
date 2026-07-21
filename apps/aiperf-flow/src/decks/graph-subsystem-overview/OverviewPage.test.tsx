/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { OverviewPage } from "./OverviewPage.js";

function renderPage(audience: "manager" | "developer" = "manager") {
  return render(
    <ReactFlowProvider>
      <OverviewPage audience={audience} />
    </ReactFlowProvider>,
  );
}

describe("OverviewPage", () => {
  it("renders the title and the one-big-idea callout", () => {
    renderPage();
    expect(screen.getByText("AIPerf Graph Subsystem")).toBeInTheDocument();
    expect(screen.getByText("The one big idea")).toBeInTheDocument();
  });

  it("renders the four-beat concept cards", () => {
    renderPage();
    expect(screen.getByText("Canonicalize")).toBeInTheDocument();
    expect(screen.getByText("Deduplicate")).toBeInTheDocument();
    expect(screen.getByText("Replay as dataflow")).toBeInTheDocument();
    expect(screen.getByText("Rebuild anywhere")).toBeInTheDocument();
  });

  it("shows a stage's WHAT/WHY detail and switches stage on click", () => {
    renderPage();
    // Default stage is Ingest.
    expect(screen.getByText(/one LLM node per recorded request/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Build stores/ }));
    expect(screen.getByText(/content-addressed segments \(blake2b/)).toBeInTheDocument();
  });

  it("hides key symbols in manager view and shows them in developer view", () => {
    const { unmount } = renderPage("manager");
    expect(screen.queryByText("GraphAdapterProtocol")).not.toBeInTheDocument();
    unmount();
    renderPage("developer");
    expect(screen.getByText("GraphAdapterProtocol")).toBeInTheDocument();
  });

  it("auto-detects an adapter by priority and warns on plain yaml", () => {
    renderPage();
    // Default input is a dir of *.jsonl.gz → dynamo_trace wins.
    expect(screen.getByText(/matches the dynamo sniff \(priority 100\)/)).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "yaml" } });
    expect(screen.getByText(/is NOT auto-detected as graph/)).toBeInTheDocument();
  });
});
