/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GraphSubsystemOverviewDeck } from "./GraphSubsystemOverviewDeck.js";

describe("GraphSubsystemOverviewDeck", () => {
  it("renders the Overview page by default with the glossary", () => {
    render(<GraphSubsystemOverviewDeck />);
    expect(screen.getByText("AIPerf Graph Subsystem")).toBeInTheDocument();
    expect(screen.getByText("Glossary")).toBeInTheDocument();
    expect(screen.getByText("ParsedGraph")).toBeInTheDocument();
  });

  it("switches to the Credit Flow page", () => {
    render(<GraphSubsystemOverviewDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Credit Flow" }));
    expect(screen.getByText("Step 1 / 8")).toBeInTheDocument();
  });

  it("switches to the Deduplication page", () => {
    render(<GraphSubsystemOverviewDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Deduplication" }));
    expect(screen.getByText("Segment-trie dedup")).toBeInTheDocument();
  });

  it("switches to the Scheduling page", () => {
    render(<GraphSubsystemOverviewDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Scheduling" }));
    expect(screen.getByText("t* snapshot chop")).toBeInTheDocument();
  });

  it("switches to the Execution page", () => {
    render(<GraphSubsystemOverviewDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Execution" }));
    expect(screen.getByText("Tick 0 / 5")).toBeInTheDocument();
  });

  it("toggles the developer audience, revealing key symbols on Overview", () => {
    render(<GraphSubsystemOverviewDeck />);
    expect(screen.queryByText("GraphAdapterProtocol")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "developer" }));
    expect(screen.getByText("GraphAdapterProtocol")).toBeInTheDocument();
  });
});
