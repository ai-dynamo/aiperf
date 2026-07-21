/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { PrefixPage } from "./PrefixPage.js";

describe("PrefixPage", () => {
  it("renders the heading and framing copy", () => {
    render(
      <ReactFlowProvider>
        <PrefixPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Prefix chains & LCP-trie lowering")).toBeInTheDocument();
    expect(screen.getByText(/parent: Option<Handle>/)).toBeInTheDocument();
  });

  it("renders the shared-prefix root chain nodes", () => {
    render(
      <ReactFlowProvider>
        <PrefixPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("H0 · system")).toBeInTheDocument();
    expect(screen.getByText("You are helpful.")).toBeInTheDocument();
    expect(screen.getByText("H1 · user")).toBeInTheDocument();
    expect(screen.getByText("What is 2+2?")).toBeInTheDocument();
  });

  it("renders the two branch nodes and their conversation requests", () => {
    render(
      <ReactFlowProvider>
        <PrefixPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("H2 · assistant")).toBeInTheDocument();
    expect(screen.getByText('"4"')).toBeInTheDocument();
    expect(screen.getByText("C1 request")).toBeInTheDocument();
    expect(screen.getByText("H0 · H1 · H2")).toBeInTheDocument();

    expect(screen.getByText("H3 · assistant")).toBeInTheDocument();
    expect(screen.getByText('"It equals four."')).toBeInTheDocument();
    expect(screen.getByText("C2 request")).toBeInTheDocument();
    expect(screen.getByText("H0 · H1 · H3")).toBeInTheDocument();
  });

  it("renders the resolve_content_parents and rebase callouts", () => {
    render(
      <ReactFlowProvider>
        <PrefixPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("resolve_content_parents")).toBeInTheDocument();
    expect(screen.getByText(/longest match yields the/)).toBeInTheDocument();
    expect(screen.getByText("graph/recorded/trie/parents.rs:18")).toBeInTheDocument();

    expect(screen.getByText("rebase on context injection")).toBeInTheDocument();
    expect(screen.getByText(/rebase_conversation_handles/)).toBeInTheDocument();
    expect(screen.getByText("dataset/compose.rs:338")).toBeInTheDocument();
  });
});
