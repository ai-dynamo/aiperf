/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SpecializedPage } from "./SpecializedPage.js";

describe("SpecializedPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <SpecializedPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Specialized endpoints")).toBeInTheDocument();
    expect(screen.getByText("Embeddings")).toBeInTheDocument();
    expect(screen.getByText("RAG and KServe HTTP")).toBeInTheDocument();
    expect(
      screen.getAllByText("RAG and KServe HTTP aliases remain HTTP routes over shared handlers.")
        .length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("rust/e2e-tests/tests/test_embeddings_endpoint.rs")).toBeInTheDocument();
  });
});
