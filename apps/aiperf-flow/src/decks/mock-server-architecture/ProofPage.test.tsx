/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ProofPage } from "./ProofPage.js";

describe("ProofPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <ProofPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Proof and boundaries")).toBeInTheDocument();
    expect(screen.getByText("Implementation-to-proof graph")).toBeInTheDocument();
    expect(screen.getByText("Unsupported combinations")).toBeInTheDocument();
    expect(
      screen.getAllByText(
        "Raw-record e2e is strongest; integration, unit, then implementation-only evidence follow.",
      ).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("Source and proof index")).toBeInTheDocument();
  });
});
