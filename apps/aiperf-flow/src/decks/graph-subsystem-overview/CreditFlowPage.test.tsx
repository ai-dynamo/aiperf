/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CreditFlowPage } from "./CreditFlowPage.js";

describe("CreditFlowPage", () => {
  it("renders the credit walkthrough at step 1", () => {
    render(<CreditFlowPage />);
    expect(screen.getByText("Step 1 / 8")).toBeInTheDocument();
    expect(screen.getByText(/in graph replay that issuer is the per-instance CreditDispatchAdapter/)).toBeInTheDocument();
  });

  it("advances the walkthrough on Next", () => {
    render(<CreditFlowPage />);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Step 2 / 8")).toBeInTheDocument();
    expect(screen.getByText(/parks an asyncio.Future keyed by/)).toBeInTheDocument();
  });

  it("renders the node-kind and dispatch-registry tables", () => {
    render(<CreditFlowPage />);
    expect(screen.getByText(/Builds a DispatchRequest and awaits the credit adapter/)).toBeInTheDocument();
    expect(screen.getByText("dispatch/barrier.py")).toBeInTheDocument();
  });

  it("switches dispatch outcome and shows its resolution", () => {
    render(<CreditFlowPage />);
    expect(screen.getByText("future.set_result(placeholder)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "overflow" }));
    expect(screen.getByText("future.set_exception(_NodeOverflowTerminate)")).toBeInTheDocument();
  });

  it("renders the backpressure meters", () => {
    render(<CreditFlowPage />);
    expect(screen.getByText("Trace lanes")).toBeInTheDocument();
    expect(screen.getByText("12 / 64 admitted")).toBeInTheDocument();
  });
});
