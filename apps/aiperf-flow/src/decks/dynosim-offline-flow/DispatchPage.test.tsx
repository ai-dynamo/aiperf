/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DispatchPage } from "./DispatchPage.js";

describe("DispatchPage", () => {
  it("renders the three-way dispatch priority nodes and the convergence target", () => {
    render(<DispatchPage level="developer" />);
    expect(screen.getByText("raw_token_ids")).toBeInTheDocument();
    expect(screen.getByText("trace_hash_ids")).toBeInTheDocument();
    expect(screen.getByText("text turn")).toBeInTheDocument();
    expect(screen.getAllByText("dispatch_tokens").length).toBeGreaterThan(0);
    expect(screen.getByText("engine")).toBeInTheDocument();
  });

  it("renders the plain observer strip labels below developer level", () => {
    render(<DispatchPage level="executive" />);
    expect(screen.getByText("arrival")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("renders maintainer observer callback names at maintainer level", () => {
    render(<DispatchPage level="maintainer" />);
    expect(screen.getByText("on_arrival")).toBeInTheDocument();
    expect(screen.getByText("on_terminal")).toBeInTheDocument();
    expect(screen.getByText(/on_arrival from ScheduledRuntime/)).toBeInTheDocument();
  });
});
