/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExecutionLifecycleSection } from "./ExecutionLifecycleSection.js";

describe("ExecutionLifecycleSection", () => {
  it("renders the heading and coordinator envelope stages", () => {
    render(<ExecutionLifecycleSection detail="engineering" />);
    expect(screen.getByText("Coordinator stages surround a separate phase clock")).toBeInTheDocument();
    expect(screen.getByText("Coordinator envelope stages")).toBeInTheDocument();
    expect(screen.getByText("run validation")).toBeInTheDocument();
  });

  it("defaults to execute and shows the execution clock band", () => {
    render(<ExecutionLifecycleSection detail="engineering" />);
    expect(screen.getByText("OperationV2::Execute")).toBeInTheDocument();
    expect(screen.getByText("Execution clock · only for --execute")).toBeInTheDocument();
    expect(screen.getByText("clock.drive")).toBeInTheDocument();
  });

  it("switches to validate and hides the execution clock band", () => {
    render(<ExecutionLifecycleSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "--validate" }));
    expect(screen.getByText("OperationV2::Validate")).toBeInTheDocument();
    expect(screen.queryByText("Execution clock · only for --execute")).not.toBeInTheDocument();
    expect(screen.getByText("deferred_checks")).toBeInTheDocument();
  });
});
