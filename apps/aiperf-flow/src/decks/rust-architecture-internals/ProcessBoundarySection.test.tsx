/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProcessBoundarySection } from "./ProcessBoundarySection.js";

describe("ProcessBoundarySection", () => {
  it("renders the section heading and both process bands", () => {
    render(<ProcessBoundarySection detail="engineering" />);
    expect(screen.getByText("One run crosses one child process boundary")).toBeInTheDocument();
    expect(screen.getByText("Parent process")).toBeInTheDocument();
    expect(screen.getByText("Fresh execution child")).toBeInTheDocument();
  });

  it("renders the parent and child pipeline nodes", () => {
    render(<ProcessBoundarySection detail="engineering" />);
    expect(screen.getByText("BenchmarkRun")).toBeInTheDocument();
    expect(screen.getByText("stdin to EOF")).toBeInTheDocument();
    expect(screen.getByText("Application")).toBeInTheDocument();
    expect(screen.getByText("typed failure")).toBeInTheDocument();
  });

  it("shows the Before clap callout only above orientation level", () => {
    const { rerender } = render(<ProcessBoundarySection detail="orientation" />);
    expect(screen.queryByText("Before clap")).not.toBeInTheDocument();
    rerender(<ProcessBoundarySection detail="engineering" />);
    expect(screen.getByText("Before clap")).toBeInTheDocument();
  });
});
