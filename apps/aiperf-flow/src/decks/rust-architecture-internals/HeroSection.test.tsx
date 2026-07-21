/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HeroSection } from "./HeroSection.js";

describe("HeroSection", () => {
  it("renders the title, tagline, and pipeline spine nodes", () => {
    render(<HeroSection detail="engineering" onDetailChange={() => {}} />);
    expect(screen.getByText("Inside Rust AIPerf")).toBeInTheDocument();
    expect(screen.getByText(/A continuous journey through one native run/)).toBeInTheDocument();
    expect(screen.getByText("Config v2")).toBeInTheDocument();
    expect(screen.getByText("--execute")).toBeInTheDocument();
    expect(screen.getByText("Config v2 → aiperf-cli → aiperf-runtime → loadgen-core → artifacts")).toBeInTheDocument();
  });

  it("renders the three framing callouts", () => {
    render(<HeroSection detail="engineering" onDetailChange={() => {}} />);
    expect(screen.getByText("One product binary")).toBeInTheDocument();
    expect(screen.getByText("Small hot-path core")).toBeInTheDocument();
    expect(screen.getByText("Standalone mock target")).toBeInTheDocument();
  });

  it("fires onDetailChange when a detail level is picked", () => {
    const onChange = vi.fn();
    render(<HeroSection detail="engineering" onDetailChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    expect(onChange).toHaveBeenCalledWith("source");
  });

  it("shows source-evidence chips only at the source detail level", () => {
    const { rerender } = render(<HeroSection detail="engineering" onDetailChange={() => {}} />);
    expect(screen.queryByText("source evidence")).not.toBeInTheDocument();
    rerender(<HeroSection detail="source" onDetailChange={() => {}} />);
    expect(screen.getByText("source evidence")).toBeInTheDocument();
    expect(screen.getByText("runtime root")).toBeInTheDocument();
  });
});
