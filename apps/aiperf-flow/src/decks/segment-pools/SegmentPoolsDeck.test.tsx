/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SegmentPoolsDeck } from "./SegmentPoolsDeck.js";

describe("SegmentPoolsDeck", () => {
  it("renders the Overview page by default", () => {
    render(<SegmentPoolsDeck />);
    expect(screen.getByText(/Rows in/i)).toBeInTheDocument();
  });

  it("switches to the Pool page when its tab is clicked", () => {
    render(<SegmentPoolsDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Pool" }));
    expect(screen.getByRole("button", { name: /Intern next/i })).toBeInTheDocument();
  });

  it("switches to the Payloads page when its tab is clicked", () => {
    render(<SegmentPoolsDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Payloads" }));
    expect(screen.getAllByRole("button", { name: /Message|TraceHashIds/i }).length).toBeGreaterThan(0);
  });

  it("switches to the BodyPlan page when its tab is clicked", () => {
    render(<SegmentPoolsDeck />);
    fireEvent.click(screen.getByRole("button", { name: "BodyPlan" }));
    expect(screen.getByText(/materialized/i)).toBeInTheDocument();
  });

  it("switches to the Prefix page when its tab is clicked", () => {
    render(<SegmentPoolsDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Prefix" }));
    expect(screen.getAllByText(/prefix/i).length).toBeGreaterThan(0);
  });

  it("switches to the Dispatch page when its tab is clicked", () => {
    render(<SegmentPoolsDeck />);
    fireEvent.click(screen.getByRole("button", { name: "Dispatch" }));
    expect(screen.getAllByText(/Turn\.body/i).length).toBeGreaterThan(0);
  });
});
