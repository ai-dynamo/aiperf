/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricsDeepDiveSection } from "./MetricsDeepDiveSection.js";

describe("MetricsDeepDiveSection", () => {
  it("renders the hot observer lane and default exact-fold variant", () => {
    render(<MetricsDeepDiveSection detail="engineering" />);
    expect(screen.getByText("A request becomes a row before it becomes a summary")).toBeInTheDocument();
    expect(screen.getByText("NativeMetricsObserver")).toBeInTheDocument();
    expect(screen.getByText("atomic native-v2.json")).toBeInTheDocument();
    expect(screen.getByText("dispatch ordinal + exact accumulator, clean row dropped")).toBeInTheDocument();
  });

  it("swaps the retention variant subtitle when sketch is picked", () => {
    render(<MetricsDeepDiveSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "Sketch" }));
    expect(screen.getByText("TagSketch t-digest + exact count/sum/extrema, row dropped")).toBeInTheDocument();
  });
});
