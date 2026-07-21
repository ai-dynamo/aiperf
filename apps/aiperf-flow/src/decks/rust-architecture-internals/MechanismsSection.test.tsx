/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MechanismsSection } from "./MechanismsSection.js";

describe("MechanismsSection", () => {
  it("renders all seven named mechanisms and the closing callout", () => {
    render(<MechanismsSection detail="engineering" />);
    expect(screen.getByText("The architecture in seven concrete mechanisms")).toBeInTheDocument();
    for (const title of [
      "Self re-exec",
      "Frozen composition",
      "Clock and observers",
      "Worker locality",
      "Prepared selection",
      "Shared prepared reduction",
      "Report persistence",
    ]) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }
    expect(screen.getByText("Startup and worker construction")).toBeInTheDocument();
  });
});
