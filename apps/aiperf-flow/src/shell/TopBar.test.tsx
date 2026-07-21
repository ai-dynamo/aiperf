/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TopBar } from "./TopBar.js";

describe("TopBar", () => {
  it("renders the brand name and section breadcrumb", () => {
    render(<TopBar section="Segment Pools" />);
    expect(screen.getByText("AIPERF")).toBeInTheDocument();
    expect(screen.getByText("Segment Pools")).toBeInTheDocument();
  });

  it("renders optional actions", () => {
    render(<TopBar section="Segment Pools" actions={<button type="button">Export</button>} />);
    expect(screen.getByRole("button", { name: "Export" })).toBeInTheDocument();
  });
});
