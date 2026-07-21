/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DatasetPage } from "./DatasetPage.js";

describe("DatasetPage", () => {
  it("renders the legend and starts with two chunks published", () => {
    render(<DatasetPage />);
    expect(
      screen.getByText("H = history before attach · R = reply replay · L = live push · own/pass = modulo decision"),
    ).toBeInTheDocument();
    expect(screen.getByText("published 2/6")).toBeInTheDocument();
    expect(screen.getByText("zpack chunk 0")).toBeInTheDocument();
  });

  it("opens the floodgate to publish more chunks", () => {
    render(<DatasetPage />);
    fireEvent.click(screen.getByRole("button", { name: "Open floodgate" }));
    expect(screen.getByText("published 3/6")).toBeInTheDocument();
  });

  it("attaches a cell channel and records the boundary", () => {
    render(<DatasetPage />);
    fireEvent.click(screen.getByRole("button", { name: /cell 0 attach now @ 2/ }));
    expect(screen.getByText("attached @ 2")).toBeInTheDocument();
  });
});
