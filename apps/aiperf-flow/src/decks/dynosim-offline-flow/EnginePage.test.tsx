/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EnginePage } from "./EnginePage.js";

describe("EnginePage", () => {
  it("defaults to aggregated topology with a KV router", () => {
    render(<EnginePage level="developer" />);
    expect(screen.getByText("SteppableAgg")).toBeInTheDocument();
    expect(screen.getByText("KV router")).toBeInTheDocument();
    expect(screen.getByText("w0")).toBeInTheDocument();
    expect(screen.getByText("w1")).toBeInTheDocument();
    expect(screen.getByText("w2")).toBeInTheDocument();
  });

  it("switching topology to single shows one worker and no router", () => {
    render(<EnginePage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "single" }));
    expect(screen.getByText("SteppableEngine")).toBeInTheDocument();
    expect(screen.getByText("w0")).toBeInTheDocument();
    expect(screen.queryByText("w1")).not.toBeInTheDocument();
  });

  it("switching topology to disaggregated shows prefill and decode pools", () => {
    render(<EnginePage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "disaggregated" }));
    expect(screen.getByText("SteppableDisagg")).toBeInTheDocument();
    expect(screen.getByText("prefill")).toBeInTheDocument();
    expect(screen.getByText("decode")).toBeInTheDocument();
    expect(screen.getByText("p0")).toBeInTheDocument();
    expect(screen.getByText("d0")).toBeInTheDocument();
  });

  it("switching router to round robin swaps the routing callout", () => {
    render(<EnginePage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "round robin" }));
    expect(screen.getByText("Round-robin")).toBeInTheDocument();
    expect(screen.queryByText("KV router")).not.toBeInTheDocument();
  });
});
