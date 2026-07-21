/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SeamsPage } from "./SeamsPage.js";

describe("SeamsPage", () => {
  it("defaults to the dynosim offline mode callout", () => {
    render(<SeamsPage level="developer" />);
    expect(screen.getByRole("button", { name: "dynosim offline" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/virtual clock, in-process engine, deterministic/)).toBeInTheDocument();
  });

  it("switching to HTTP online updates the mode callout", () => {
    render(<SeamsPage level="developer" />);
    fireEvent.click(screen.getByRole("button", { name: "HTTP online" }));
    expect(screen.getByText(/real HTTP to a real server, wall clock/)).toBeInTheDocument();
  });

  it("renders the shared spine and both fork lanes", () => {
    render(<SeamsPage level="developer" />);
    expect(screen.getByText("aiperf profile")).toBeInTheDocument();
    expect(screen.getByText("RequestSink<HttpRequest>")).toBeInTheDocument();
    expect(screen.getByText("TransportSink")).toBeInTheDocument();
    expect(screen.getByText("DynosimSink")).toBeInTheDocument();
  });

  it("shows the composition-not-is_virtual callout at developer level and above", () => {
    render(<SeamsPage level="executive" />);
    expect(screen.queryByText("The fork is chosen at composition — not is_virtual()")).not.toBeInTheDocument();

    render(<SeamsPage level="developer" />);
    expect(screen.getByText("The fork is chosen at composition — not is_virtual()")).toBeInTheDocument();
  });
});
