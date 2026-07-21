/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { OverviewPage } from "./OverviewPage.js";

describe("OverviewPage", () => {
  it("renders the four-layer stack labels", () => {
    render(<OverviewPage level="developer" />);
    expect(screen.getByText("RUST CLI ENTRY POINT")).toBeInTheDocument();
    expect(screen.getByText("EXECUTION ENGINE")).toBeInTheDocument();
    expect(screen.getByText("AIPERF LIBRARY")).toBeInTheDocument();
    expect(screen.getByText("ENGINE / WIRE")).toBeInTheDocument();
  });

  it("renders the key node titles", () => {
    render(<OverviewPage level="developer" />);
    expect(screen.getByText("aiperf profile")).toBeInTheDocument();
    expect(screen.getAllByText("RunnerApplication").length).toBeGreaterThan(0);
    expect(screen.getByText("EngineHost → SteppableReplay")).toBeInTheDocument();
  });

  it("shows maintainer subtitles only at maintainer level", () => {
    render(<OverviewPage level="executive" />);
    expect(screen.queryByText("load.rs / yaml.rs")).not.toBeInTheDocument();

    render(<OverviewPage level="maintainer" />);
    expect(screen.getByText("load.rs / yaml.rs")).toBeInTheDocument();
  });

  it("renders the summary callouts", () => {
    render(<OverviewPage level="developer" />);
    expect(screen.getByText("Shared above the seam")).toBeInTheDocument();
    expect(screen.getByText("No sockets (dynosim)")).toBeInTheDocument();
    expect(screen.getByText("Verified")).toBeInTheDocument();
  });
});
