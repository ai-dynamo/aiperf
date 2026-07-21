/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ParityPage } from "./ParityPage.js";

describe("ParityPage", () => {
  it("renders the AIPerf/Dynamo comparator diagram", () => {
    render(<ParityPage level="developer" />);
    expect(screen.getByText("AIPerf")).toBeInTheDocument();
    expect(screen.getByText("Dynamo")).toBeInTheDocument();
    expect(screen.getByText("byte-equal")).toBeInTheDocument();
    expect(screen.getByText("74 fields (+3 goodput)")).toBeInTheDocument();
  });

  it("shows the mismatch callout at developer level and above", () => {
    render(<ParityPage level="executive" />);
    expect(screen.queryByText("Mismatch → the run bails")).not.toBeInTheDocument();

    render(<ParityPage level="developer" />);
    expect(screen.getByText("Mismatch → the run bails")).toBeInTheDocument();
  });

  it("shows the compared-fields table only at maintainer level", () => {
    render(<ParityPage level="developer" />);
    expect(screen.queryByText("ttft")).not.toBeInTheDocument();

    render(<ParityPage level="maintainer" />);
    expect(screen.getByText("ttft")).toBeInTheDocument();
    expect(screen.getByText("gpu_hours")).toBeInTheDocument();
  });
});
