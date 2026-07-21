/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WorkerTopologySection } from "./WorkerTopologySection.js";

describe("WorkerTopologySection", () => {
  it("defaults to sharded with three OS threads and a run coordinator", () => {
    render(<WorkerTopologySection detail="engineering" />);
    expect(screen.getByText("Scale by tiling self-contained execution cells")).toBeInTheDocument();
    expect(screen.getByText("run coordinator")).toBeInTheDocument();
    expect(screen.getByText("OS THREAD 0")).toBeInTheDocument();
    expect(screen.getByText("OS THREAD 2")).toBeInTheDocument();
  });

  it("switches to the local single reactor", () => {
    render(<WorkerTopologySection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "1 worker" }));
    expect(screen.getByText("COORDINATOR REACTOR")).toBeInTheDocument();
  });

  it("switches to cellular processes with controller merge", () => {
    render(<WorkerTopologySection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "N processes" }));
    expect(screen.getByText("cell controller")).toBeInTheDocument();
    expect(screen.getByText("aiperf --cell 1")).toBeInTheDocument();
    expect(screen.getByText("controller merge")).toBeInTheDocument();
  });
});
