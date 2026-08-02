/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { RustArchitectureInternalsDeck } from "./RustArchitectureInternalsDeck.js";

describe("RustArchitectureInternalsDeck", () => {
  it("composes the hero and all thirteen sections in one view", () => {
    render(<RustArchitectureInternalsDeck />);
    expect(screen.getByText("Inside Rust AIPerf")).toBeInTheDocument();
    expect(screen.getByText("One run crosses one child process boundary")).toBeInTheDocument();
    expect(screen.getByText("Coordinator stages surround a separate phase clock")).toBeInTheDocument();
    expect(screen.getByText("A frozen universe, composed once")).toBeInTheDocument();
    expect(screen.getByText("Cold endpoint preparation feeds a hot wire lane")).toBeInTheDocument();
    expect(screen.getByText("Facts become artifacts in one direction")).toBeInTheDocument();
    expect(screen.getByText("The architecture in seven concrete mechanisms")).toBeInTheDocument();
  });

  it("drives the global detail level from the hero control", () => {
    render(<RustArchitectureInternalsDeck />);
    expect(screen.queryAllByText("source evidence").length).toBe(0);
    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    expect(screen.queryAllByText("source evidence").length).toBeGreaterThan(0);
  });
});
