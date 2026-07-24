/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProtocolPage } from "./ProtocolPage.js";

describe("ProtocolPage", () => {
  it("renders the intro and the hub", () => {
    render(<ProtocolPage />);
    expect(screen.getByText("One child lifecycle")).toBeInTheDocument();
    // The hub renders in both the wide (ring) and narrow (stacked) layouts.
    expect(screen.getAllByText("How is a run isolated?").length).toBeGreaterThan(0);
  });

  it("renders spoke cards", () => {
    render(<ProtocolPage />);
    expect(screen.getAllByText("Resolve and spawn").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Terminal contract").length).toBeGreaterThan(0);
  });

  it("renders the evidence anchors", () => {
    render(<ProtocolPage />);
    expect(screen.getByText("rust/cli/src/exec_bin.rs")).toBeInTheDocument();
  });
});
