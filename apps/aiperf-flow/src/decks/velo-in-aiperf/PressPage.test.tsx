/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PressPage } from "./PressPage.js";

describe("PressPage", () => {
  it("starts on the typed-value stage", () => {
    render(<PressPage />);
    expect(screen.getByText("CellMessage::Heartbeat")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "1 / Load typed value" })).toBeInTheDocument();
  });

  it("shows the 16 raw MessagePack bytes on the inspect stage", () => {
    render(<PressPage />);
    fireEvent.click(screen.getByRole("button", { name: "3 / Inspect raw bytes" }));
    expect(screen.getByText("83")).toBeInTheDocument();
    expect(screen.getByText("cb")).toBeInTheDocument();
    expect(screen.getByText("2a")).toBeInTheDocument();
  });

  it("reconstructs the decoded heartbeat at the last stage", () => {
    render(<PressPage />);
    fireEvent.click(screen.getByRole("button", { name: "4 / Reconstruct" }));
    expect(screen.getByText("Decoded Heartbeat")).toBeInTheDocument();
  });

  it("keeps the round-trip callout about NaN and infinity", () => {
    render(<PressPage />);
    expect(screen.getByText("Preserves NaN and infinity")).toBeInTheDocument();
  });
});
