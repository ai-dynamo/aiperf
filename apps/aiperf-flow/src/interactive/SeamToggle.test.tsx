/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SeamToggle, type SeamToggleOption } from "./SeamToggle.js";

type Clock = "real" | "sim";
const OPTIONS: SeamToggleOption<Clock>[] = [
  { value: "real", label: "RealClock", tone: "green" },
  { value: "sim", label: "SimClock", tone: "purple" },
];

describe("SeamToggle", () => {
  it("renders each option as a button", () => {
    render(<SeamToggle label="Clock" options={OPTIONS} value="real" onChange={() => {}} />);
    expect(screen.getByRole("button", { name: "RealClock" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "SimClock" })).toBeInTheDocument();
  });

  it("marks the selected option with aria-pressed", () => {
    render(<SeamToggle options={OPTIONS} value="real" onChange={() => {}} />);
    expect(screen.getByRole("button", { name: "RealClock" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "SimClock" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onChange with the clicked value", () => {
    const onChange = vi.fn();
    render(<SeamToggle options={OPTIONS} value="real" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "SimClock" }));
    expect(onChange).toHaveBeenCalledWith("sim");
  });

  it("moves the selection when driven by state", () => {
    function Controlled(): React.JSX.Element {
      const [value, setValue] = useState<Clock>("real");
      return <SeamToggle options={OPTIONS} value={value} onChange={setValue} />;
    }
    render(<Controlled />);
    fireEvent.click(screen.getByRole("button", { name: "SimClock" }));
    expect(screen.getByRole("button", { name: "SimClock" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "RealClock" })).toHaveAttribute("aria-pressed", "false");
  });
});
