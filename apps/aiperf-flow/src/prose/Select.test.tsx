/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Select } from "./Select.js";

const OPTIONS = [
  { value: "a", label: "Option A" },
  { value: "b", label: "Option B" },
];

describe("Select", () => {
  it("renders a native select with the given options", () => {
    render(<Select options={OPTIONS} value="a" onChange={vi.fn()} />);
    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(select.tagName).toBe("SELECT");
    expect(select.value).toBe("a");
    expect(screen.getByRole("option", { name: "Option A" })).toBeDefined();
    expect(screen.getByRole("option", { name: "Option B" })).toBeDefined();
  });

  it("calls onChange with the new value when the selection changes", () => {
    const onChange = vi.fn();
    render(<Select options={OPTIONS} value="a" onChange={onChange} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "b" } });
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("renders an associated label when label is given", () => {
    render(<Select options={OPTIONS} value="a" onChange={vi.fn()} label="Model" />);
    expect(screen.getByLabelText("Model")).toBe(screen.getByRole("combobox"));
  });

  it("does not render a label element when label is omitted", () => {
    const { container } = render(<Select options={OPTIONS} value="a" onChange={vi.fn()} />);
    expect(container.querySelector("label")).toBeNull();
  });

  it("merges a caller-supplied className", () => {
    render(<Select options={OPTIONS} value="a" onChange={vi.fn()} className="extra-class" />);
    expect(screen.getByRole("combobox").className).toContain("extra-class");
  });
});
