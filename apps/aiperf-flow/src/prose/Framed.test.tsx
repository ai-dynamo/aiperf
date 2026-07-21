/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Framed } from "./Framed.js";

describe("Framed", () => {
  it("renders its children inside a bordered panel", () => {
    render(<Framed>Body content</Framed>);
    expect(screen.getByText("Body content")).toBeInTheDocument();
  });

  it("defaults to the page surface role", () => {
    render(<Framed>Body</Framed>);
    expect(screen.getByText("Body").className).toContain("bg-surface-page");
  });

  it("applies the given surface role", () => {
    render(<Framed surfaceRole="elevated">Body</Framed>);
    expect(screen.getByText("Body").className).toContain("bg-surface-elevated");
  });

  it("merges a caller-supplied className", () => {
    render(<Framed className="extra-framed-class">Body</Framed>);
    expect(screen.getByText("Body").className).toContain("extra-framed-class");
  });
});
