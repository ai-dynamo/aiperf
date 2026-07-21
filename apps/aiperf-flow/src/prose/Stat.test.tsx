/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Stat } from "./Stat.js";

describe("Stat", () => {
  it("renders label and value", () => {
    render(<Stat label="Throughput" value="1,284 req/s" />);
    expect(screen.getByText("Throughput")).toBeInTheDocument();
    expect(screen.getByText("1,284 req/s")).toBeInTheDocument();
  });

  it("renders a numeric value", () => {
    render(<Stat label="Requests" value={1284} />);
    expect(screen.getByText("1284")).toBeInTheDocument();
  });

  it("omits the trend when absent", () => {
    render(<Stat label="Throughput" value="1,284 req/s" />);
    expect(screen.queryByText("+8.2%")).not.toBeInTheDocument();
  });

  it("renders an optional trend", () => {
    render(<Stat label="Throughput" value="1,284 req/s" trend="+8.2%" />);
    expect(screen.getByText("+8.2%")).toBeInTheDocument();
  });

  it("defaults to the neutral tone (ink-secondary trend)", () => {
    render(<Stat label="Throughput" value="1,284 req/s" trend="+8.2%" />);
    const trend = screen.getByText("+8.2%");
    expect(trend.className).toContain("text-ink-secondary");
  });

  it("maps positive tone to the accent-primary color", () => {
    render(<Stat label="Throughput" value="1,284 req/s" trend="+8.2%" tone="positive" />);
    const trend = screen.getByText("+8.2%");
    expect(trend.className).toContain("text-accent-primary");
  });

  it("maps negative tone to the category-red color", () => {
    render(<Stat label="Error rate" value="0.4%" trend="-1.1%" tone="negative" />);
    const trend = screen.getByText("-1.1%");
    expect(trend.className).toContain("text-category-red");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <Stat label="Throughput" value="1,284 req/s" className="extra-stat-class" />,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-stat-class");
    expect(root.className).toMatch(/rounded-(sm|md|lg|xl|full)/);
  });

  it("gives the label small uppercase tracking-wide styling", () => {
    render(<Stat label="Throughput" value="1,284 req/s" />);
    const label = screen.getByText("Throughput");
    expect(label.className).toContain("uppercase");
    expect(label.className).toContain("tracking-wide");
  });
});
