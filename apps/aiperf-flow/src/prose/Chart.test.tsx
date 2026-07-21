/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BarChart, LineChart } from "./Chart.js";

const DATA = [
  { label: "Mon", value: 10 },
  { label: "Tue", value: 40 },
  { label: "Wed", value: 25 },
];

describe("BarChart", () => {
  it("renders one bar per data point", () => {
    const { container } = render(<BarChart data={DATA} />);
    const rects = container.querySelectorAll("rect");
    expect(rects.length).toBe(DATA.length);
  });

  it("renders x-axis labels for each data point", () => {
    render(<BarChart data={DATA} />);
    expect(screen.getByText("Mon")).toBeInTheDocument();
    expect(screen.getByText("Tue")).toBeInTheDocument();
    expect(screen.getByText("Wed")).toBeInTheDocument();
  });

  it("scales the tallest bar to the max value", () => {
    const { container } = render(<BarChart data={DATA} height={100} />);
    const rects = Array.from(container.querySelectorAll("rect"));
    const heights = rects.map((r) => Number(r.getAttribute("height")));
    // The "Tue" bar (value 40, the max) should be the tallest.
    expect(Math.max(...heights)).toBe(heights[1]);
  });

  it("uses categoryBgClassName for the bar fill via a static lookup", () => {
    const { container } = render(<BarChart data={DATA} color="purple" />);
    const rect = container.querySelector("rect");
    expect(rect?.getAttribute("class")).toContain("fill-category-purple");
  });

  it("does not throw and renders no bars for empty data", () => {
    const { container } = render(<BarChart data={[]} />);
    expect(container.querySelectorAll("rect").length).toBe(0);
  });

  it("merges a caller-supplied className onto the svg root", () => {
    const { container } = render(<BarChart data={DATA} className="extra-bar-class" />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("class")).toContain("extra-bar-class");
  });
});

describe("LineChart", () => {
  it("renders a single polyline through the data points", () => {
    const { container } = render(<LineChart data={DATA} />);
    const polylines = container.querySelectorAll("polyline");
    expect(polylines.length).toBe(1);
  });

  it("renders a circle marker for each data point", () => {
    const { container } = render(<LineChart data={DATA} />);
    const circles = container.querySelectorAll("circle");
    expect(circles.length).toBe(DATA.length);
  });

  it("renders x-axis labels for each data point", () => {
    render(<LineChart data={DATA} />);
    expect(screen.getByText("Mon")).toBeInTheDocument();
    expect(screen.getByText("Tue")).toBeInTheDocument();
    expect(screen.getByText("Wed")).toBeInTheDocument();
  });

  it("uses categoryBgClassName-derived color for markers via a static lookup", () => {
    const { container } = render(<LineChart data={DATA} color="blue" />);
    const circle = container.querySelector("circle");
    expect(circle?.getAttribute("class")).toContain("fill-category-blue");
  });

  it("does not throw and renders no polyline for empty data", () => {
    const { container } = render(<LineChart data={[]} />);
    expect(container.querySelectorAll("polyline").length).toBe(0);
    expect(container.querySelectorAll("circle").length).toBe(0);
  });

  it("merges a caller-supplied className onto the svg root", () => {
    const { container } = render(<LineChart data={DATA} className="extra-line-class" />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("class")).toContain("extra-line-class");
  });
});
