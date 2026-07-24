/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, type RenderResult } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TimeAxis } from "./TimeAxis.js";
import { Lane } from "./Lane.js";
import { StageRegion } from "./StageRegion.js";
import { SeamFrame } from "./SeamFrame.js";
import { RequestLine } from "./RequestLine.js";
import { EventMarker } from "./EventMarker.js";

function renderSvg(node: React.ReactNode): RenderResult {
  return render(<svg width={400} height={300}>{node}</svg>);
}

describe("TimeAxis", () => {
  it("draws the unit caption + a tick label per tick, with SVG-safe stroke classes", () => {
    const { container } = renderSvg(
      <TimeAxis x1={20} x2={380} y={40} unitLabel="RealClock · wall-ms" ticks={[{ x: 20, label: "0" }, { x: 380, label: "200" }]} />,
    );
    expect(container.textContent).toContain("RealClock · wall-ms");
    expect(container.textContent).toContain("0");
    expect(container.textContent).toContain("200");
    const line = container.querySelector("line");
    expect(line?.getAttribute("stroke")).toBe("currentColor");
    // ink-based helpers must NOT emit a bg-*/border-* class on an SVG shape.
    expect(line?.getAttribute("class") ?? "").not.toMatch(/bg-|border-/);
  });
});

describe("Lane", () => {
  it("draws a fill-category band + gutter label (no bg-* on the rect)", () => {
    const { container } = renderSvg(
      <Lane x={100} y={50} width={280} height={40} label="Dataset" labelX={10} tone="green" />,
    );
    expect(container.textContent).toContain("Dataset");
    const rect = container.querySelector("rect");
    expect(rect?.getAttribute("class")).toContain("fill-category-green");
    expect(rect?.getAttribute("class") ?? "").not.toMatch(/\bbg-/);
  });
});

describe("StageRegion", () => {
  it("fires onClick and exposes a drill accessible name", () => {
    const onClick = vi.fn();
    const { getByRole } = renderSvg(
      <StageRegion x={0} y={0} width={80} height={30} label="Clock seam" tone="orange" onClick={onClick} />,
    );
    const region = getByRole("button", { name: "Drill into Clock seam" });
    fireEvent.click(region);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("colors the block with fill-/stroke-category (not bg-/border-) and marks active state", () => {
    const { container } = renderSvg(
      <StageRegion x={0} y={0} width={80} height={30} label="Transport seam" tone="yellow" active onClick={() => {}} />,
    );
    const rect = container.querySelector("rect");
    expect(rect?.getAttribute("class")).toContain("fill-category-yellow");
    expect(rect?.getAttribute("class")).toContain("stroke-category-yellow");
    expect(rect?.getAttribute("class") ?? "").not.toMatch(/\bbg-|\bborder-/);
    expect(container.querySelector('[data-testid="stage-region"]')?.getAttribute("data-active")).toBe("true");
  });
});

describe("SeamFrame", () => {
  it("draws a dashed translucent frame + label", () => {
    const { container } = renderSvg(
      <SeamFrame x={0} y={0} width={200} height={120} label="Transport" tone="yellow" />,
    );
    expect(container.textContent).toContain("Transport");
    const rect = container.querySelector("rect");
    expect(rect?.getAttribute("stroke-dasharray")).toBe("5 4");
    expect(rect?.getAttribute("class")).toContain("stroke-category-yellow");
  });
});

describe("RequestLine", () => {
  it("renders a polyline through the given points with a stroke-category class", () => {
    const { container } = renderSvg(
      <RequestLine tone="cyan" reducedMotion points={[{ x: 0, y: 0 }, { x: 10, y: 20 }, { x: 30, y: 5 }]} />,
    );
    const line = container.querySelector('[data-testid="request-line"]');
    expect(line?.getAttribute("points")).toBe("0,0 10,20 30,5");
    expect(line?.getAttribute("class")).toContain("stroke-category-cyan");
    expect(line?.getAttribute("fill")).toBe("none");
  });
});

describe("EventMarker", () => {
  it("draws a fill-category dot and a label", () => {
    const { container } = renderSvg(<EventMarker x={40} y={40} tone="blue" label="TTFT" />);
    expect(container.textContent).toContain("TTFT");
    const circle = container.querySelector("circle");
    expect(circle?.getAttribute("class")).toContain("fill-category-blue");
    expect(container.querySelector('[data-testid="event-marker"]')?.getAttribute("data-active")).toBe("false");
  });

  it("marks active state (adds the pulse halo)", () => {
    const { container } = renderSvg(<EventMarker x={40} y={40} tone="blue" label="TTFT" active />);
    expect(container.querySelector('[data-testid="event-marker"]')?.getAttribute("data-active")).toBe("true");
    // Active marker draws two circles (halo + dot).
    expect(container.querySelectorAll("circle").length).toBeGreaterThanOrEqual(2);
  });
});
