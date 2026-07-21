/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Slide } from "./Slide.js";
import type { SlideDefinition } from "./types.js";

const slide: SlideDefinition = {
  id: "s1",
  eyebrow: "Overview",
  title: "Rows in → wire bytes out",
  lede: "Every request body starts as a dataset row.",
  narration: "Dataset rows become wire bytes.",
  caption: "BUILD → FREEZE → DISPATCH",
  nodes: [
    { id: "header", type: "header", position: { x: 0, y: 0 }, data: { title: "BUILD" } },
    { id: "panel1", type: "panel", position: { x: 0, y: 80 }, data: { title: "Turn.body" } },
  ],
  edges: [],
};

describe("Slide", () => {
  it("renders the slide's title and lede", () => {
    render(<Slide slide={slide} />);
    expect(screen.getByText("Rows in → wire bytes out")).toBeInTheDocument();
    expect(screen.getByText("Every request body starts as a dataset row.")).toBeInTheDocument();
  });

  it("renders the first reveal-order node immediately", () => {
    render(<Slide slide={slide} />);
    expect(screen.getByText("BUILD")).toBeInTheDocument();
  });
});
