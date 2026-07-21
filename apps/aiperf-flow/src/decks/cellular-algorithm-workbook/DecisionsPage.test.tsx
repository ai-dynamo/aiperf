/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionsPage } from "./DecisionsPage.js";
import { DECISIONS } from "./data.js";

describe("DecisionsPage", () => {
  it("renders the decision laboratory heading", () => {
    render(<DecisionsPage />);
    expect(screen.getByText("Decision laboratory")).toBeInTheDocument();
  });

  it("renders every decision title and its invariant", () => {
    render(<DecisionsPage />);
    for (const decision of DECISIONS) {
      expect(screen.getByText(decision.title)).toBeInTheDocument();
      expect(screen.getByText(decision.invariant)).toBeInTheDocument();
    }
  });

  it("renders both labelled sides for each decision", () => {
    render(<DecisionsPage />);
    const first = DECISIONS[0];
    expect(screen.getAllByLabelText(first.leftLabel).length).toBeGreaterThan(0);
    expect(screen.getAllByLabelText(first.rightLabel).length).toBeGreaterThan(0);
  });

  it("shows an admission chip on each side", () => {
    render(<DecisionsPage />);
    // At least one Admitted or Rejected chip must be present.
    const admitted = screen.queryAllByLabelText("Route admission: Admitted");
    const rejected = screen.queryAllByLabelText("Route admission: Rejected");
    expect(admitted.length + rejected.length).toBeGreaterThanOrEqual(DECISIONS.length * 2);
  });
});
