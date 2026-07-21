/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CompositionConstellationSection } from "./CompositionConstellationSection.js";

describe("CompositionConstellationSection", () => {
  it("renders the registry hub, coordinator, and all seven satellites", () => {
    render(<CompositionConstellationSection detail="engineering" />);
    expect(screen.getByText("A frozen universe, composed once")).toBeInTheDocument();
    expect(screen.getByText("AIPerfRegistry")).toBeInTheDocument();
    expect(screen.getByText("COORDINATOR")).toBeInTheDocument();
    for (const label of ["endpoints", "dataset formats", "samplers", "transports", "workloads", "exporters", "actuators"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("renders the prepared-execution-selection callout", () => {
    render(<CompositionConstellationSection detail="engineering" />);
    expect(screen.getByText("Prepared execution selection")).toBeInTheDocument();
  });
});
