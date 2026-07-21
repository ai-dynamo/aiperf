/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MeasurementRiverSection } from "./MeasurementRiverSection.js";

describe("MeasurementRiverSection", () => {
  it("renders the three storage-policy nodes and commit band", () => {
    render(<MeasurementRiverSection detail="engineering" />);
    expect(screen.getByText("Facts become artifacts in one direction")).toBeInTheDocument();
    expect(screen.getByText("Exact storage")).toBeInTheDocument();
    expect(screen.getByText("MetricsAccumulator")).toBeInTheDocument();
    expect(screen.getByText("Sketch fold")).toBeInTheDocument();
    expect(screen.getByText("native-v2.json")).toBeInTheDocument();
  });

  it("renders the five artifact outputs and the exact/sketch modes", () => {
    render(<MeasurementRiverSection detail="engineering" />);
    expect(screen.getByText("summary JSON / CSV")).toBeInTheDocument();
    expect(screen.getByText("OTLP · MLflow · W&B")).toBeInTheDocument();
    expect(screen.getByText("Exact mode")).toBeInTheDocument();
    expect(screen.getByText("Sketch mode")).toBeInTheDocument();
  });
});
