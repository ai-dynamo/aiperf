/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CombinedTimeline } from "./CombinedTimeline.js";

describe("CombinedTimeline", () => {
  it("labels the shared t* line and every trace·lane row", () => {
    render(<CombinedTimeline tStars={[0, 0, 0]} />);
    expect(screen.getByText("t*")).toBeInTheDocument();
    expect(screen.getByText("linear·main")).toBeInTheDocument();
    expect(screen.getByText("one-sub·main")).toBeInTheDocument();
    expect(screen.getByText("one-sub·sub")).toBeInTheDocument();
    expect(screen.getByText("two-subs·main")).toBeInTheDocument();
    expect(screen.getByText("two-subs·alpha")).toBeInTheDocument();
    expect(screen.getByText("two-subs·beta")).toBeInTheDocument();
  });

  it("shifts tick labels with a + prefix on the positive side", () => {
    render(<CombinedTimeline tStars={[6, 0, 0]} />);
    expect(screen.getAllByText(/^\+\d+s$/).length).toBeGreaterThan(0);
  });
});
