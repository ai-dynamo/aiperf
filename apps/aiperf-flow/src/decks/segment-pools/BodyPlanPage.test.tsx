/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BodyPlanPage } from "./BodyPlanPage.js";

describe("BodyPlanPage", () => {
  it("renders the heading and framing copy", () => {
    render(<BodyPlanPage />);
    expect(screen.getByText("BodyPlan — shape now, bytes later")).toBeInTheDocument();
    expect(
      screen.getByText(/literals are the/),
    ).toBeInTheDocument();
  });

  it("defaults to BodyPlan::Fields with stream on, tools off, and materializes accordingly", () => {
    render(<BodyPlanPage />);
    expect(screen.getByText('"messages":[')).toBeInTheDocument();
    expect(screen.getByText(',"stream":true')).toBeInTheDocument();
    expect(screen.queryByText(/"tools":/)).not.toBeInTheDocument();
    // serde ops stat should be nonzero in Fields mode
    expect(screen.getByText("serde_json ops (hot path)")).toBeInTheDocument();
  });

  it("toggling tools on adds the tools segment splice to the materialized output", async () => {
    render(<BodyPlanPage />);
    expect(screen.queryByText(/"tools":/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /tools off/i }));
    expect(screen.getByText(/"tools":/)).toBeInTheDocument();
  });

  it("toggling stream off changes the materialized stream tail from true to false", async () => {
    render(<BodyPlanPage />);
    expect(screen.getByText(',"stream":true')).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /stream on/i }));
    expect(screen.queryByText(',"stream":true')).not.toBeInTheDocument();
    expect(screen.getByText(',"stream":false')).toBeInTheDocument();
  });

  it("toggling rawMode switches to BodyPlan::Raw and replaces the materialized body entirely", async () => {
    render(<BodyPlanPage />);
    expect(screen.getByText('"messages":[')).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "BodyPlan::Raw" }));

    expect(screen.queryByText('"messages":[')).not.toBeInTheDocument();
    expect(
      screen.getByText(
        '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":false}',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(',"stream":true')).toBeInTheDocument();
    // serde_json ops should be zero in Raw mode (whole body is a passthrough segment)
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("tools toggle is not shown while in raw mode", async () => {
    render(<BodyPlanPage />);
    expect(screen.getByRole("button", { name: /tools off/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "BodyPlan::Raw" }));
    expect(screen.queryByRole("button", { name: /tools/i })).not.toBeInTheDocument();
  });
});
