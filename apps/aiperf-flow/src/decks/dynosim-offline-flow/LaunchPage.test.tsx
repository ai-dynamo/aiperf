/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LaunchPage } from "./LaunchPage.js";

describe("LaunchPage", () => {
  it("renders the four preflight steps", () => {
    render(<LaunchPage level="developer" />);
    expect(screen.getByText("project")).toBeInTheDocument();
    expect(screen.getByText("registry")).toBeInTheDocument();
    expect(screen.getByText("re-exec")).toBeInTheDocument();
    expect(screen.getByText("handle_v2")).toBeInTheDocument();
    expect(screen.getByText("reject")).toBeInTheDocument();
  });

  it("shows the fail-closed callout at developer level and above", () => {
    render(<LaunchPage level="executive" />);
    expect(screen.queryByText("What fails closed")).not.toBeInTheDocument();

    render(<LaunchPage level="developer" />);
    expect(screen.getByText("What fails closed")).toBeInTheDocument();
  });

  it("shows maintainer file citations only at maintainer level", () => {
    render(<LaunchPage level="developer" />);
    expect(screen.queryByText("dynosim_offline registered")).not.toBeInTheDocument();

    render(<LaunchPage level="maintainer" />);
    expect(screen.getByText("dynosim_offline registered")).toBeInTheDocument();
  });
});
