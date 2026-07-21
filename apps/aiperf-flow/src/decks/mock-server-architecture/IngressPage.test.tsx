/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { IngressPage } from "./IngressPage.js";

describe("IngressPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <IngressPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Runtime and ingress")).toBeInTheDocument();
    expect(screen.getByText("TCP listener")).toBeInTheDocument();
    expect(screen.getByText("Axum route surface")).toBeInTheDocument();
    expect(
      screen.getAllByText("The tuned Hyper listener accepts TCP and serves the shared router.").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rust/mock-server/src/listener.rs").length).toBeGreaterThan(0);
  });
});
