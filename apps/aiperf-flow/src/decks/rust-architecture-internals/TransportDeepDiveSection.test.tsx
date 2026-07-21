/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TransportDeepDiveSection } from "./TransportDeepDiveSection.js";

describe("TransportDeepDiveSection", () => {
  it("renders the cold path chain and default HTTP hot path", () => {
    render(<TransportDeepDiveSection detail="engineering" />);
    expect(screen.getByText("Cold endpoint preparation feeds a hot wire lane")).toBeInTheDocument();
    expect(screen.getByText("EndpointRegistry::prepare")).toBeInTheDocument();
    expect(screen.getByText("EndpointKey(u32)")).toBeInTheDocument();
    expect(screen.getAllByText("TransportSink").length).toBeGreaterThan(0);
    expect(screen.getByText("SSE reader")).toBeInTheDocument();
  });

  it("switches to the gRPC hot path", () => {
    render(<TransportDeepDiveSection detail="engineering" />);
    fireEvent.click(screen.getByRole("button", { name: "gRPC" }));
    expect(screen.getByText("GrpcTransportSink")).toBeInTheDocument();
    expect(screen.getByText("Tonic dispatch")).toBeInTheDocument();
  });
});
