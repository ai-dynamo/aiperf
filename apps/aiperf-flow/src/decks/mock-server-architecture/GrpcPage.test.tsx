/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GrpcPage } from "./GrpcPage.js";

describe("GrpcPage", () => {
  it("renders the chapter heading and catalog entries including the Riva boundary", () => {
    render(
      <ReactFlowProvider>
        <GrpcPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("gRPC and Riva")).toBeInTheDocument();
    expect(screen.getByText("KServe unary ModelInfer")).toBeInTheDocument();
    expect(screen.getByText("Riva NLP is gRPC-only")).toBeInTheDocument();
    expect(
      screen.getAllByText("Riva ASR, TTS, and NLP have no HTTP route in the mock router.").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rust/mock-server/src/grpc.rs").length).toBeGreaterThan(0);
  });
});
