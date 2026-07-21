/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { LlmProtocolsPage } from "./LlmProtocolsPage.js";

describe("LlmProtocolsPage", () => {
  it("renders the chapter heading and catalog entries", () => {
    render(
      <ReactFlowProvider>
        <LlmProtocolsPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("LLM protocols")).toBeInTheDocument();
    expect(screen.getByText("OpenAI chat completions")).toBeInTheDocument();
    expect(screen.getByText("SSE stream assembly")).toBeInTheDocument();
    expect(screen.getByText("Anthropic Messages")).toBeInTheDocument();
    expect(
      screen.getAllByText("Generated token events precede terminal usage and the stream terminator.")
        .length,
    ).toBeGreaterThan(0);
  });
});
