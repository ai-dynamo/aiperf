/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ProtocolPage } from "./ProtocolPage.js";

describe("ProtocolPage", () => {
  it("renders the intro and lifecycle nodes", () => {
    render(
      <ReactFlowProvider>
        <ProtocolPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("One child lifecycle")).toBeInTheDocument();
    expect(screen.getByText("exec_bin::resolve")).toBeInTheDocument();
    expect(screen.getByText("execute operation")).toBeInTheDocument();
  });

  it("renders the callouts and evidence anchors", () => {
    render(
      <ReactFlowProvider>
        <ProtocolPage />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Isolation")).toBeInTheDocument();
    expect(screen.getByText("Override")).toBeInTheDocument();
    expect(screen.getByText("rust/cli/src/exec_bin.rs")).toBeInTheDocument();
  });
});
