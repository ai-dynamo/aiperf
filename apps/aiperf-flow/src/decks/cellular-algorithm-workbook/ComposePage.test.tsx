/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComposePage } from "./ComposePage.js";
import { cachedRoute, DEFAULT_SELECTION } from "./data.js";

describe("ComposePage", () => {
  it("renders the compose heading and the derived-route success callout for the default selection", () => {
    render(<ComposePage />);
    expect(screen.getByText("Compose an execution route")).toBeInTheDocument();
    const route = cachedRoute(DEFAULT_SELECTION);
    expect(route.valid).toBe(true);
    if (route.valid) {
      expect(
        screen.getByText(`${route.algorithmIds.length} ordered algorithm stops`),
      ).toBeInTheDocument();
    }
  });

  it("renders the storage invariant matrix with all three modes", () => {
    render(<ComposePage />);
    expect(screen.getByRole("heading", { name: "Retain" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Exact fold" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Sketch" })).toBeInTheDocument();
  });

  it("changing a selector to a rejected shape shows a rejection callout", () => {
    render(<ComposePage />);
    // Transport = offline is rejected at the controller-prelaunch stage.
    const transportSelect = screen.getByLabelText("Transport") as HTMLSelectElement;
    fireEvent.change(transportSelect, { target: { value: "offline" } });
    expect(screen.getByText(/offline-transport-rejected/)).toBeInTheDocument();
  });

  it("renders the requested → effective settings callout", () => {
    render(<ComposePage />);
    expect(screen.getByText("Requested → effective runtime settings")).toBeInTheDocument();
  });
});
