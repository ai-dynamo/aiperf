/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App.js";

describe("App", () => {
  it("renders a not-found message when no deck route matches", () => {
    window.history.pushState({}, "", "/unregistered-deck");
    render(<App />);
    expect(screen.getByText(/no deck registered/i)).toBeInTheDocument();
  });
});
