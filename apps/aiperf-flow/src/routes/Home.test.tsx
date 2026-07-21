/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { Home } from "./Home.js";

describe("Home", () => {
  it("lists every deck with a link to its route", () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );
    const link = screen.getByRole("link", { name: /Segment Pools/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/segment-pools");
  });

  it("renders the page heading", () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "Explainer decks" })).toBeInTheDocument();
  });
});
