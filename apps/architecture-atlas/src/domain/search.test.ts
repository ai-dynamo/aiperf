// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { parseAtlasSearch } from "./search";

describe("atlas search parsing", () => {
  it("preserves valid fields when another search field is invalid", () => {
    expect(
      parseAtlasSearch({
        audience: "maintainer",
        layout: "lifecycle",
        modes: "online_http",
        ownership: "rust",
        query: "Clock",
        selected: "component.clock-seam",
        statuses: "not-a-status",
        present: "true",
      }),
    ).toEqual({
      audience: "maintainer",
      layout: "lifecycle",
      modes: "online_http",
      ownership: "rust",
      present: true,
      query: "Clock",
      selected: "component.clock-seam",
    });
  });

  it("drops invalid atlas fields independently", () => {
    expect(
      parseAtlasSearch({
        audience: "developer",
        layout: "radial",
        ownership: "rust,operator",
        query: 42,
        selected: "../clock",
      }),
    ).toEqual({ audience: "developer" });
  });

  it("preserves meaningful query whitespace for controlled URL input", () => {
    expect(parseAtlasSearch({ query: "virtual clock " })).toEqual({
      query: "virtual clock ",
    });
  });
});
