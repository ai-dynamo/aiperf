// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { parseAtlasSearch } from "./search";

describe("atlas search parsing", () => {
  it("preserves valid fields when another search field is invalid", () => {
    expect(
      parseAtlasSearch({
        audience: "maintainer",
        modes: "online_http",
        statuses: "not-a-status",
        present: "true",
      }),
    ).toEqual({
      audience: "maintainer",
      modes: "online_http",
      present: true,
    });
  });
});
