// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  guidedRoutePaths,
  presentationRoutePaths,
  routeCapabilities,
  routeSupports,
} from "./routes";

describe("route capabilities", () => {
  it("centralizes navigation, filters, presentation, and atlas state", () => {
    expect(routeCapabilities.map(({ path }) => path)).toEqual([
      "/",
      "/journey",
      "/execution",
      "/data-plane",
      "/observability",
      "/parity",
      "/atlas",
    ]);
    expect(guidedRoutePaths).toEqual([
      "/",
      "/journey",
      "/execution",
      "/data-plane",
      "/observability",
      "/parity",
    ]);
    expect(presentationRoutePaths).toEqual(guidedRoutePaths);
    expect(routeSupports("/execution", "filters")).toBe(true);
    expect(routeSupports("/journey", "filters")).toBe(false);
    expect(routeSupports("/atlas", "atlasState")).toBe(true);
    expect(routeSupports("/atlas", "presentation")).toBe(false);
  });
});
