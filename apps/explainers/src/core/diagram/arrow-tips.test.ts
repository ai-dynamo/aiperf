/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactElement } from "react";
import { describe, expect, it } from "vitest";

import {
  TIP_SIZE_UNITS,
  markerGeometry,
  resolveMarkerTip,
  type MarkerTipSize,
} from "./arrow-tips.js";

describe("tee tip inset vs geometry", () => {
  const sizes: MarkerTipSize[] = ["sm", "md", "lg"];

  it("keeps path inset small for every size", () => {
    for (const size of sizes) {
      const tip = resolveMarkerTip({ kind: "tee", size });
      expect(tip).not.toBeNull();
      const base = TIP_SIZE_UNITS[size];
      expect(tip!.insetUnits).toBe(Math.max(1, base * 0.2));
    }
  });

  it("sizes the perpendicular bar from TIP_SIZE_UNITS, not reduced inset", () => {
    for (const size of sizes) {
      const tip = resolveMarkerTip({ kind: "tee", size });
      expect(tip).not.toBeNull();
      const base = TIP_SIZE_UNITS[size];
      const geom = markerGeometry(tip!);
      const rect = geom.children as ReactElement<{
        width: number;
        height: number;
      }>;

      expect(rect.props.height).toBe(base);
      expect(geom.markerHeight).toBe(base + 1);
      // Must not collapse to the reduced inset (~1) for md/lg.
      expect(rect.props.height).toBeGreaterThan(tip!.insetUnits);
    }
  });

  it("centers the tee bar on the path endpoint", () => {
    for (const size of sizes) {
      const tip = resolveMarkerTip({ kind: "tee", size });
      expect(tip).not.toBeNull();
      const base = TIP_SIZE_UNITS[size];
      const geom = markerGeometry(tip!);
      const rect = geom.children as ReactElement<{
        width: number;
        height: number;
      }>;

      expect(geom.refX).toBeCloseTo(rect.props.width / 2);
      expect(geom.refY).toBeCloseTo(base / 2);
    }
  });
});
