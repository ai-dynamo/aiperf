// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WCAG contrast validation for resolved themes.

import { REQUIRED_CONTRAST_PAIRS } from "@aiperf/flow-schema";
import { wcagContrast } from "culori";

import {
  ThemeContrastError,
  ThemeRoleKindError,
  type ResolvedTheme,
} from "./types.js";

export function validateThemeContrast(theme: ResolvedTheme): void {
  for (const pair of REQUIRED_CONTRAST_PAIRS) {
    const foreground = theme.values[pair.foreground];
    const background = theme.values[pair.background];
    if (foreground.kind !== "color" || background.kind !== "color") {
      throw new ThemeRoleKindError(
        `Theme "${theme.id}" contrast roles must contain color values`,
      );
    }

    const ratio = wcagContrast(foreground.value, background.value);
    if (ratio + Number.EPSILON < pair.minRatio) {
      throw new ThemeContrastError(
        `Theme "${theme.id}" contrast for "${pair.foreground}" on "${pair.background}" is ${ratio.toFixed(2)}; requires ${pair.minRatio.toFixed(1)}`,
      );
    }
  }
}
