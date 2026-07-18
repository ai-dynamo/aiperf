// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  hudVisibilityFor,
  type HudVisibility,
  type HudVisibilityInput,
} from "../src/hud-policy.js";

function input(overrides: Partial<HudVisibilityInput> = {}): HudVisibilityInput {
  return {
    playing: true,
    exploring: false,
    commandOpen: false,
    focusedWithinHud: false,
    inactive: false,
    ...overrides,
  };
}

describe("hudVisibilityFor", () => {
  test.each<{ label: string; overrides: Partial<HudVisibilityInput> }>([
    { label: "paused playback", overrides: { playing: false } },
    { label: "active exploration", overrides: { exploring: true } },
    { label: "open command palette", overrides: { commandOpen: true } },
    { label: "focus inside the HUD", overrides: { focusedWithinHud: true } },
  ])(
    "keeps chrome present while $label overrides quiet policy",
    ({ overrides }) => {
      expect(hudVisibilityFor(input(overrides))).toBe<HudVisibility>("present");
    },
  );

  test("quiets chrome during uninterrupted active playback", () => {
    expect(hudVisibilityFor(input({ inactive: false }))).toBe<HudVisibility>(
      "quiet",
    );
  });

  test("hides chrome once playback goes inactive without interaction", () => {
    expect(hudVisibilityFor(input({ inactive: true }))).toBe<HudVisibility>(
      "hidden",
    );
  });

  test("prefers present over hidden when playback is inactive but explored", () => {
    expect(
      hudVisibilityFor(input({ inactive: true, exploring: true })),
    ).toBe<HudVisibility>("present");
  });
});
