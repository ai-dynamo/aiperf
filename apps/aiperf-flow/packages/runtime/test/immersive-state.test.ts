// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  createImmersiveState,
  immersiveReducer,
  type ImmersiveState,
} from "../src/immersive-state.js";

describe("immersiveReducer", () => {
  test("restores an exact immutable serializable snapshot", () => {
    const snapshot: ImmersiveState = {
      selectedEntityId: "entity",
      contextLensOpen: true,
      focusWorldEntityId: "focus",
      comparisonEntityId: "comparison",
      commandOpen: true,
      hud: "quiet",
      fullscreen: "layout",
    };

    const restored = immersiveReducer(createImmersiveState(), {
      type: "replace",
      state: snapshot,
    });

    expect(restored).toEqual(snapshot);
    expect(restored).not.toBe(snapshot);
    expect(Object.isFrozen(restored)).toBe(true);
    expect(JSON.parse(JSON.stringify(restored))).toEqual(snapshot);
  });
});
