// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  createFocusCoordinator,
  type FocusCoordinatorState,
} from "../../src/semantic/focus-coordinator.js";
import type { SemanticProjection } from "../../src/evaluate/types.js";

const projection: SemanticProjection = {
  sceneId: "lifecycle",
  readingOrder: ["arrive", "admit", "observe"],
  entities: [
    { id: "arrive", label: "Arrive" },
    { id: "admit", label: "Admit" },
    { id: "observe", label: "Observe" },
  ],
  relations: [
    { id: "r0", fromId: "arrive", toId: "admit", role: "next" },
    { id: "r1", fromId: "admit", toId: "observe", role: "next" },
  ],
};

function snapshot(state: FocusCoordinatorState): Readonly<{
  focusedEntityId: string | null;
  selectedEntityId: string | null;
  visualSelectedEntityId: string | null;
}> {
  return {
    focusedEntityId: state.focusedEntityId,
    selectedEntityId: state.selectedEntityId,
    visualSelectedEntityId: state.visualSelectedEntityId,
  };
}

describe("focus coordinator", () => {
  test("maps Canvas visual selection onto semantic focus", () => {
    const coordinator = createFocusCoordinator(projection);
    const next = coordinator.selectFromVisual("admit");

    expect(snapshot(next)).toEqual({
      focusedEntityId: "admit",
      selectedEntityId: "admit",
      visualSelectedEntityId: "admit",
    });
  });

  test("maps semantic keyboard activation onto visual selection", () => {
    const coordinator = createFocusCoordinator(projection);
    const next = coordinator.activateFromSemantic("observe");

    expect(snapshot(next)).toEqual({
      focusedEntityId: "observe",
      selectedEntityId: "observe",
      visualSelectedEntityId: "observe",
    });
  });

  test("moves focus through reading order with keyboard navigation", () => {
    const coordinator = createFocusCoordinator(projection);
    coordinator.selectFromVisual("arrive");

    expect(coordinator.focusNext().focusedEntityId).toBe("admit");
    expect(coordinator.focusNext().focusedEntityId).toBe("observe");
    expect(coordinator.focusPrevious().focusedEntityId).toBe("admit");
  });

  test("ignores unknown entity ids without clearing current focus", () => {
    const coordinator = createFocusCoordinator(projection);
    coordinator.selectFromVisual("arrive");

    const ignored = coordinator.selectFromVisual("missing");
    expect(snapshot(ignored)).toEqual({
      focusedEntityId: "arrive",
      selectedEntityId: "arrive",
      visualSelectedEntityId: "arrive",
    });
  });

  test("restores a previously focused entity id", () => {
    const coordinator = createFocusCoordinator(projection);
    coordinator.selectFromVisual("observe");
    coordinator.clear();

    expect(snapshot(coordinator.getState())).toEqual({
      focusedEntityId: null,
      selectedEntityId: null,
      visualSelectedEntityId: null,
    });

    expect(snapshot(coordinator.restore("admit"))).toEqual({
      focusedEntityId: "admit",
      selectedEntityId: "admit",
      visualSelectedEntityId: "admit",
    });
  });
});
