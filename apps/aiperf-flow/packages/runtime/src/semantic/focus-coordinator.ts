// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SemanticProjection } from "../evaluate/types.js";

export type FocusCoordinatorState = Readonly<{
  focusedEntityId: string | null;
  selectedEntityId: string | null;
  visualSelectedEntityId: string | null;
}>;

export type FocusCoordinator = Readonly<{
  getState(): FocusCoordinatorState;
  selectFromVisual(entityId: string): FocusCoordinatorState;
  activateFromSemantic(entityId: string): FocusCoordinatorState;
  focusNext(): FocusCoordinatorState;
  focusPrevious(): FocusCoordinatorState;
  restore(entityId: string): FocusCoordinatorState;
  clear(): FocusCoordinatorState;
}>;

function initialState(): FocusCoordinatorState {
  return {
    focusedEntityId: null,
    selectedEntityId: null,
    visualSelectedEntityId: null,
  };
}

function knownEntityIds(projection: SemanticProjection): ReadonlySet<string> {
  return new Set(projection.entities.map(({ id }) => id));
}

function readingOrderIndex(
  projection: SemanticProjection,
  entityId: string | null,
): number {
  if (entityId === null) {
    return -1;
  }
  return projection.readingOrder.indexOf(entityId);
}

/**
 * Synchronizes visual hit selection with semantic twin focus and keyboard
 * activation. Entity identity is authoritative; unknown ids are ignored.
 */
export function createFocusCoordinator(
  projection: SemanticProjection,
): FocusCoordinator {
  const known = knownEntityIds(projection);
  let state = initialState();

  function commit(next: FocusCoordinatorState): FocusCoordinatorState {
    state = next;
    return state;
  }

  function selectKnown(entityId: string): FocusCoordinatorState {
    if (!known.has(entityId)) {
      return state;
    }
    return commit({
      focusedEntityId: entityId,
      selectedEntityId: entityId,
      visualSelectedEntityId: entityId,
    });
  }

  function moveFocus(delta: 1 | -1): FocusCoordinatorState {
    const order = projection.readingOrder.filter((id) => known.has(id));
    if (order.length === 0) {
      return state;
    }

    const current = readingOrderIndex(projection, state.focusedEntityId);
    const nextIndex =
      current < 0
        ? delta > 0
          ? 0
          : order.length - 1
        : (current + delta + order.length) % order.length;
    const entityId = order[nextIndex];
    if (entityId === undefined) {
      return state;
    }
    return selectKnown(entityId);
  }

  return {
    getState: () => state,
    selectFromVisual: selectKnown,
    activateFromSemantic: selectKnown,
    focusNext: () => moveFocus(1),
    focusPrevious: () => moveFocus(-1),
    restore: selectKnown,
    clear: () => commit(initialState()),
  };
}
