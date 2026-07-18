// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Visibility level for immersive runtime controls. */
export type HudVisibility = "present" | "quiet" | "hidden";

/** Fullscreen presentation mode owned by the immersive runtime. */
export type FullscreenState = "windowed" | "layout" | "native";

/** Serializable interaction state for the immersive runtime surface. */
export type ImmersiveState = Readonly<{
  selectedEntityId: string | null;
  contextLensOpen: boolean;
  focusWorldEntityId: string | null;
  comparisonEntityId: string | null;
  commandOpen: boolean;
  hud: HudVisibility;
  fullscreen: FullscreenState;
}>;

/** Exhaustive interaction vocabulary for immersive runtime state. */
export type ImmersiveAction =
  | Readonly<{ type: "replace"; state: ImmersiveState }>
  | Readonly<{ type: "select"; entityId: string | null }>
  | Readonly<{ type: "open-context"; entityId: string }>
  | Readonly<{ type: "close-context" }>
  | Readonly<{ type: "enter-focus-world"; entityId: string }>
  | Readonly<{ type: "leave-focus-world" }>
  | Readonly<{ type: "open-command" }>
  | Readonly<{ type: "close-command" }>
  | Readonly<{ type: "set-hud"; visibility: HudVisibility }>
  | Readonly<{ type: "set-fullscreen"; state: FullscreenState }>;

function immutableState(state: ImmersiveState): ImmersiveState {
  return Object.freeze(state);
}

/** Creates the default serializable immersive interaction state. */
export function createImmersiveState(): ImmersiveState {
  return immutableState({
    selectedEntityId: null,
    contextLensOpen: false,
    focusWorldEntityId: null,
    comparisonEntityId: null,
    commandOpen: false,
    hud: "present",
    fullscreen: "windowed",
  });
}

/** Applies one immutable immersive interaction transition. */
export function immersiveReducer(
  state: ImmersiveState,
  action: ImmersiveAction,
): ImmersiveState {
  switch (action.type) {
    case "replace":
      return immutableState({ ...action.state });
    case "select":
      return immutableState({ ...state, selectedEntityId: action.entityId });
    case "open-context":
      return immutableState({
        ...state,
        selectedEntityId: action.entityId,
        contextLensOpen: true,
      });
    case "close-context":
      return immutableState({ ...state, contextLensOpen: false });
    case "enter-focus-world":
      return immutableState({
        ...state,
        selectedEntityId: action.entityId,
        focusWorldEntityId: action.entityId,
        comparisonEntityId: state.selectedEntityId,
      });
    case "leave-focus-world":
      return immutableState({
        ...state,
        selectedEntityId: state.comparisonEntityId,
        focusWorldEntityId: null,
        comparisonEntityId: null,
      });
    case "open-command":
      return immutableState({ ...state, commandOpen: true });
    case "close-command":
      return immutableState({ ...state, commandOpen: false });
    case "set-hud":
      return immutableState({ ...state, hud: action.visibility });
    case "set-fullscreen":
      return immutableState({ ...state, fullscreen: action.state });
    default: {
      const unhandledAction: never = action;
      return unhandledAction;
    }
  }
}
