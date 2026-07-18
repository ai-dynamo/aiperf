/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";

import {
  SceneRenderer,
  type SceneIrLike,
} from "./diagram/SceneRenderer";
import { usePrefersReducedMotion } from "./diagram/usePrefersReducedMotion";

/** Scene-backed `DeckPackage.finalCard` render shape. */
export type SceneFinalCardRender = Readonly<{
  kind: "scene";
  scene: SceneIrLike;
}>;

/**
 * Builds a `DeckDefinition.FinalCard` that mounts `SceneRenderer` when
 * `finalCard.kind === "scene"`. Returns `undefined` when absent or not a scene.
 */
export function finalCardFromScene(
  finalCard: SceneFinalCardRender | undefined,
): (() => ReactNode) | undefined {
  if (finalCard === undefined || finalCard.kind !== "scene") {
    return undefined;
  }

  const { scene } = finalCard;

  return function FinalCard(): ReactNode {
    const reducedMotion = usePrefersReducedMotion();
    return (
      <SceneRenderer
        scene={scene}
        playing
        restartKey={0}
        reducedMotion={reducedMotion}
      />
    );
  };
}
