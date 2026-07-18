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
import { Card, CardBody, CardHeader, Pill } from "./ui";

/**
 * Structured or scene-backed `DeckPackage.finalCard` shape.
 * Scene-only packages use `{ kind: "scene"; scene }`; structured end cards may
 * also carry title / summary / cta chrome around an optional scene.
 */
export type SceneFinalCardRender = Readonly<{
  kind?: "scene";
  title?: string;
  summary?: string;
  cta?: string;
  scene?: SceneIrLike;
}>;

/**
 * Builds a `DeckDefinition.FinalCard` from structured FinalCard fields and/or a
 * scene. Returns `undefined` when nothing renderable is present.
 */
export function finalCardFromScene(
  finalCard: SceneFinalCardRender | undefined,
): (() => ReactNode) | undefined {
  if (finalCard === undefined) {
    return undefined;
  }

  const title = finalCard.title?.trim() || undefined;
  const summary = finalCard.summary?.trim() || undefined;
  const cta = finalCard.cta?.trim() || undefined;
  const scene =
    finalCard.scene ??
    (finalCard.kind === "scene" && "scene" in finalCard
      ? finalCard.scene
      : undefined);

  const hasChrome = title !== undefined || summary !== undefined || cta !== undefined;
  if (scene === undefined && !hasChrome) {
    return undefined;
  }

  return function FinalCard(): ReactNode {
    const reducedMotion = usePrefersReducedMotion();
    const diagram =
      scene !== undefined ? (
        <SceneRenderer
          scene={scene}
          playing
          restartKey={0}
          reducedMotion={reducedMotion}
        />
      ) : null;

    if (!hasChrome) {
      return diagram;
    }

    return (
      <Card>
        <CardHeader trailing={cta !== undefined ? <Pill size="sm">{cta}</Pill> : undefined}>
          {title ?? "Next steps"}
        </CardHeader>
        <CardBody>
          {summary !== undefined ? <p>{summary}</p> : null}
          {diagram}
        </CardBody>
      </Card>
    );
  };
}
