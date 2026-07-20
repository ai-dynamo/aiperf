/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";

import { SceneRenderer } from "./diagram/SceneRenderer";
import type { SceneIrLike } from "./diagram/scene-types";
import type { FinalCardProps } from "./types";
import { Card, CardBody, CardHeader, Pill, Stack } from "./ui";

/**
 * Structured or scene-backed `DeckPackage.finalCard` shape.
 *
 * Canonical packages emit `{ kind: "scene"; scene }` with chrome baked into the
 * scene IR. Optional `title` / `summary` / `cta` wrap a Card chrome only when
 * those fields are authored explicitly on the card (not copied from scene IR).
 */
export type SceneFinalCardRender = Readonly<{
  kind?: "scene";
  title?: string;
  summary?: string;
  cta?: string;
  scene?: SceneIrLike;
}>;

function normalizeText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

/** True when the payload has anything ExplainerShell can mount. */
export function hasRenderableFinalCard(
  finalCard: SceneFinalCardRender | undefined,
): finalCard is SceneFinalCardRender {
  if (finalCard === undefined) {
    return false;
  }
  return (
    finalCard.scene !== undefined ||
    normalizeText(finalCard.title) !== undefined ||
    normalizeText(finalCard.summary) !== undefined ||
    normalizeText(finalCard.cta) !== undefined
  );
}

/**
 * Builds a `DeckDefinition.FinalCard` from structured FinalCard fields and/or a
 * scene. Returns `undefined` when nothing renderable is present.
 *
 * Scene-only cards mount `SceneRenderer` directly (scene IR owns the chrome).
 * Explicit title/summary/cta add Card chrome around an optional scene.
 */
export function finalCardFromScene(
  finalCard: SceneFinalCardRender | undefined,
): ((props: FinalCardProps) => ReactNode) | undefined {
  if (!hasRenderableFinalCard(finalCard)) {
    return undefined;
  }

  const title = normalizeText(finalCard.title);
  const summary = normalizeText(finalCard.summary);
  const cta = normalizeText(finalCard.cta);
  const scene = finalCard.scene;
  const hasChrome =
    title !== undefined || summary !== undefined || cta !== undefined;

  if (scene === undefined && !hasChrome) {
    return undefined;
  }

  return function FinalCard({
    playing = true,
    restartKey = 0,
    reducedMotion = false,
    playbackRate = 1,
  }: FinalCardProps): ReactNode {
    const diagram =
      scene !== undefined ? (
        <SceneRenderer
          scene={scene}
          playing={playing}
          restartKey={restartKey}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
        />
      ) : null;

    // Canonical package form: scene IR already paints header / paths / pill.
    if (!hasChrome) {
      return (
        <div
          className="explainer-final-card explainer-final-card--scene"
          role="region"
          aria-label={
            scene?.accessibility?.label?.trim() ||
            scene?.title?.trim() ||
            "Next steps"
          }
        >
          {diagram}
        </div>
      );
    }

    return (
      <div className="explainer-final-card explainer-final-card--chrome">
        <Card>
          <CardHeader
            trailing={
              cta !== undefined ? <Pill size="sm">{cta}</Pill> : undefined
            }
          >
            {title ?? "Next steps"}
          </CardHeader>
          <CardBody>
            <Stack gap={12}>
              {summary !== undefined ? <p style={{ margin: 0 }}>{summary}</p> : null}
              {diagram}
            </Stack>
          </CardBody>
        </Card>
      </div>
    );
  };
}
