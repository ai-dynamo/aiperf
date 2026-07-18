/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import {
  SceneRenderer,
  type SceneIrLike,
} from "./diagram/SceneRenderer";
import {
  finalCardFromScene,
  type SceneFinalCardRender,
} from "./final-card-from-scene";
import type {
  DeckDefinition,
  DeckHubMeta,
  MentalModelProps,
  SlideDefinition,
} from "./types";

/** Scene render payload attached to a packaged slide or final card. */
export type SceneRenderPackage = Readonly<{
  kind: "scene";
  scene: SceneIrLike;
  title?: string;
  summary?: string;
  cta?: string;
}>;

/** Structured FinalCard fields from a DeckPackage (scene and/or chrome). */
export type FinalCardPackage = SceneFinalCardRender;

/** One slide from a flow-backed DeckPackage. */
export type SlidePackage = Readonly<{
  id: string;
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: readonly string[];
  caption: string;
  render?: SceneRenderPackage;
}>;

/** Hub fields as authored on DeckPackage (canonical + common aliases). */
export type DeckPackageHub = Readonly<{
  title?: string;
  highlight?: string;
  description?: string;
  summary?: string;
  cta?: string;
}>;

/** Compiled DeckPackage shape consumed by the legacy ExplainerShell adapter. */
export type DeckPackage = Readonly<{
  schemaVersion: 1;
  id: string;
  route: string;
  topic: string;
  storagePrefix: string;
  classPrefix: string;
  eyebrowLabel: string;
  startGateTitle: string;
  hub: DeckPackageHub | DeckHubMeta;
  css?: string;
  finalCard?: FinalCardPackage | SceneRenderPackage;
  slides: readonly SlidePackage[];
  glossary: readonly { word: string; meaning: string }[];
}>;

function toSlideDefinition(slide: SlidePackage): SlideDefinition {
  return {
    eyebrow: slide.eyebrow,
    title: slide.title,
    lede: slide.lede,
    narration: slide.narration,
    ...(slide.term !== undefined ? { term: slide.term } : {}),
    points: slide.points,
    caption: slide.caption,
  };
}

/** Maps DeckPackage.hub onto DeckDefinition.hub (title / highlight / description). */
export function resolveHubMeta(hub: DeckPackage["hub"]): DeckHubMeta {
  const title = hub.title?.trim() || "";
  const highlight = hub.highlight?.trim() || "";
  const description =
    ("description" in hub && hub.description?.trim()) ||
    ("summary" in hub && hub.summary?.trim()) ||
    ("cta" in hub && hub.cta?.trim()) ||
    "";
  return { title, highlight, description };
}

/**
 * Prefer an explicit `DeckPackage.finalCard` (scene and/or title/summary/cta);
 * otherwise reuse the last slide's scene so ExplainerShell can mount FinalCard
 * without MentalModel.
 */
export function resolveFinalCardScene(
  pkg: DeckPackage,
): FinalCardPackage | undefined {
  const card = pkg.finalCard;
  if (card !== undefined) {
    const scene =
      card.scene ??
      (card.kind === "scene" && "scene" in card ? card.scene : undefined);
    const title =
      card.title?.trim() ||
      (scene && "title" in scene && typeof scene.title === "string"
        ? scene.title
        : undefined);
    const summary =
      card.summary?.trim() ||
      (scene && "summary" in scene && typeof scene.summary === "string"
        ? scene.summary
        : undefined);
    const cta = card.cta?.trim() || undefined;
    if (
      scene !== undefined ||
      title !== undefined ||
      summary !== undefined ||
      cta !== undefined
    ) {
      return {
        ...(card.kind !== undefined ? { kind: card.kind } : { kind: "scene" as const }),
        ...(title !== undefined ? { title } : {}),
        ...(summary !== undefined ? { summary } : {}),
        ...(cta !== undefined ? { cta } : {}),
        ...(scene !== undefined ? { scene } : {}),
      };
    }
  }

  const last = pkg.slides[pkg.slides.length - 1];
  if (last?.render?.kind === "scene" && last.render.scene !== undefined) {
    return {
      kind: "scene",
      scene: last.render.scene,
      ...(last.render.title !== undefined ? { title: last.render.title } : {}),
      ...(last.render.summary !== undefined
        ? { summary: last.render.summary }
        : {}),
      ...(last.render.cta !== undefined ? { cta: last.render.cta } : {}),
    };
  }
  return undefined;
}

function PackageMentalModel({
  pkg,
  slideIndex,
  playing = false,
  restartKey = 0,
  reducedMotion = false,
}: {
  pkg: DeckPackage;
} & Pick<
  MentalModelProps,
  "slideIndex" | "playing" | "restartKey" | "reducedMotion"
>): ReactNode {
  const scene = pkg.slides[slideIndex]?.render?.scene;
  if (scene === undefined) {
    return null;
  }
  return (
    <SceneRenderer
      scene={scene}
      playing={playing}
      restartKey={restartKey}
      reducedMotion={reducedMotion}
    />
  );
}

/**
 * Maps a flow-backed DeckPackage onto the legacy DeckDefinition consumed by
 * ExplainerShell. MentalModel mounts SceneRenderer from each slide's scene IR,
 * forwarding playing/restartKey/reducedMotion from ExplainerShell so timelines
 * animate with the slideshow. FinalCard mounts title/summary/cta chrome and/or
 * SceneRenderer from finalCard (or the last slide's scene).
 */
export function packageToDeckDefinition(pkg: DeckPackage): DeckDefinition {
  const slides = pkg.slides.map(toSlideDefinition);
  const FinalCard = finalCardFromScene(resolveFinalCardScene(pkg));
  return {
    id: pkg.id,
    route: pkg.route,
    storagePrefix: pkg.storagePrefix,
    classPrefix: pkg.classPrefix,
    eyebrowLabel: pkg.eyebrowLabel,
    startGateTitle: pkg.startGateTitle,
    hub: resolveHubMeta(pkg.hub),
    slides,
    css: pkg.css ?? "",
    MentalModel: ({ slideIndex, playing, restartKey, reducedMotion }) => (
      <PackageMentalModel
        pkg={pkg}
        slideIndex={slideIndex}
        playing={playing}
        restartKey={restartKey}
        reducedMotion={reducedMotion}
      />
    ),
    ...(FinalCard !== undefined ? { FinalCard } : {}),
  };
}
