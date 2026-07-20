/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import { SceneRenderer } from "./diagram/SceneRenderer";
import type { SceneIrLike } from "./diagram/scene-types";
import {
  finalCardFromScene,
  hasRenderableFinalCard,
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

function trimField(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

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
 * Resolve `DeckPackage.finalCard` for ExplainerShell.
 *
 * Canonical form is `{ kind: "scene"; scene }` — chrome lives in the scene IR,
 * so scene `title` / `summary` are not copied into Card chrome (that would
 * double-render headers already painted by SceneRenderer). Explicit card-level
 * `title` / `summary` / `cta` are preserved when authored.
 *
 * No last-slide fallback: reusing the last slide scene would duplicate the
 * diagram under the slideshow on the final slide.
 */
export function resolveFinalCardScene(
  pkg: DeckPackage,
): FinalCardPackage | undefined {
  const card = pkg.finalCard;
  if (card === undefined) {
    return undefined;
  }

  const scene = card.scene;
  const title = trimField(card.title);
  const summary = trimField(card.summary);
  const cta = trimField(card.cta);

  const resolved: FinalCardPackage = {
    kind: "scene",
    ...(title !== undefined ? { title } : {}),
    ...(summary !== undefined ? { summary } : {}),
    ...(cta !== undefined ? { cta } : {}),
    ...(scene !== undefined ? { scene } : {}),
  };

  return hasRenderableFinalCard(resolved) ? resolved : undefined;
}

function PackageMentalModel({
  pkg,
  slideIndex,
  playing = false,
  restartKey = 0,
  reducedMotion = false,
  playbackRate = 1,
}: {
  pkg: DeckPackage;
} & Pick<
  MentalModelProps,
  "slideIndex" | "playing" | "restartKey" | "reducedMotion" | "playbackRate"
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
      playbackRate={playbackRate}
    />
  );
}

/**
 * Maps a flow-backed DeckPackage onto the legacy DeckDefinition consumed by
 * ExplainerShell. MentalModel mounts SceneRenderer from each slide's scene IR,
 * forwarding playing/restartKey/reducedMotion/playbackRate from ExplainerShell so timelines
 * animate with the slideshow. FinalCard mounts SceneRenderer (and optional Card
 * chrome) only when `pkg.finalCard` is present — never a cloned last slide.
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
    glossary: pkg.glossary ?? [],
    css: pkg.css ?? "",
    MentalModel: ({ slideIndex, playing, restartKey, reducedMotion, playbackRate }) => (
      <PackageMentalModel
        pkg={pkg}
        slideIndex={slideIndex}
        playing={playing}
        restartKey={restartKey}
        reducedMotion={reducedMotion}
        playbackRate={playbackRate}
      />
    ),
    ...(FinalCard !== undefined ? { FinalCard } : {}),
  };
}
