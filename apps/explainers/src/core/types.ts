/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";

export type SlideDefinition = {
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: readonly string[];
  caption: string;
};

export type DeckHubMeta = {
  title: string;
  highlight: string;
  description: string;
};

/**
 * Props ExplainerShell passes into the deck diagram slot.
 * Package-backed decks forward playing/restartKey/reducedMotion/playbackRate to SceneRenderer.
 */
export type MentalModelProps = {
  slideIndex: number;
  slide: SlideDefinition;
  playing?: boolean;
  restartKey?: number;
  reducedMotion?: boolean;
  /** Wall-clock multiplier for scene timelines (1 = realtime). */
  playbackRate?: number;
};

export type DeckDefinition = {
  id: string;
  route: string;
  storagePrefix: string;
  classPrefix: string;
  eyebrowLabel: string;
  startGateTitle: string;
  hub: DeckHubMeta;
  slides: readonly SlideDefinition[];
  glossary: readonly { word: string; meaning: string }[];
  MentalModel: (props: MentalModelProps) => ReactNode;
  css: string;
  /** Optional end card; package decks omit this unless DeckPackage.finalCard is set. */
  FinalCard?: () => ReactNode;
};

export function slideNarrations(slides: readonly SlideDefinition[]): readonly string[] {
  return slides.map((slide) => slide.narration);
}
