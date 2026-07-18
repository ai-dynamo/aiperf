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

export type MentalModelProps = {
  slideIndex: number;
  slide: SlideDefinition;
  playing?: boolean;
  restartKey?: number;
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
  MentalModel: (props: MentalModelProps) => ReactNode;
  css: string;
  FinalCard?: () => ReactNode;
};

export function slideNarrations(slides: readonly SlideDefinition[]): readonly string[] {
  return slides.map((slide) => slide.narration);
}
