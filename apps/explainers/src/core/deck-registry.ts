import type { DeckDefinition } from "./types";
import { cellularAlgorithmsDeck } from "../decks/cellular-algorithms";
import { cellularInternalsDeck } from "../decks/cellular-internals";
import { dynosimDeck } from "../decks/dynosim";
import { rustArchitectureDeck } from "../decks/rust-architecture";
import { rustArchitectureAtlasDeck } from "../decks/rust-architecture-atlas";
import { segmentPoolsDeck } from "../decks/segment-pools";
import { slurmVeloDeck } from "../decks/slurm-velo";
import { veloDeepDiveDeck } from "../decks/velo-deep-dive";

export const DECK_REGISTRY: readonly DeckDefinition[] = [
  rustArchitectureDeck,
  rustArchitectureAtlasDeck,
  segmentPoolsDeck,
  slurmVeloDeck,
  veloDeepDiveDeck,
  cellularInternalsDeck,
  cellularAlgorithmsDeck,
  dynosimDeck,
];

export function deckByRoute(route: string): DeckDefinition | undefined {
  return DECK_REGISTRY.find((deck) => deck.route === route);
}

export function validateDeckRegistry(decks: readonly DeckDefinition[] = DECK_REGISTRY): string[] {
  const errors: string[] = [];
  const routes = new Set<string>();
  const ids = new Set<string>();

  for (const deck of decks) {
    if (routes.has(deck.route)) errors.push(`duplicate route: ${deck.route}`);
    routes.add(deck.route);
    if (ids.has(deck.id)) errors.push(`duplicate id: ${deck.id}`);
    ids.add(deck.id);
    if (deck.slides.length === 0) errors.push(`${deck.id}: no slides`);
    deck.slides.forEach((slide, index) => {
      if (!slide.narration.trim()) errors.push(`${deck.id}: slide ${index + 1} missing narration`);
      if (!slide.title.trim()) errors.push(`${deck.id}: slide ${index + 1} missing title`);
    });
  }

  return errors;
}
