import { describe, expect, it } from "vitest";
import { validateDeckRegistry, DECK_REGISTRY } from "../core/deck-registry";

const EXPECTED_DECKS = new Map([
  ["rust-architecture", 16],
  ["slurm-velo", 16],
  ["dynosim", 18],
  ["rust-architecture-atlas", 11],
  ["velo-deep-dive", 10],
  ["cellular-internals", 20],
  ["cellular-algorithms", 16],
  ["dynosim-offline", 7],
  ["segment-pools", 6],
  ["mock-server", 10],
]);

describe("deck registry", () => {
  it("has the expected decks and slide counts", () => {
    expect(validateDeckRegistry()).toEqual([]);
    expect(new Map(DECK_REGISTRY.map((deck) => [deck.id, deck.slides.length]))).toEqual(
      EXPECTED_DECKS,
    );
  });

  it("has unique IDs and routes", () => {
    expect(new Set(DECK_REGISTRY.map((deck) => deck.id)).size).toBe(DECK_REGISTRY.length);
    expect(new Set(DECK_REGISTRY.map((deck) => deck.route)).size).toBe(DECK_REGISTRY.length);
  });

  it("has complete display and narration content on every slide", () => {
    for (const deck of DECK_REGISTRY) {
      expect(deck.hub.title.trim()).not.toBe("");
      for (const slide of deck.slides) {
        expect(slide.title.trim()).not.toBe("");
        expect(slide.narration.trim()).not.toBe("");
        expect(slide.caption.trim()).not.toBe("");
        expect(slide.points.length).toBeGreaterThan(0);
        expect(slide.points.every((point) => point.trim().length > 0)).toBe(true);
      }
    }
  });

  it("preserves legacy storage prefixes for rust and slurm", () => {
    expect(DECK_REGISTRY.find((d) => d.id === "rust-architecture")?.storagePrefix).toBe(
      "rust-arch-explainer",
    );
    expect(DECK_REGISTRY.find((d) => d.id === "slurm-velo")?.storagePrefix).toBe(
      "slurm-explainer",
    );
  });
});
