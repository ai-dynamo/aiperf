/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useParams } from "react-router-dom";
import { getDeck } from "./registry.js";
import { Slide } from "./Slide.js";
import { PresentationShell } from "../shell/PresentationShell.js";
import { useNarratedDeck } from "../audio/index.js";

export function DeckRoute(): React.JSX.Element {
  const { deckId } = useParams<{ deckId: string }>();
  const deck = deckId !== undefined ? getDeck(deckId) : undefined;
  // Hooks must run unconditionally, so narrate an empty deck rather than
  // returning early above this call.
  const narrated = useNarratedDeck({
    narrations: (deck?.slides ?? []).map((slide) => slide.narration),
    storagePrefix: `deck:${deckId ?? "unknown"}`,
  });

  if (deck === undefined) {
    return <div className="p-6">No deck registered for id "{deckId}".</div>;
  }

  const slide = deck.slides[narrated.index];
  if (slide === undefined) {
    return <div className="p-6">Deck "{deck.id}" has no slides.</div>;
  }

  return (
    <PresentationShell
      slides={deck.slides}
      slideIndex={narrated.index}
      onSlideIndexChange={narrated.goTo}
      narrated={narrated}
      title={deck.title}
    >
      {/* Restarting the slide on `restartKey` keeps the reveal cascade in step
          with the narration when a slide is revisited. */}
      <Slide key={`${slide.id}-${narrated.restartKey}`} slide={slide} />
    </PresentationShell>
  );
}
