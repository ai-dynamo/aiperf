/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { useParams } from "react-router-dom";
import { getDeck } from "./registry.js";
import { Slide } from "./Slide.js";

export function DeckRoute(): React.JSX.Element {
  const { deckId } = useParams<{ deckId: string }>();
  const deck = deckId !== undefined ? getDeck(deckId) : undefined;
  const [slideIndex] = useState(0);

  if (deck === undefined) {
    return <div className="p-6">No deck registered for id "{deckId}".</div>;
  }

  const slide = deck.slides[slideIndex];
  if (slide === undefined) {
    return <div className="p-6">Deck "{deck.id}" has no slides.</div>;
  }

  return <Slide slide={slide} />;
}
