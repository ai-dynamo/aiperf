/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural checks over every declarative deck.
//!
//! These failures are all silent at runtime rather than loud: `Slide` hides any node missing from
//! `revealOrder`, and React Flow drops an edge whose endpoint does not resolve. A deck with either
//! mistake renders as a diagram with a hole in it and no error anywhere.

import { describe, expect, it } from "vitest";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { ASYNC_DATAFLOW_ENGINE_DECK } from "../decks/async-dataflow-engine/deck.js";
import { PYTHON_GRAPH_WORKLOAD_DECK } from "../decks/python-graph-workload/deck.js";
import { METRICS_PLANE_DECK } from "../decks/metrics-plane/deck.js";
import { NATIVE_DIAGRAM_VOCABULARY_DECK } from "../decks/native-diagram-vocabulary/deck.js";
import type { DeckDefinition } from "./types.js";

const DECKS: readonly DeckDefinition[] = [
  ASYNC_DATAFLOW_ENGINE_DECK,
  PYTHON_GRAPH_WORKLOAD_DECK,
  METRICS_PLANE_DECK,
  NATIVE_DIAGRAM_VOCABULARY_DECK,
];

describe.each(DECKS.map((deck) => [deck.id, deck] as const))("deck %s", (_id, deck) => {
  it("reveals every node it declares, and declares every node it reveals", () => {
    for (const slide of deck.slides) {
      if (slide.revealOrder === undefined) continue;
      const nodeIds = new Set(slide.nodes.map((n) => n.id));
      const revealIds = new Set(slide.revealOrder);

      // A node absent from revealOrder stays hidden for the whole slide.
      expect({ slide: slide.id, missing: [...nodeIds].filter((id) => !revealIds.has(id)) })
        .toEqual({ slide: slide.id, missing: [] });
      // A revealOrder id with no node is a typo that silently hides the real node.
      expect({ slide: slide.id, unknown: [...revealIds].filter((id) => !nodeIds.has(id)) })
        .toEqual({ slide: slide.id, unknown: [] });
    }
  });

  it("connects only endpoints that exist", () => {
    for (const slide of deck.slides) {
      const nodeIds = new Set(slide.nodes.map((n) => n.id));
      const dangling = slide.edges
        .filter((e) => !nodeIds.has(e.source) || !nodeIds.has(e.target))
        .map((e) => e.id);
      expect({ slide: slide.id, dangling }).toEqual({ slide: slide.id, dangling: [] });
    }
  });

  it("uses only registered node types, since an unknown type renders as a bare default node", () => {
    for (const slide of deck.slides) {
      const unknown = slide.nodes
        .map((n) => n.type)
        .filter((t) => t === undefined || !(t in nodeTypes));
      expect({ slide: slide.id, unknown }).toEqual({ slide: slide.id, unknown: [] });
    }
  });

  it("gives every slide a unique id, which the deck registry and narration index key on", () => {
    const ids = deck.slides.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
