/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared diagram vocabulary for narrated decks.
//!
//! The engine's whole subject is fan-out and fan-in — successors spawning, AND-joins on
//! channel counts, traces spreading across workers. These helpers place nodes on an
//! explicit grid so a slide can draw that shape, rather than flattening every idea into
//! a left-to-right chain.

import type { Edge, Node } from "@xyflow/react";

/** Grid step. One column clears a card's max width; one row clears its height. */
export const COL = 300;
export const ROW = 118;

/**
 * Semantic colour, shared by nodes and the edges between them so a hop and the boxes it
 * joins agree. Mirrors the categories in `theme/tokens.ts`.
 */
export type Tone = "control" | "data" | "time" | "failure" | "done" | "muted";

export const TONE_STROKE: Record<Tone, string> = {
  control: "var(--color-accent-primary)",
  data: "var(--color-category-purple)",
  time: "var(--color-category-yellow)",
  failure: "var(--color-category-red)",
  done: "var(--color-category-green)",
  muted: "var(--color-category-gray)",
};

// Tailwind only sees class strings it can read literally, so each tone's arbitrary
// property is spelled out rather than interpolated.
const TONE_TINT: Record<Tone, string> = {
  control: "[--accent:var(--color-accent-primary)]",
  data: "[--accent:var(--color-category-purple)]",
  time: "[--accent:var(--color-category-yellow)]",
  failure: "[--accent:var(--color-category-red)]",
  done: "[--accent:var(--color-category-green)]",
  muted: "[--accent:var(--color-category-gray)]",
};

interface Placed {
  /** Grid column; fractional values are allowed for centring a fan's apex. */
  col: number;
  /** Grid row. */
  row: number;
}

/** A titled card, the workhorse node. */
export function card(
  id: string,
  title: string,
  subtitle: string | undefined,
  tone: Tone,
  { col, row }: Placed,
): Node {
  return {
    id,
    type: "card",
    position: { x: col * COL, y: row * ROW },
    data: { title, subtitle, className: TONE_TINT[tone] },
  };
}

/** Small junction label — a gate, a combiner, a count. Reads as an operator, not a stage. */
export function chip(id: string, label: string, { col, row }: Placed): Node {
  return {
    id,
    type: "chip",
    // Nudged toward the row's vertical centre, since a chip is far shorter than a card.
    position: { x: col * COL, y: row * ROW + 26 },
    data: { label },
  };
}

/** Aside panel carrying a claim the diagram itself cannot state. */
export function note(id: string, title: string, detail: string, { col, row }: Placed): Node {
  return {
    id,
    type: "panel",
    position: { x: col * COL, y: row * ROW },
    data: { title, detail },
  };
}

/** Band caption above a group of nodes. */
export function band(id: string, title: string, { col, row }: Placed): Node {
  return {
    id,
    type: "header",
    position: { x: col * COL, y: row * ROW - 34 },
    data: { title },
  };
}

/** Directed connector. Ids derive from the pair; no slide connects a pair twice. */
export function link(
  source: string,
  target: string,
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    type: "flow",
    data: { color: TONE_STROKE[tone], speed },
  };
}

/** One source branching to many targets. */
export function fanOut(
  source: string,
  targets: readonly string[],
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge[] {
  return targets.map((target) => link(source, target, tone, speed));
}

/** Many sources converging on one target — an AND-join in engine terms. */
export function fanIn(
  sources: readonly string[],
  target: string,
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge[] {
  return sources.map((source) => link(source, target, tone, speed));
}

/** Vertically centred row for a fan of `count` items starting at `firstRow`. */
export function apexRow(firstRow: number, count: number): number {
  return firstRow + (count - 1) / 2;
}
