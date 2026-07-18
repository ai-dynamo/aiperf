// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type WaterfallEvent = Readonly<{
  id: string;
  lane: string;
  start: number;
  end: number;
}>;

export type WaterfallLayoutOptions = Readonly<{
  laneOrder: readonly string[];
  originX: number;
  originY: number;
  laneHeight: number;
  laneGap: number;
  pxPerMs: number;
}>;

export type WaterfallLayoutResult = Readonly<{
  version: 1;
  nodes: readonly Readonly<{
    nodeId: string;
    bounds: { x: number; y: number; width: number; height: number };
  }>[];
  routes: readonly [];
}>;

/** Lays out nested waterfall intervals and point events across ordered lanes. */
export function layoutWaterfallNest(
  events: readonly WaterfallEvent[],
  options: WaterfallLayoutOptions,
): WaterfallLayoutResult {
  const laneIndex = new Map(options.laneOrder.map((lane, index) => [lane, index]));

  const nodes = events.map((event) => {
    const index = laneIndex.get(event.lane);
    if (index === undefined) {
      throw new Error(`Unknown waterfall lane "${event.lane}".`);
    }

    const durationMs = Math.max(event.end - event.start, 0);
    const width =
      event.start === event.end
        ? Math.max(1, options.pxPerMs)
        : durationMs * options.pxPerMs;

    return {
      nodeId: event.id,
      bounds: {
        x: options.originX + event.start * options.pxPerMs,
        y: options.originY + index * (options.laneHeight + options.laneGap),
        width,
        height: options.laneHeight,
      },
    };
  });

  return { version: 1, nodes, routes: [] };
}
