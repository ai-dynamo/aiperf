// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type SegmentStripInput = Readonly<{
  id: string;
  tokens: number;
  role: string;
  reused?: boolean;
  truncated?: boolean;
}>;

export type SegmentStripLayoutOptions = Readonly<{
  originX: number;
  originY: number;
  rowHeight: number;
  gap: number;
  unitWidth: number;
  seed: number;
}>;

export type SegmentStripLayoutResult = Readonly<{
  version: 1;
  nodes: readonly Readonly<{
    nodeId: string;
    bounds: { x: number; y: number; width: number; height: number };
    clip?: boolean;
    continuation?: boolean;
  }>[];
  routes: readonly [];
}>;

/** Lays out prompt segments left-to-right as a deterministic strip plan. */
export function layoutSegmentStrip(
  segments: readonly SegmentStripInput[],
  options: SegmentStripLayoutOptions,
): SegmentStripLayoutResult {
  void options.seed;

  const nodes: SegmentStripLayoutResult["nodes"][number][] = [];
  let cursorX = options.originX;

  for (const segment of segments) {
    const width = Math.max(segment.tokens, 1) * options.unitWidth;
    nodes.push({
      nodeId: segment.id,
      bounds: {
        x: cursorX,
        y: options.originY,
        width,
        height: options.rowHeight,
      },
      ...(segment.truncated === true ? { clip: true } : {}),
      ...(segment.reused === true ? { continuation: true } : {}),
    });
    cursorX += width + options.gap;
  }

  return { version: 1, nodes, routes: [] };
}
