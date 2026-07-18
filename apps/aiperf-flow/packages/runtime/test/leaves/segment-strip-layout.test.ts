// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { layoutSegmentStrip } from "../../src/leaves/segment-strip-layout.js";

describe("leaf.core.segment-strip.layout", () => {
  test("lays out PromptSegmentComposer fixture deterministically", () => {
    const layout = layoutSegmentStrip(
      [
        { id: "seg-system", tokens: 12, role: "system" },
        { id: "seg-prefix", tokens: 48, role: "prefix", reused: true },
        { id: "seg-user", tokens: 24, role: "user" },
        { id: "seg-image", tokens: 0, role: "image" },
        { id: "seg-tool", tokens: 18, role: "tool" },
        { id: "seg-assistant", tokens: 16, role: "assistant" },
        { id: "seg-tail", tokens: 8, role: "tail", truncated: true },
      ],
      {
        originX: 0,
        originY: 0,
        rowHeight: 24,
        gap: 4,
        unitWidth: 2,
        seed: 42,
      },
    );

    expect(layout.version).toBe(1);
    expect(layout.routes).toEqual([]);

    const first = layout.nodes[0];
    expect(first?.nodeId).toBe("seg-system");
    expect(first?.bounds).toEqual({ x: 0, y: 0, width: 24, height: 24 });

    const prefix = layout.nodes.find((node) => node.nodeId === "seg-prefix");
    expect(prefix?.continuation).toBe(true);
    expect(prefix?.clip).toBeUndefined();

    const tail = layout.nodes.find((node) => node.nodeId === "seg-tail");
    expect(tail?.clip).toBe(true);
    expect(tail?.continuation).toBeUndefined();

    const totalWidth =
      (layout.nodes.at(-1)?.bounds.x ?? 0) + (layout.nodes.at(-1)?.bounds.width ?? 0);
    expect(totalWidth).toBe(278);
  });
});
