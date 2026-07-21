/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! T / Aggregator tree — folded stores reduce through aggregators; retained raw records stay
//! flat so the controller can restore global dispatch order. Ported from the canvas `Tree`:
//! toggle flat vs. tree topology and adjust payload volume; eight cell stores either fan
//! straight into the controller (flat) or reduce through two subtree aggregators (tree).

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { categoryClassName, inkClassName } from "../../theme/tokens.js";
import { MechHeader } from "./parts.js";

export function TreePage(): React.JSX.Element {
  const [shape, setShape] = useState<"flat" | "tree">("tree");
  const [payload, setPayload] = useState(64);
  const safePayload = Math.min(96, Math.max(8, payload));
  const cells = [0, 1, 2, 3, 4, 5, 6, 7];

  const nodes: Node[] = [
    {
      id: "controller",
      type: "card",
      position: { x: 300, y: 0 },
      data: {
        title: "controller",
        subtitle: shape === "tree" ? "2 stores in" : "8 partitions in",
        className: categoryClassName("cyan"),
      },
    },
    ...(shape === "tree"
      ? [
          { id: "agg-0", type: "panel", position: { x: 120, y: 180 }, data: { title: "aggregator 0", detail: "cells 0–3" } } as Node,
          { id: "agg-1", type: "panel", position: { x: 500, y: 180 }, data: { title: "aggregator 1", detail: "cells 4–7" } } as Node,
        ]
      : []),
    ...cells.map((c): Node => ({
      id: `cell-${c}`,
      type: "chip",
      position: { x: c * 95, y: 380 },
      data: { label: `c${c} · ${safePayload}u`, strokeRole: "secondary" },
    })),
  ];

  const edges: Edge[] = [
    ...(shape === "tree"
      ? [
          { id: "e-agg0", source: "agg-0", target: "controller", type: "flow" } as Edge,
          { id: "e-agg1", source: "agg-1", target: "controller", type: "flow" } as Edge,
          ...cells.map((c): Edge => ({
            id: `e-cell-${c}`,
            source: `cell-${c}`,
            target: c < 4 ? "agg-0" : "agg-1",
          })),
        ]
      : cells.map((c): Edge => ({ id: `e-cell-${c}`, source: `cell-${c}`, target: "controller", type: "flow" }))),
  ];

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="T / aggregator tree"
        title="Collapse payload upward"
        sentence="Folded stores can reduce through aggregators; retained raw records stay flat so the controller can restore global dispatch order."
      />

      <Row gap={12} align="center" wrap>
        <Button variant={shape === "flat" ? "primary" : "secondary"} aria-pressed={shape === "flat"} onClick={() => setShape("flat")}>
          Flat records
        </Button>
        <Button variant={shape === "tree" ? "primary" : "secondary"} aria-pressed={shape === "tree"} onClick={() => setShape("tree")}>
          Folded tree
        </Button>
        <label className={clsx("flex items-center gap-2 text-xs font-bold uppercase", inkClassName("tertiary"))}>
          payload volume
          <input
            aria-label="Payload volume"
            type="range"
            min={8}
            max={96}
            value={safePayload}
            onChange={(e) => setPayload(Number(e.target.value))}
          />
          <span className="font-mono">{safePayload}u</span>
        </label>
      </Row>

      <div style={{ height: 480 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      <p className={clsx("text-xs", categoryClassName("cyan"))}>
        {shape === "tree"
          ? "8 cell stores → 2 subtree stores → 1 report"
          : "8 raw partitions → controller global-order merge"}
      </p>
    </div>
  );
}
