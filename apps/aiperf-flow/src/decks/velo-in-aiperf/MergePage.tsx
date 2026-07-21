/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! M / Merge machine — raw records restore global dispatch order; folded stores append exact
//! algebra and merge approximate t-digests. Ported from the canvas `Merge`: feed four radial
//! inputs into the associative center; at 4/4 the output resolves differently per mode (sorted
//! ordinals for records, exact+approximate reduction for stores).

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
import { MechHeader, NODE_ACTIVE } from "./parts.js";

const FEED_POS: ReadonlyArray<[number, number]> = [
  [200, -180],
  [420, 40],
  [200, 260],
  [-20, 40],
];

export function MergePage(): React.JSX.Element {
  const [mode, setMode] = useState<"records" | "store">("records");
  const [fed, setFed] = useState<boolean[]>([false, false, false, false]);
  const labels = mode === "records" ? ["#8", "#2", "#11", "#4"] : ["Σ c0", "Σ c1", "Σ c2", "Σ c3"];
  const count = fed.filter(Boolean).length;

  const nodes: Node[] = [
    {
      id: "hub",
      type: "chip",
      position: { x: 200, y: 40 },
      data: {
        label: `${count}/4 inputs · ${mode === "records" ? "sort ordinal" : "append_store"}`,
        strokeRole: "primary",
        className: count === 4 ? categoryClassName("cyan") : undefined,
      },
    },
    ...labels.map((label, i): Node => ({
      id: `feed-${i}`,
      type: "panel",
      position: { x: FEED_POS[i]![0], y: FEED_POS[i]![1] },
      data: { title: label, className: fed[i] ? NODE_ACTIVE : undefined },
    })),
  ];

  const edges: Edge[] = labels.map((_, i): Edge => ({
    id: `e-${i}`,
    source: `feed-${i}`,
    target: "hub",
    type: fed[i] ? "flow" : undefined,
    style: fed[i] ? undefined : { strokeDasharray: "2 8", opacity: 0.4 },
  }));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="M / merge machine"
        title="Feed the associative center"
        sentence="Raw records restore global dispatch order; folded stores append exact algebra and merge approximate t-digests."
      />

      <Row gap={12} align="center" wrap>
        <Button
          variant="primary"
          aria-pressed={mode === "store"}
          onClick={() => { setMode((m) => (m === "records" ? "store" : "records")); setFed([false, false, false, false]); }}
        >
          {mode === "records" ? "Switch to folded stores" : "Switch to exact records"}
        </Button>
        <span className={clsx("text-xs font-medium", count === 4 ? categoryClassName("cyan") : inkClassName("tertiary"))}>
          {count === 4
            ? mode === "records"
              ? "output: #2 · #4 · #8 · #11"
              : "output: exact count/sum/extrema · approximate percentiles"
            : "Select radial feeds to complete the reduction."}
        </span>
      </Row>

      <div style={{ height: 440 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          onNodeClick={(_, node) => {
            const m = /^feed-(\d)$/.exec(node.id);
            if (m) setFed((f) => f.map((v, n) => (n === Number(m[1]) ? !v : v)));
          }}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>
    </div>
  );
}
