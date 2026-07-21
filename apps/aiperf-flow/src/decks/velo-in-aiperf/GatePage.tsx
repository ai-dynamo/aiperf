/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! G / Start gate — asynchronous cell arrival with one barrier release. Ported from the canvas
//! `Gate`: click each of four cells to register; once all four have arrived the controller
//! barrier can fire START, waking every awaiting cell. Cells are React Flow nodes that advance
//! toward the gate as they arrive.

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

export function GatePage(): React.JSX.Element {
  const [arrivals, setArrivals] = useState<boolean[]>([false, false, false, false]);
  const [released, setReleased] = useState(false);
  const all = arrivals.every(Boolean);
  const count = arrivals.filter(Boolean).length;

  const arrive = (i: number) =>
    setArrivals((a) => a.map((v, n) => (n === i ? true : v)));

  const nodes: Node[] = [
    ...[0, 1, 2, 3].map((i): Node => ({
      id: `cell-${i}`,
      type: "card",
      position: { x: i * 180, y: arrivals[i] ? 0 : 240 },
      data: {
        title: `c${i}`,
        subtitle: arrivals[i] ? "registered" : "awaiting",
        className: arrivals[i] ? NODE_ACTIVE : undefined,
      },
    })),
    {
      id: "gate",
      type: "header",
      position: { x: 250, y: -120 },
      data: {
        title: "controller barrier",
        caption: all ? "released → START" : `${count}/4 registered`,
        surfaceRole: "elevated",
        className: all ? categoryClassName("cyan") : undefined,
      },
    },
  ];

  const edges: Edge[] = [0, 1, 2, 3].map((i): Edge => ({
    id: `e-${i}`,
    source: `cell-${i}`,
    target: "gate",
    type: released && arrivals[i] ? "flow" : undefined,
    style: arrivals[i] ? undefined : { strokeDasharray: "2 8", opacity: 0.4 },
  }));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="G / start gate"
        title="Asynchronous arrival. One release."
        sentence="Each cell registers once; the Nth registration releases the controller barrier, and START wakes every awaiting cell."
      />

      <Row gap={12} align="center" wrap>
        {[0, 1, 2, 3].map((i) => (
          <Button
            key={i}
            variant="secondary"
            aria-pressed={arrivals[i]}
            onClick={() => arrive(i)}
            disabled={arrivals[i] || released}
          >
            Register c{i}
          </Button>
        ))}
        <Button variant="primary" onClick={() => setReleased(true)} disabled={!all || released}>
          Trigger START
        </Button>
        <Button variant="ghost" onClick={() => { setArrivals([false, false, false, false]); setReleased(false); }}>
          Reset apparatus
        </Button>
      </Row>

      <span className={clsx("text-xs font-medium", released ? categoryClassName("cyan") : inkClassName("tertiary"))}>
        {released ? "all awaiters → Ready" : `${count} / 4 registered`}
      </span>

      <div style={{ height: 440 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
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
