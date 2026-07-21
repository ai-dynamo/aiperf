/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! R / Connection radar — resolve a controller endpoint via the hello exchange. Ported from the
//! canvas `Radar`: each user-triggered sweep discovers one more endpoint (0..4); clicking a
//! resolved endpoint locks it, yielding `endpoint → _hello → register_peer(controller)`.

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

const ENDPOINTS = ["tcp://host:port", "uds://path", "loopback", "ephemeral"] as const;
const POS: ReadonlyArray<[number, number]> = [
  [-360, -140],
  [360, -160],
  [340, 140],
  [-360, 160],
];

export function RadarPage(): React.JSX.Element {
  const [sweep, setSweep] = useState(0);
  const [locked, setLocked] = useState(-1);
  const discovered = Math.min(4, Math.max(0, sweep));

  const nodes: Node[] = [
    {
      id: "center",
      type: "chip",
      position: { x: 0, y: 0 },
      data: { label: "controller", strokeRole: "primary", className: categoryClassName("cyan") },
    },
    ...ENDPOINTS.map((label, i): Node => {
      const resolved = i < discovered;
      return {
        id: `ep-${i}`,
        type: "panel",
        position: { x: POS[i]![0], y: POS[i]![1] },
        data: {
          title: resolved ? label : "unresolved",
          detail: resolved ? "PeerInfo" : "awaiting sweep",
          className: locked === i ? NODE_ACTIVE : undefined,
        },
      };
    }),
  ];

  const edges: Edge[] = ENDPOINTS.map((_, i): Edge => ({
    id: `e-${i}`,
    source: "center",
    target: `ep-${i}`,
    type: locked === i ? "flow" : undefined,
    style: i < discovered ? undefined : { strokeDasharray: "2 8", opacity: 0.4 },
  }));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="R / connection radar"
        title="Resolve a controller"
        sentence="A known TCP or UDS endpoint enters the hello exchange and becomes routable controller PeerInfo."
      />

      <Row gap={12} align="center" wrap>
        <Button variant="primary" onClick={() => setSweep((s) => Math.min(4, s + 1))} disabled={discovered >= 4}>
          {discovered < 4 ? "Sweep sector" : "All resolved"}
        </Button>
        <Button variant="ghost" onClick={() => { setSweep(0); setLocked(-1); }}>
          Reset lock
        </Button>
        <span className={clsx("text-xs font-medium", locked >= 0 ? categoryClassName("cyan") : inkClassName("tertiary"))}>
          {locked >= 0
            ? `${ENDPOINTS[locked]} → _hello → register_peer(controller)`
            : `${discovered}/4 sectors resolved · each sweep is user-triggered`}
        </span>
      </Row>

      <div style={{ height: 460 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          onNodeClick={(_, node) => {
            const m = /^ep-(\d)$/.exec(node.id);
            if (m && Number(m[1]) < discovered) setLocked(Number(m[1]));
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
