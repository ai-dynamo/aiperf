/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! C / Partition courier — the terminal unary payload carries fresh shipper PeerInfo plus a
//! records/folded-store partition; the controller registers that peer before ACKing. Ported
//! from the canvas `Courier`: move a packet through origin → relay → controller, then return a
//! CellAck; a Retry button resets the route and counts attempts.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { inkClassName } from "../../theme/tokens.js";
import { MechHeader, NODE_ACTIVE } from "./parts.js";

export function CourierPage(): React.JSX.Element {
  const [position, setPosition] = useState<0 | 1 | 2>(0);
  const [acked, setAcked] = useState(false);
  const [attempt, setAttempt] = useState(1);

  const deliver = () => { setPosition(2); setAcked(false); };

  const nodes: Node[] = [
    {
      id: "origin",
      type: "card",
      position: { x: 0, y: 40 },
      data: { title: "fresh ship Velo", subtitle: "cell_peer", detail: "partition", className: position === 0 ? NODE_ACTIVE : undefined },
    },
    {
      id: "relay",
      type: "panel",
      position: { x: 300, y: 60 },
      data: { title: "raw unary", detail: "Velo route", className: position === 1 ? NODE_ACTIVE : undefined },
    },
    {
      id: "controller",
      type: "card",
      position: { x: 600, y: 40 },
      data: {
        title: "controller handler",
        subtitle: position === 2 ? "register_peer(shipper)" : "await payload",
        detail: acked ? "ACK returned" : "",
        className: position === 2 ? NODE_ACTIVE : undefined,
      },
    },
    {
      id: "packet",
      type: "chip",
      position: { x: position === 0 ? 60 : position === 1 ? 330 : 640, y: 180 },
      data: { label: "P · partition", strokeRole: "primary" },
    },
  ];

  const edges: Edge[] = [
    { id: "e-o-r", source: "origin", target: "relay", type: position >= 1 ? "flow" : undefined },
    { id: "e-r-c", source: "relay", target: "controller", type: position >= 2 ? "flow" : undefined },
  ];

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="C / partition courier"
        title="Ship through a fresh return route"
        sentence="The terminal unary payload includes the fresh shipper PeerInfo and a records or folded-store partition; the controller registers that peer before acknowledging."
      />

      <Row gap={12} align="center" wrap>
        <Button variant="primary" onClick={() => setPosition(1)} disabled={position !== 0}>
          Send toward controller
        </Button>
        <Button variant="secondary" onClick={deliver} disabled={position !== 1}>
          Deliver
        </Button>
        <Button variant="secondary" onClick={() => setAcked(true)} disabled={position !== 2 || acked}>
          {acked ? "ACK returned" : "Return CellAck"}
        </Button>
        <Button variant="ghost" onClick={() => { setPosition(0); setAcked(false); setAttempt((a) => Math.min(99, a + 1)); }}>
          Retry · attempt {attempt}
        </Button>
      </Row>

      <div style={{ height: 380 }}>
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

      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        {position === 2 ? "register_peer(shipper) → Return CellAck" : "packet in transit on the raw unary Velo route"}
      </p>
    </div>
  );
}
