/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! T / Flat controller merge — every cell ships its terminal partition directly to the
//! controller. Hierarchy requests are refused before startup.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import { Row } from "../../layout/Row.js";
import { categoryClassName, inkClassName } from "../../theme/tokens.js";
import { MechHeader } from "./parts.js";

export function TreePage(): React.JSX.Element {
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
        subtitle: "8 terminal partitions in",
        className: categoryClassName("cyan"),
      },
    },
    ...cells.map((c): Node => ({
      id: `cell-${c}`,
      type: "chip",
      position: { x: c * 95, y: 380 },
      data: { label: `c${c} · ${safePayload}u`, strokeRole: "secondary" },
    })),
  ];

  const edges: Edge[] = [
    ...cells.map((c): Edge => ({ id: `e-cell-${c}`, source: `cell-${c}`, target: "controller", type: "flow" })),
  ];

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="T / flat controller merge"
        title="Ship every cell partition to the controller"
        sentence="Cells ship directly to the controller for global-order or associative store merge. Hierarchy requests are refused before startup."
      />

      <Row gap={12} align="center" wrap>
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

      <AutoLayoutFlow nodes={nodes} edges={edges} layout={{ direction: "DOWN" }} height={480} />

      <p className={clsx("text-xs", categoryClassName("cyan"))}>
        8 cell partitions → controller merge → one report; hierarchy requests are unavailable.
      </p>
    </div>
  );
}
