/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Φ / Phaser clock — Attach captures the current generation; that prefix returns in the unary
//! reply (replay), later generations arrive by active-message push (live). Ported from the
//! canvas `Phaser`: a `useStepSimulator` advances the generation counter (2..12) rendered as a
//! clock face of g-nodes; attaching freezes the boundary and classifies each generation.

import { useState } from "react";
import type { Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Button } from "../../prose/Button.js";
import { inkClassName, categoryClassName, strokeClassName } from "../../theme/tokens.js";
import { MechHeader, NODE_ACTIVE } from "./parts.js";

// Generation counter starts at 2 and advances up to 12 (canvas `Phaser` initial state).
const GENERATIONS = Array.from({ length: 11 }, (_, i) => i + 2);

export function PhaserPage(): React.JSX.Element {
  const sim = useStepSimulator(GENERATIONS);
  const generation = GENERATIONS[sim.index] ?? 2;
  const [attach, setAttach] = useState<number | null>(null);
  const events = Array.from({ length: generation }, (_, i) => i + 1);

  const nodes: Node[] = [
    {
      id: "hub",
      type: "chip",
      position: { x: 0, y: 0 },
      data: { label: `generation ${generation}`, strokeRole: "primary", className: categoryClassName("cyan") },
    },
    ...events.map((g, i): Node => {
      const angle = (i / Math.max(8, generation)) * Math.PI * 2 - Math.PI / 2;
      const replay = attach !== null && g <= attach;
      const live = attach !== null && g > attach;
      return {
        id: `g-${g}`,
        type: "chip",
        position: { x: 260 * Math.cos(angle), y: 260 * Math.sin(angle) },
        data: { label: `g${g}`, strokeRole: "secondary", className: live ? NODE_ACTIVE : replay ? inkClassName("tertiary") : undefined },
      };
    }),
  ];

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="Φ / phaser clock"
        title="Replay, then live"
        sentence="Attach captures the current generation; that entire prefix returns in the unary reply, and only later generations arrive by active-message push."
      />

      <Row gap={12} align="center" wrap>
        <Button variant="primary" onClick={sim.next} disabled={generation >= 12}>
          Advance
        </Button>
        <Button variant="secondary" onClick={() => setAttach(generation)} disabled={attach !== null}>
          Attach subscriber now
        </Button>
        <Button variant="ghost" onClick={() => { sim.reset(); setAttach(null); }}>
          Reset
        </Button>
        <span className={clsx("text-xs font-medium", inkClassName("tertiary"))}>
          {attach !== null ? `attached @ generation ${attach}` : "not attached"}
        </span>
      </Row>

      <Grid columns="1.4fr 1fr" gap={16}>
        <div style={{ height: 420 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={[]}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>

        <div className={clsx("rounded-lg border p-3 shadow-sm", strokeClassName("primary"))}>
          {attach === null ? (
            <p className={clsx("text-sm", inkClassName("tertiary"))}>Attach a subscriber to classify each generation.</p>
          ) : (
            <ul className="flex flex-col gap-1">
              {events.map((g) => (
                <li key={g} className="grid grid-cols-[36px_1fr] gap-2 font-mono text-xs">
                  <span className={g <= attach ? inkClassName("tertiary") : categoryClassName("cyan")}>g{g}</span>
                  <span className={g <= attach ? inkClassName("secondary") : categoryClassName("cyan")}>
                    {g <= attach ? "reply replay" : "live push"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Grid>
    </div>
  );
}
