/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constellation index — ten mechanism cards arranged around a Velo "core", ported from the
//! canvas source's `Index` component. Rendered as a real `@xyflow/react` graph: dashed edges
//! radiate from a center chip to each mechanism card, and clicking a card navigates to that
//! mechanism's page (the aiperf-flow analogue of the canvas's `open(view)`).

import { useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { categoryClassName, inkClassName } from "../../theme/tokens.js";
import type { VeloPageId } from "./VeloInAiperfDeck.js";

/** The ten mechanisms, in canvas order — id (page), mnemonic mark, and title. */
export const MECHANISMS: ReadonlyArray<{ id: Exclude<VeloPageId, "index">; mark: string; title: string }> = [
  { id: "radar", mark: "R", title: "Connection radar" },
  { id: "xray", mark: "X", title: "Registration X-ray" },
  { id: "gate", mark: "G", title: "Start gate" },
  { id: "press", mark: "P", title: "MessagePack press" },
  { id: "scope", mark: "H", title: "Heartbeat scope" },
  { id: "courier", mark: "C", title: "Partition courier" },
  { id: "merge", mark: "M", title: "Merge machine" },
  { id: "phaser", mark: "Φ", title: "Phaser clock" },
  { id: "dataset", mark: "D", title: "Dataset floodgate" },
  { id: "tree", mark: "T", title: "Hierarchy refusal" },
];

// Constellation offsets around a center core, roughly matching the canvas `.i0..i9` placement,
// scaled down from the original spacing so the ring reads as one coherent cluster instead of ten
// cards scattered across a mostly-empty canvas.
const RING: ReadonlyArray<[number, number]> = [
  [-336, -132],
  [-114, -180],
  [312, -126],
  [-384, 24],
  [336, 24],
  [-324, 180],
  [-90, 216],
  [312, 180],
  [-228, -48],
  [180, 126],
];

export function IndexPage({ onSelect }: { onSelect: (id: VeloPageId) => void }): React.JSX.Element {
  const { nodes, edges } = useMemo(() => {
    const nodeList: Node[] = [
      {
        // `chip` nodes render no Handles (they're not connectable, per the shared node
        // vocabulary), so edges couldn't attach to a chip-typed hub — this must be a `card` (or
        // `panel`) for the ten radiating edges below to actually draw.
        id: "core",
        type: "card",
        position: { x: 0, y: 40 },
        data: { title: "Velo plane", strokeRole: "secondary", className: categoryClassName("cyan") },
      },
    ];
    const edgeList: Edge[] = [];
    MECHANISMS.forEach((m, i) => {
      const [dx, dy] = RING[i]!;
      nodeList.push({
        id: m.id,
        type: "card",
        position: { x: dx, y: 40 + dy },
        data: {
          title: m.title,
          subtitle: `${m.mark} / ${String(i + 1).padStart(2, "0")}`,
        },
      });
      edgeList.push({ id: `e-core-${m.id}`, source: "core", target: m.id, style: { strokeDasharray: "2 8" } });
    });
    return { nodes: nodeList, edges: edgeList };
  }, []);

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <div>
        <Eyebrow tone="cyan">AIPerf cellular transport</Eyebrow>
        <h2 className={clsx("mt-1 text-2xl font-semibold", inkClassName("primary"))}>Velo mechanisms</h2>
        <p className={clsx("mt-1 max-w-2xl text-sm", inkClassName("secondary"))}>
          Ten interactive instruments expose how cellular identity, synchronization, distribution,
          and reduction cross the Velo plane.
        </p>
      </div>

      <div style={{ height: 620 }}>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          onNodeClick={(_, node) => {
            if (node.id !== "core") onSelect(node.id as VeloPageId);
          }}
          fitView
          fitViewOptions={{ padding: 0.1 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>

      <p className={clsx("text-xs", inkClassName("tertiary"))}>
        Select any instrument above, or click a card to open it.
      </p>
    </div>
  );
}
