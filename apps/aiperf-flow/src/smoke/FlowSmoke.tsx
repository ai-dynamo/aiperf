/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { ReactFlow, Background, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion } from "motion/react";

function SmokeNode({ data }: { data: { label: string } }): React.JSX.Element {
  return (
    <motion.div
      layout
      className="rounded-none border border-neutral-800 bg-white px-4 py-2 text-sm font-medium"
    >
      {data.label}
    </motion.div>
  );
}

const nodeTypes = { smoke: SmokeNode };

const initialNodes: Node[] = [
  { id: "a", type: "smoke", position: { x: 0, y: 0 }, data: { label: "A" } },
  { id: "b", type: "smoke", position: { x: 220, y: 0 }, data: { label: "B" } },
];

const initialEdges: Edge[] = [{ id: "a-b", source: "a", target: "b" }];

/** Proves React Flow + Motion render and interoperate; deleted once real components exist. */
export function FlowSmoke(): React.JSX.Element {
  const [nodes] = useState(initialNodes);
  const [edges] = useState(initialEdges);

  return (
    <div style={{ width: 400, height: 200 }}>
      <ReactFlow nodeTypes={nodeTypes} nodes={nodes} edges={edges} fitView>
        <Background />
      </ReactFlow>
    </div>
  );
}
