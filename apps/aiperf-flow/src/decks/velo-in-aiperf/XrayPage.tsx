/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! X / Registration X-ray — the controller learns cell identity and returns a pre-sliced
//! protocol-v2 envelope with the run-wide START handle. Ported from the canvas `Xray`: a
//! `useStepSimulator` drives the four-step decode→register→lookup→encode trace, and a toggle
//! spreads the three-layer request envelope apart to inspect it.

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
import { Callout } from "../../prose/Callout.js";
import { inkClassName, categoryClassName } from "../../theme/tokens.js";
import { MechHeader, NODE_ACTIVE } from "./parts.js";

const LAYERS = ["handler / aiperf.cell.register", "cell_peer / MessagePack bytes", "cell_id / u32"] as const;

const STEPS = [
  { label: "decode CellRegister", detail: "raw payload → CellRegister" },
  { label: "register_peer(cell)", detail: "establish return route" },
  { label: "spec_for(cell_id)", detail: "pure lookup by cell ID" },
  { label: "encode RegisterReply", detail: "envelope bytes + EventHandle" },
] as const;

export function XrayPage(): React.JSX.Element {
  const sim = useStepSimulator(STEPS, { autoPlayMs: 1100 });
  const trace = sim.index;
  const [opened, setOpened] = useState(false);

  const nodes: Node[] = LAYERS.map((label, i): Node => ({
    id: `layer-${i}`,
    type: "panel",
    position: { x: opened ? i * 90 : 0, y: i * 84 },
    data: {
      title: label,
      className: i === trace ? NODE_ACTIVE : undefined,
    },
  }));

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <MechHeader
        eyebrow="X / registration X-ray"
        title="Open the request"
        sentence="The controller learns cell identity, selects a pre-sliced protocol-v2 envelope, and returns it with the run-wide START handle."
      />

      <Grid columns="1fr 1fr" gap={16}>
        <div className="flex flex-col gap-3">
          <Button variant="secondary" aria-pressed={opened} onClick={() => setOpened((v) => !v)}>
            {opened ? "Close envelope" : "Dissect envelope"}
          </Button>
          <div style={{ height: 320 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={nodes}
              edges={[]}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
            </ReactFlow>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <Row gap={8} align="center" wrap>
            <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>
              Back
            </Button>
            <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
              Next step
            </Button>
            <span className={clsx("text-xs font-medium", inkClassName("tertiary"))}>
              {trace + 1}/{STEPS.length}
            </span>
          </Row>

          <ol className="flex flex-col gap-2">
            {STEPS.map((step, i) => (
              <li
                key={step.label}
                className={clsx(
                  "rounded-md border px-3 py-2 shadow-sm",
                  i === trace ? NODE_ACTIVE : clsx("border-stroke-secondary", inkClassName("tertiary")),
                )}
              >
                <div className="text-xs font-bold uppercase tracking-wide">
                  {i + 1} / {step.label}
                </div>
                {i === trace && (
                  <div className={clsx("mt-1 text-xs", inkClassName("secondary"))}>{step.detail}</div>
                )}
              </li>
            ))}
          </ol>

          <Callout tone={trace === 3 ? "success" : "info"} title="RegisterReply">
            <div className={clsx("font-mono text-xs", inkClassName("primary"))}>envelope: protocol-v2 bytes</div>
            <div className={clsx("font-mono text-xs", inkClassName("primary"))}>start_event: EventHandle</div>
          </Callout>
        </div>
      </Grid>

      <p className={clsx("text-xs", categoryClassName("cyan"))}>
        RegisterReply {"{"} envelope: protocol-v2 bytes, start_event: EventHandle {"}"}
      </p>
    </div>
  );
}
