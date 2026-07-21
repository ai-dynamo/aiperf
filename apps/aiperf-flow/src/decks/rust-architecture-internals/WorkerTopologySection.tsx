/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 08 — scale by tiling self-contained execution cells. The coordinator node plus a
//! scale-dependent worker layout: one coordinator reactor, three OS threads, or three
//! --cell processes with controller merge. Ported from `WorkerTopology`.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  cardNode,
  headerNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type Scale = "local" | "sharded" | "cellular";

function buildNodes(scale: Scale): Node[] {
  const nodes: Node[] = [
    cardNode(
      "coordinator",
      330,
      0,
      scale === "cellular" ? "cell controller" : "run coordinator",
      scale === "cellular" ? "partition budgets + merge results" : "own phases + partitions",
      undefined,
      "primary",
    ),
  ];

  if (scale === "local") {
    nodes.push(cardNode("worker-0", 330, 140, "COORDINATOR REACTOR", "current_thread + LocalSet", "schedule + sink · local"));
  } else if (scale === "sharded") {
    for (let i = 0; i < 3; i += 1) {
      nodes.push(cardNode(`worker-${i}`, i * 300, 140, `OS THREAD ${i}`, "current_thread + LocalSet", "schedule + sink · local"));
    }
  } else {
    for (let i = 0; i < 3; i += 1) {
      nodes.push(
        headerNode(`cell-band-${i}`, i * 300, 120, `aiperf --cell ${i}`),
        cardNode(`worker-${i}`, i * 300, 170, "ORDINARY RUN CORE", "run_v2 + partition env"),
        cardNode(`shipper-${i}`, i * 300, 260, "CellRecordsShipper", "heartbeat + terminal partition"),
      );
    }
    nodes.push(cardNode("merge", 330, 380, "controller merge", "records | folded stores", undefined, "primary"));
  }
  return nodes;
}

function buildEdges(scale: Scale): Edge[] {
  const edges: Edge[] = [];
  const count = scale === "local" ? 1 : 3;
  for (let i = 0; i < count; i += 1) {
    edges.push(flowEdge(`e-coord-worker-${i}`, "coordinator", `worker-${i}`));
  }
  if (scale === "cellular") {
    for (let i = 0; i < 3; i += 1) {
      edges.push(flowEdge(`e-worker-shipper-${i}`, `worker-${i}`, `shipper-${i}`));
      edges.push(flowEdge(`e-shipper-merge-${i}`, `shipper-${i}`, "merge"));
    }
  }
  return edges;
}

const CAPTION: Record<Scale, string> = {
  local: "coordinator reactor executes its local sink directly",
  sharded: "request or conversation budgets tile exactly across sub-cells",
  cellular: "each cell injects partition state and invokes the ordinary run_v2 path",
};

/** Section 08 diagram: the scale-selected worker/process tiling of the execution cell. */
export function WorkerTopologySection({ detail }: { detail: Detail }): React.JSX.Element {
  const [scale, setScale] = useState<Scale>("sharded");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="08"
          title="Scale by tiling self-contained execution cells"
          subtitle="The scheduler, endpoint table, transport, and observer graph stay co-located; ownership is partitioned across worker or process boundaries."
        />
        <Segmented
          ariaLabel="Scale"
          value={scale}
          onChange={setScale}
          options={[
            { id: "local", label: "1 worker" },
            { id: "sharded", label: "N threads" },
            { id: "cellular", label: "N processes" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(scale)} edges={buildEdges(scale)} height={480} />
      <p className={`text-center text-xs ${inkClassName("tertiary")}`}>{CAPTION[scale]}</p>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Thread topology">
          Each OS thread owns one current-thread Tokio runtime and <Code inline>LocalSet</Code>.
        </Callout>
        <Callout tone="success" title="Local metrics state">
          Measurement accumulates inside each worker and merges after callbacks stop.
        </Callout>
        <Callout tone="info" title="Cell execution">
          A cell injects partition state into the environment and invokes the ordinary <Code inline>run_v2</Code> path.
        </Callout>
      </Grid>
      {rank(detail) > 0 && scale !== "cellular" && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          transports contribute WorkerSink only; shared orchestration owns measurement and drain
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "turn execution", path: "rust/runtime/src/engine/turn_execution.rs" },
          { label: "sharded scheduled", path: "rust/runtime/src/engine/sharded_scheduled.rs" },
          { label: "cell controller", path: "rust/runtime/src/engine/cellular_controller.rs" },
          { label: "cellular seams", path: "rust/runtime/src/cellular/mod.rs" },
        ]}
      />
    </SectionShell>
  );
}
