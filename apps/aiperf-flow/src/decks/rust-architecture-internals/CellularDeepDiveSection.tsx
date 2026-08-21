/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 09 — cellular wraps the run core with ownership and merge planes. Controller,
//! three cells (ordinary run core + CellRecordsShipper), the dense dispatch ordinal formula,
//! and the focus-dependent merge. Ported from `CellularDeepDive` in the canvas source.

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
  panelNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type CellularFocus = "default" | "phaser" | "hierarchy";

function buildNodes(): Node[] {
  const nodes: Node[] = [
    cardNode("controller", 330, 0, "cellular controller", "cells > 1 promotion", "slice envelope · launch · collect · commit", "primary"),
  ];
  for (let i = 0; i < 3; i += 1) {
    nodes.push(
      headerNode(`cell-band-${i}`, i * 300, 110, `aiperf --cell ${i}`),
      cardNode(`core-${i}`, i * 300, 160, "ordinary run core", "run_v2 + partition env"),
      cardNode(`shipper-${i}`, i * 300, 250, "CellRecordsShipper", "heartbeat + terminal partition"),
    );
  }
  nodes.push(
    panelNode("ordinals", 200, 360, "SCHEDULED DENSE DISPATCH ORDINALS", "phase_base + local_index × cell_count + cell_id", "primary"),
  );

  nodes.push(
    cardNode("merge", 270, 460, "controller merge", "records by order | graph concat | folded store", undefined, "primary"),
    cardNode("commit", 270, 560, "native-v2.json + exporters", "one final commit point", undefined, "primary"),
  );
  return nodes;
}

function buildEdges(): Edge[] {
  const edges: Edge[] = [];
  for (let i = 0; i < 3; i += 1) {
    edges.push(flowEdge(`e-ctl-core-${i}`, "controller", `core-${i}`));
    edges.push(flowEdge(`e-core-shipper-${i}`, `core-${i}`, `shipper-${i}`));
  }
  for (let i = 0; i < 3; i += 1) {
    edges.push(flowEdge(`e-ship-merge-${i}`, `shipper-${i}`, "merge"));
  }
  edges.push(flowEdge("e-merge-commit", "merge", "commit"));
  return edges;
}

/** Section 09 diagram: cellular controller, cells, and the focus-selected merge plane. */
export function CellularDeepDiveSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [focus, setFocus] = useState<CellularFocus>("default");
  const isHierarchyRefusal = focus === "hierarchy";
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="09"
          title="Cellular wraps the run core with ownership and merge planes"
          subtitle="The controller slices one run, cells execute the ordinary path, Velo carries progress and partitions, and the controller commits one merged report."
        />
        <Segmented
          ariaLabel="Cellular focus"
          value={focus}
          onChange={setFocus}
          options={[
            { id: "default", label: "Default star" },
            { id: "phaser", label: "Phaser opt-in" },
            { id: "hierarchy", label: "Hierarchy refusal" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes()} edges={buildEdges()} height={620} />
      <p className={`text-center text-xs ${inkClassName("tertiary")}`}>
        await all registrations → {focus === "phaser" ? "Phaser Started generation" : "Velo start event"}
      </p>

      <Grid columns="1fr 1fr 1fr" gap={12}>
        <Callout tone="success" title="Built with Velo">
          The <Code inline>velo</Code> feature carries register, heartbeat, raw records, and folded stores as
          MessagePack payloads.
        </Callout>
        <Callout tone={isHierarchyRefusal ? "warning" : "info"} title="Hierarchy policy">
          Phaser START, dataset fan-out, and barrier-free start are opt-in. Hierarchy requests are refused before startup.
        </Callout>
        <Callout tone="danger" title="Cellular validation and deadlines">
          Cellular validation returns an error for DynoSim transports; registration and collect deadlines return
          failures for incomplete runs.
        </Callout>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          Cells ship every terminal partition directly to the controller; hierarchy requests are refused before startup.
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "controller", path: "rust/runtime/src/engine/cellular_controller.rs" },
          { label: "cell", path: "rust/runtime/src/engine/cellular_cell.rs" },
          { label: "launcher", path: "rust/runtime/src/engine/cell_launcher.rs" },
          { label: "issuance", path: "rust/runtime/src/cellular/issuance.rs" },
          { label: "Velo transport", path: "rust/runtime/src/cellular/transport/velo_transport.rs" },
          { label: "hierarchy refusal", path: "rust/runtime/src/engine/cellular_aggregator.rs" },
        ]}
      />
    </SectionShell>
  );
}
