/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the CellularView page: multi-process scale.

const nodes: Node[] = [
  bandHeader("b-promote", "Controller promotion", 0, 0),
  panel("execute", "aiperf --execute", "detect cells > 1", 0, 60),
  card("launcher", "cell launcher", undefined, "partition budgets + envelope", 300, 60),
  card("controller", "controller transport", undefined, "Velo endpoints + lifecycle", 620, 60),

  bandHeader("b-cell", "Cell execution", 0, 200),
  card("cell0", "aiperf --cell 0", undefined, "fetch sliced envelope", 0, 260),
  card("celln", "aiperf --cell N", undefined, "fetch sliced envelope", 0, 380),
  card("ordinary", "ordinary execute path", undefined, "prepare · phases · dispatch · metrics", 320, 320),
  panel("recordpart", "records partition", "global-order merge input", 660, 240),
  card("folded", "folded store", undefined, "exact-fold or sketch input", 660, 360),
  card("heartbeats", "heartbeats", undefined, "progress + health", 660, 480),

  bandHeader("b-merge", "Hierarchical merge", 0, 620),
  panel("messages", "cell messages", "partitions + artifacts", 0, 680),
  card("aggregators", "optional aggregators", undefined, "merge subtree stores", 300, 680),
  card("ctrlmerge", "controller merge", undefined, "global order or associative store merge", 620, 680),

  bandHeader("b-commit", "Single commit point", 0, 820),
  panel("sidecars", "sidecars on primary cell only", undefined, 0, 880),
  card("final", "final report + exporters", undefined, undefined, 340, 880),
];

const edges: Edge[] = [
  flow("execute", "launcher"),
  flow("launcher", "controller"),
  flow("cell0", "ordinary"),
  flow("celln", "ordinary"),
  flow("ordinary", "recordpart"),
  flow("ordinary", "folded"),
  dashed("ordinary", "heartbeats"),
  flow("messages", "aggregators"),
  flow("aggregators", "ctrlmerge"),
  flow("ctrlmerge", "final"),
];

/** CellularView: cells > 1 promotes the child into a controller over Velo. */
export function CellularPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Multi-process scale">
        A request with <code>cells &gt; 1</code> promotes its execution child into a controller. Cells receive sliced
        envelopes over Velo, run the ordinary single-process engine, and return mergeable partitions.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={640} />

      <Grid columns={4} gap={16}>
        <Callout tone="info" title="S1">Issuance authority assigns aggregate dispatch ordinals.</Callout>
        <Callout tone="info" title="S2">Records shards expose mergeable partitions.</Callout>
        <Callout tone="info" title="S3">Metrics heartbeats carry live snapshots.</Callout>
        <Callout tone="info" title="S4">Cell partitions define deterministic ownership.</Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "Controller", path: "rust/aiperf/src/runner_protocol/cellular_controller.rs" },
          { label: "Cell mode", path: "rust/aiperf/src/runner_protocol/cellular_cell.rs" },
          { label: "Aggregator", path: "rust/aiperf/src/runner_protocol/cellular_aggregator.rs" },
          { label: "Cellular seams", path: "rust/aiperf/src/cellular/mod.rs" },
        ]}
      />
    </div>
  );
}
