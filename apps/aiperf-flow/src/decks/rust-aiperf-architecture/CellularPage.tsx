/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, RoundNode, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of AIPerf's cellular scale-out: cells > 1 promotes the execution child
// into a controller over Velo, cells run the ordinary engine, and one controller merge commits.

/** CellularView: cells > 1 promotes the child into a controller over Velo. */
export function CellularPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Multi-process scale">
        A request with <code>cells &gt; 1</code> promotes its execution child into a controller. Cells receive sliced
        envelopes over Velo, run the ordinary single-process engine, and return mergeable partitions.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · CELLULAR",
          title: "How does one run scale out?",
          body: "cells > 1 promotes the child to a controller fanning work over Velo.",
        }}
        liveWire={0}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Controller promotion",
            diagram: (
              <Diagram>
                <NodeChip>--execute</NodeChip>
                <MiniArrow />
                <NodeChip accent>controller</NodeChip>
              </Diagram>
            ),
            children: "aiperf --execute detects cells > 1: the cell launcher partitions budgets and opens Velo endpoints.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "cell launcher",
            diagram: (
              <Diagram>
                <NodeChip accent>envelope</NodeChip>
                <MiniArrow />
                <RoundNode>0</RoundNode>
                <RoundNode accent>N</RoundNode>
              </Diagram>
            ),
            children: "Sliced envelopes fan out to each cell, carrying its budget partition and dispatch ordinals.",
          },
          {
            accent: "green",
            badge: 3,
            title: "aiperf --cell 0",
            diagram: (
              <Diagram>
                <RoundNode accent>0</RoundNode>
                <MiniArrow />
                <NodeChip>execute path</NodeChip>
              </Diagram>
            ),
            children: "Each cell fetches its envelope and runs the ordinary engine: prepare · phases · dispatch · metrics.",
          },
          {
            accent: "purple",
            badge: 4,
            title: "Mergeable partitions",
            diagram: (
              <Diagram>
                <NodeChip>records</NodeChip>
                <MiniArrow />
                <NodeChip accent>folded store</NodeChip>
              </Diagram>
            ),
            children: "Cells emit a global-order records partition and an exact-fold or sketch folded store.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "heartbeats",
            diagram: (
              <Diagram>
                <RoundNode accent>N</RoundNode>
                <MiniArrow />
                <NodeChip>controller</NodeChip>
              </Diagram>
            ),
            children: "Cells stream progress and health snapshots back to the controller during the run.",
          },
          {
            accent: "orange",
            badge: 6,
            title: "controller merge",
            diagram: (
              <Diagram>
                <RoundNode>0</RoundNode>
                <RoundNode accent>N</RoundNode>
                <MiniArrow />
                <NodeChip accent>merge</NodeChip>
              </Diagram>
            ),
            children: "Cells ship directly to the controller, which performs global-order or associative store merge. Hierarchy requests are refused before startup.",
          },
          {
            accent: "red",
            badge: 7,
            title: "Single commit point",
            diagram: (
              <Diagram>
                <NodeChip accent>merge</NodeChip>
                <MiniArrow />
                <NodeChip>report + exporters</NodeChip>
              </Diagram>
            ),
            children: "Sidecars run on the primary cell only; one final report plus exporters commit once.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Controller", path: "rust/runtime/src/engine/cellular_controller.rs" },
          { label: "Cell mode", path: "rust/runtime/src/engine/cellular_cell.rs" },
          { label: "Hierarchy refusal", path: "rust/runtime/src/engine/cellular_aggregator.rs" },
          { label: "Cellular seams", path: "rust/runtime/src/cellular/mod.rs" },
        ]}
      />
    </div>
  );
}
