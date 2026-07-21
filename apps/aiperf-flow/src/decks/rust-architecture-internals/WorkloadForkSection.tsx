/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 06 — one execution core, two workload shapes. Scheduled conversations and compiled
//! trace graphs share phase orchestration, then join into worker-local dispatch. Ported from
//! `WorkloadFork` in the canvas source.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  headerNode,
  cardNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type Workload = "scheduled" | "graph";

function buildNodes(): Node[] {
  return [
    headerNode("band-scheduled", 0, 0, "Scheduled conversations"),
    cardNode("s-dataset", 0, 50, "Dataset + sampler", "conversation positions"),
    cardNode("s-arrival", 0, 150, "arrival policy", "constant · Poisson · Gamma · burst"),
    cardNode("s-slotpool", 0, 250, "SlotPool", "bounded admission"),
    cardNode("s-dispatcher", 0, 350, "TurnDispatcher", "dispatch_turn to terminal"),

    headerNode("band-graph", 460, 0, "Compiled trace graph"),
    cardNode("g-resolver", 460, 50, "Trace resolver", "DAG · AIPerf · WEKA · Dynamo"),
    cardNode("g-bundle", 460, 150, "GraphInputBundle", "program + SegmentStore"),
    cardNode("g-warmup", 380, 250, "warmup rewrite", "prime at t*"),
    cardNode("g-profiling", 600, 250, "profiling chop", "replay from t*"),
    cardNode("g-executor", 460, 350, "graph executor", "firing gates + dependencies"),

    cardNode("join", 230, 470, "worker-local dispatch", "WorkerSink | GraphSink", undefined, "primary"),
  ];
}

const edges: Edge[] = [
  flowEdge("e-s1", "s-dataset", "s-arrival"),
  flowEdge("e-s2", "s-arrival", "s-slotpool"),
  flowEdge("e-s3", "s-slotpool", "s-dispatcher"),
  flowEdge("e-g1", "g-resolver", "g-bundle"),
  flowEdge("e-g2", "g-bundle", "g-warmup"),
  flowEdge("e-g3", "g-bundle", "g-profiling"),
  flowEdge("e-g4", "g-warmup", "g-executor"),
  flowEdge("e-g5", "g-profiling", "g-executor"),
  flowEdge("e-s-join", "s-dispatcher", "join"),
  flowEdge("e-g-join", "g-executor", "join"),
];

/** Section 06 diagram: the two workload shapes converging on worker-local dispatch. */
export function WorkloadForkSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [workload, setWorkload] = useState<Workload>("scheduled");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="06"
          title="One execution core, two workload shapes"
          subtitle="Scheduled conversations and compiled trace graphs share phase orchestration, then use their worker-local dispatch and observation seams."
        />
        <Segmented
          ariaLabel="Workload"
          value={workload}
          onChange={setWorkload}
          options={[
            { id: "scheduled", label: "Scheduled" },
            { id: "graph", label: "Graph" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes()} edges={edges} height={560} />
      <p className={`text-center text-xs ${inkClassName("tertiary")}`}>
        scheduled and graph workloads dispatch through worker-local sinks
      </p>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          {workload === "scheduled" ? (
            <>
              Warmup and profiling share <Code inline>PhaseOrchestrator</Code>; stop, grace, cancel, and drain remain
              Clock-driven.
            </>
          ) : (
            <>The handoff frontier is consumed once. Terminal trajectory warmup failure stops before profiling.</>
          )}
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "scheduled.rs", path: "rust/runtime/src/scheduled.rs" },
          { label: "phase_runtime.rs", path: "rust/runtime/src/phase_runtime.rs" },
          { label: "graph input", path: "rust/runtime/src/engine/graph_input.rs" },
          { label: "graph phases", path: "rust/runtime/src/engine/graph_phase_runtime.rs" },
        ]}
      />
    </SectionShell>
  );
}
