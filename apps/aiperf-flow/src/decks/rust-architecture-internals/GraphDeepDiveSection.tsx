/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 07 — a graph is compiled once, then cut differently by phase. Four trace formats,
//! strict decode → trie lowerer → SegmentPool::freeze → GraphInputBundle, the t* transform band,
//! the firing-gate chain, and the two terminal branches. Ported from `GraphDeepDive`.

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
  chipNode,
  panelNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type GraphFocus = "warmup" | "profiling" | "handoff";

const TRACE_FORMATS = ["dag_jsonl", "aiperf_trace", "weka_trace", "dynamo_trace"] as const;

function transformTitle(focus: GraphFocus): string {
  switch (focus) {
    case "warmup":
      return "rewrite_for_warmup(t*) · boundary nodes only";
    case "profiling":
      return "chop_trie_at_tstar(t*) · drop pre-t* and re-root";
    case "handoff":
      return "chop_trie_at_frontier · drop executed nodes using lane handoff";
  }
}

function buildNodes(focus: GraphFocus): Node[] {
  const nodes: Node[] = [];
  TRACE_FORMATS.forEach((format, index) => {
    nodes.push(chipNode(`fmt-${format}`, index * 190, 0, format));
  });
  nodes.push(
    cardNode("decode", 40, 90, "strict format decode", "format-specific policy"),
    cardNode("lowerer", 230, 90, "recorded trie lowerer", "arrival_offset_us", "content-addressed segments"),
    cardNode("freeze", 430, 90, "SegmentPool::freeze", "InMemorySegmentStore", "bundle wraps Arc<dyn SegmentStore>"),
    cardNode("bundle", 640, 90, "GraphInputBundle", "plans + metadata"),

    cardNode("sampler", 20, 210, "WindowTStarSampler", "lane 0 at phase split", "default [0,0] → full replay", "primary"),
    panelNode("transform", 260, 210, transformTitle(focus), undefined, "primary"),
  );
  for (let i = 0; i < 6; i += 1) {
    nodes.push(chipNode(`node-n${i}`, 260 + i * 90, 300, `n${i}`));
  }
  nodes.push(
    cardNode("await-inputs", 0, 400, "await_inputs", "AND fan-in channel state"),
    cardNode("firing-gate", 220, 400, "firing gate max()", "4 edge delay families", "+ node min_start_delay_us"),
    cardNode("sleep", 450, 400, "Clock::sleep_ns", "delay to firing wall"),
    cardNode("graphsink", 680, 400, "worker GraphSink", "materialize + dispatch"),

    panelNode("recycle", 150, 510, "GraphPressureRecycle", "only with agentic cache warmup duration"),
    panelNode("warmup-failed", 500, 510, "trajectory_warmup_failed", "abort before profiling at phase finalize", "primary"),
  );
  return nodes;
}

const edges: Edge[] = [
  flowEdge("e-decode-lowerer", "decode", "lowerer"),
  flowEdge("e-lowerer-freeze", "lowerer", "freeze"),
  flowEdge("e-freeze-bundle", "freeze", "bundle"),
  flowEdge("e-sampler-transform", "sampler", "transform"),
  flowEdge("e-await-gate", "await-inputs", "firing-gate"),
  flowEdge("e-gate-sleep", "firing-gate", "sleep"),
  flowEdge("e-sleep-sink", "sleep", "graphsink"),
];

/** Section 07 diagram: single graph compile, phase-specific t* cut, and firing-gate chain. */
export function GraphDeepDiveSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [focus, setFocus] = useState<GraphFocus>("profiling");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="07"
          title="A graph is compiled once, then cut differently by phase"
          subtitle="One adapter owns strict decode, one bundle owns plans and frozen segments, and t* produces distinct warmup, profiling, and optional handoff transforms."
        />
        <Segmented
          ariaLabel="Graph focus"
          value={focus}
          onChange={setFocus}
          options={[
            { id: "warmup", label: "Warmup" },
            { id: "profiling", label: "Profiling" },
            { id: "handoff", label: "Handoff" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(focus)} edges={edges} height={600} />

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Graph compile output">
          <Code inline>GraphInputBundle</Code> contains plans, frozen segments, and metadata.
        </Callout>
        <Callout tone="info" title="Lane nuance">
          Normal phase splitting samples lane 0; pressure recycle creates additional lane-salted plans.
        </Callout>
        <Callout tone="warning" title="Handoff condition">
          The runtime stashes handoff after all pressure traces return with an empty failure ledger, then consumes it
          with <Code inline>take()</Code>.
        </Callout>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          Warmup records node failures during resilient execution and promotes the ledger to a terminal run failure at
          phase finalization.
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "input resolver", path: "rust/runtime/src/engine/graph_input.rs" },
          { label: "t* sampler", path: "rust/runtime/src/graph/tstar.rs" },
          { label: "snapshot transforms", path: "rust/runtime/src/graph/snapshot.rs" },
          { label: "graph phases", path: "rust/runtime/src/engine/graph_phase_runtime.rs" },
          { label: "executor", path: "rust/runtime/src/graph/executor.rs" },
        ]}
      />
    </SectionShell>
  );
}
