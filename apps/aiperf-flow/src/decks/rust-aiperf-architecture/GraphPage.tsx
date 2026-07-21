/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the GraphView page: the trace replay path.

const nodes: Node[] = [
  bandHeader("b-decode", "Decode and compile once", 0, 0),
  panel("source", "trace source", "dag_jsonl · WEKA · Dynamo", 0, 60),
  card("resolver", "GraphInputAdapterResolver", undefined, "identity selection + strict decode", 280, 60),
  panel("compiler", "compiler", "LCP trie + dense interning", 600, 60),
  card("bundle", "GraphInputBundle", undefined, "program + SegmentStore", 840, 60),

  bandHeader("b-derive", "Derive phase programs", 0, 200),
  card("tstar", "TStarSampler", undefined, "seeded trajectory start", 0, 260),
  card("warmup", "warmup rewrite", undefined, "prime prefixes before the frontier", 280, 260),
  panel("handoff", "handoff frontier", "resume exactly once after warmup", 580, 260),
  card("chop", "profiling chop", undefined, "replay from sampled t*", 840, 260),

  bandHeader("b-execute", "Execute graph", 0, 400),
  panel("policies", "graph policies", "root · arrival · admission · failure", 0, 460),
  card("executor", "graph executor", undefined, "firing gates + dependencies", 280, 460),
  card("placement", "placement factory", undefined, "trace to worker-local sink", 580, 460),
  card("sink", "RequestSink<R>", undefined, "one dispatch per graph node", 860, 460),

  bandHeader("b-outputs", "Outputs", 0, 600),
  panel("record", "per-node CapturedRecord", undefined, 0, 660),
  card("metrics", "phase metrics + warmup handoff", undefined, undefined, 300, 660),
];

const edges: Edge[] = [
  flow("source", "resolver"),
  flow("resolver", "compiler"),
  flow("compiler", "bundle"),
  flow("tstar", "warmup"),
  flow("warmup", "handoff"),
  flow("handoff", "chop"),
  flow("policies", "executor"),
  flow("executor", "placement"),
  flow("placement", "sink"),
  flow("executor", "record"),
  flow("placement", "metrics"),
];

/** GraphView: trace datasets compiled once, then derived into warmup and profiling programs. */
export function GraphPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Trace replay path">
        Trace datasets bypass the linear dataset loader: one graph resolver strictly decodes the source, compiles it
        into shared segments, then derives phase-specific programs for warmup and profiling.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={600} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="One compiler">
          Python does not parse or lower WEKA/Dynamo graph inputs on the Rust path.
        </Callout>
        <Callout tone="info" title="Shared execution seam">
          Graph placement still ends at the same Clock and RequestSink/Observer interfaces as scheduled work.
        </Callout>
        <Callout tone="warning" title="Warmup failure">
          A terminal trajectory-warmup failure stops before profiling and returns a typed v2 failure.
        </Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "Graph input", path: "rust/aiperf/src/runner_protocol/graph_input.rs" },
          { label: "Graph phases", path: "rust/aiperf/src/runner_protocol/graph_phase_runtime.rs" },
          { label: "Snapshot transforms", path: "rust/aiperf/src/graph/snapshot.rs" },
          { label: "Graph executor", path: "rust/aiperf/src/graph/executor.rs" },
        ]}
      />
    </div>
  );
}
