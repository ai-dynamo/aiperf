/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, DbNode, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of the GraphView: trace datasets compiled once, then derived into
// warmup and profiling programs and executed through the shared Clock/RequestSink seam.

/** GraphView: trace datasets compiled once, then derived into warmup and profiling programs. */
export function GraphPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Trace replay path">
        Trace datasets bypass the linear dataset loader: one graph resolver strictly decodes the source, compiles it
        into shared segments, then derives phase-specific programs for warmup and profiling.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · GRAPH",
          title: "How are traces replayed?",
          body: "Decoded once, compiled to segments, derived into warmup and profiling.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Resolve source",
            diagram: (
              <Diagram>
                <NodeChip>dag_jsonl</NodeChip>
                <MiniArrow />
                <NodeChip accent>Resolver</NodeChip>
              </Diagram>
            ),
            children:
              "The GraphInputAdapterResolver selects an identity and strictly decodes dag_jsonl, WEKA, or Dynamo sources.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Compile once",
            diagram: (
              <Diagram>
                <NodeChip>LCP trie</NodeChip>
                <MiniArrow />
                <DbNode accent>SegmentStore</DbNode>
              </Diagram>
            ),
            children:
              "The compiler builds an LCP trie with dense interning into a GraphInputBundle of program + SegmentStore.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Sample t*",
            diagram: (
              <Diagram>
                <NodeChip>TStarSampler</NodeChip>
                <MiniArrow />
                <NodeChip accent>t*</NodeChip>
              </Diagram>
            ),
            children: "The TStarSampler picks a seeded trajectory start; profiling chop replays from the sampled t*.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Warmup handoff",
            diagram: (
              <Diagram>
                <NodeChip>warmup</NodeChip>
                <MiniArrow />
                <NodeChip accent>frontier</NodeChip>
              </Diagram>
            ),
            children: "Warmup rewrite primes prefixes before the frontier, resuming exactly once at the handoff.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "Execute graph",
            diagram: (
              <Diagram>
                <NodeChip accent>executor</NodeChip>
                <MiniArrow />
                <MiniBars heights={[52, 88, 44, 70]} />
              </Diagram>
            ),
            children:
              "Graph policies (root, arrival, admission, failure) drive an executor of firing gates and dependencies.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Place dispatch",
            diagram: (
              <Diagram>
                <NodeChip>placement</NodeChip>
                <MiniArrow />
                <NodeChip accent>RequestSink</NodeChip>
              </Diagram>
            ),
            children: "The placement factory routes each trace to a worker-local sink — one dispatch per graph node.",
          },
          {
            accent: "orange",
            badge: 7,
            title: "Emit outputs",
            diagram: (
              <Diagram>
                <NodeChip>CapturedRecord</NodeChip>
                <MiniArrow />
                <NodeChip accent>phase metrics</NodeChip>
              </Diagram>
            ),
            children: "Per-node CapturedRecords feed phase metrics and the warmup handoff back into the run.",
          },
        ]}
      />

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
