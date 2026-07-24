/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, RoundNode, DbNode, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of the one-run hot path: frozen registries and strict DTOs at
// startup, then transport-native request execution on local observer graphs, then a single
// commit of native-v2.json plus compatibility exports. Each spoke is one beat of that run.

/** RuntimeView: the one-run hot path from frozen registries to committed report. */
export function RuntimePage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One request, end to end">
        This is the one-run hot path. Startup uses frozen registries and strict DTOs; request execution then stays on
        transport-native request types and local observer graphs. The final commit writes{" "}
        <code>native-v2.json</code> plus compatibility exports.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · ONE RUN",
          title: "What happens per run?",
          body: "Freeze registries, dispatch on local observers, commit once.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Author and bootstrap",
            diagram: (
              <Diagram>
                <NodeChip>Config v2</NodeChip>
                <MiniArrow />
                <NodeChip accent>stock</NodeChip>
              </Diagram>
            ),
            children: "AuthoredRunSpecV2 on stdin; RunnerApplication::stock freezes registries, resolvers, factories.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Validate and prepare",
            diagram: (
              <Diagram>
                <NodeChip>Coordinator</NodeChip>
                <MiniArrow />
                <NodeChip accent>Prepared op</NodeChip>
              </Diagram>
            ),
            children: "Coordinator resolves IDs and fails closed; workload/transport factories yield a one-shot op.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Phase runtime",
            diagram: (
              <Diagram>
                <NodeChip>warmup</NodeChip>
                <MiniArrow />
                <NodeChip accent>profiling</NodeChip>
              </Diagram>
            ),
            children: "Phase runtime drives warmup → profiling; the workload driver runs scheduled or graph work.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Admission and pacing",
            diagram: (
              <Diagram>
                <RoundNode>1</RoundNode>
                <RoundNode accent>2</RoundNode>
                <MiniArrow />
                <NodeChip>endpoint</NodeChip>
              </Diagram>
            ),
            children: "SlotPool, arrivals, and stop pace the prepared endpoint's request body + parser.",
          },
          {
            accent: "orange",
            badge: 5,
            title: "Clock authority",
            diagram: (
              <Diagram>
                <NodeChip>RealClock</NodeChip>
                <MiniArrow />
                <NodeChip accent>SimClock</NodeChip>
              </Diagram>
            ),
            children: "Arrival, admission, token, cancellation, and phase timing all come from the injected Clock.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Dispatch and observe",
            diagram: (
              <Diagram>
                <NodeChip accent>dispatch</NodeChip>
                <MiniArrow />
                <NodeChip>Observer</NodeChip>
              </Diagram>
            ),
            children: "RequestSink<R>::dispatch over HTTP/gRPC feeds a local RequestObserver: arrival→token→usage→terminal.",
          },
          {
            accent: "yellow",
            badge: 7,
            title: "Reduce and commit",
            diagram: (
              <Diagram>
                <DbNode accent>capture</DbNode>
                <MiniArrow />
                <MiniBars heights={[38, 72, 100, 82]} />
              </Diagram>
            ),
            children: "Per-worker capture merges once after drain; the accumulator + side channels commit report + artifacts.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Application composition", path: "rust/aiperf/src/runner_protocol/application.rs" },
          { label: "Registry contracts", path: "rust/aiperf/src/runner_protocol/registry.rs" },
          { label: "Request seam", path: "rust/loadgen-core/src/sink.rs" },
        ]}
      />
    </div>
  );
}
