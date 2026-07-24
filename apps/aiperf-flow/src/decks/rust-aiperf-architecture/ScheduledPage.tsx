/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, RoundNode, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of the ScheduledView: the paced workload path from dataset lowering,
// through phase/arrival policy and bounded admission, to prepared-turn dispatch and worker topology.

/** ScheduledView: the paced workload path from dataset lowering to worker topology. */
export function ScheduledPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Paced workload path">
        The scheduled workload lowers datasets into conversations, applies one arrival policy, admits work through
        bounded slots, and dispatches prepared turns over HTTP, gRPC, or DynoSim. Phases carry ramp, cancellation,
        grace, and drain; continuations receive FIFO priority.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · SCHEDULED",
          title: "How is load paced?",
          body: "Datasets lowered, phases driven, slots bounded, turns dispatched.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Lower dataset",
            diagram: (
              <Diagram>
                <NodeChip>INPUT</NodeChip>
                <MiniArrow />
                <NodeChip accent>LOADER</NodeChip>
              </Diagram>
            ),
            children: "Synthetic, file, or public inputs become dataset conversations through the loader.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Sample turns",
            diagram: (
              <Diagram>
                <NodeChip>SAMPLER</NodeChip>
                <MiniArrow />
                <NodeChip accent>NativeRunSpec</NodeChip>
              </Diagram>
            ),
            children:
              "Sequential, shuffle, or random sampling resolves into phases, limits, arrival, and endpoint profiles.",
          },
          {
            accent: "green",
            badge: 3,
            title: "Drive phases",
            diagram: (
              <Diagram>
                <NodeChip>warmup</NodeChip>
                <MiniArrow />
                <NodeChip accent>PhaseOrchestrator</NodeChip>
              </Diagram>
            ),
            children: "The PhaseOrchestrator runs warmup then profiling, carrying ramp, cancellation, grace, and drain.",
          },
          {
            accent: "yellow",
            badge: 4,
            title: "Arrival policy",
            diagram: (
              <Diagram>
                <NodeChip accent>policy</NodeChip>
                <MiniArrow />
                <MiniBars heights={[40, 68, 52, 90]} />
              </Diagram>
            ),
            children: "Request-rate, user-centric, or fixed policy drives constant, Poisson, Gamma, or burst arrivals.",
          },
          {
            accent: "red",
            badge: 5,
            title: "Bound admission",
            diagram: (
              <Diagram>
                <RoundNode>1</RoundNode>
                <RoundNode accent>2</RoundNode>
                <NodeChip>SlotPool + StopChecker</NodeChip>
              </Diagram>
            ),
            children: "The SlotPool admits work while the StopChecker enforces request and duration bounds.",
          },
          {
            accent: "purple",
            badge: 6,
            title: "Dispatch turns",
            diagram: (
              <Diagram>
                <NodeChip>PreparedTurn</NodeChip>
                <MiniArrow />
                <NodeChip accent>RequestSink</NodeChip>
              </Diagram>
            ),
            children:
              "The TurnDispatcher places materialized turns onto a worker-local HTTP, gRPC, or DirectRequest sink.",
          },
          {
            accent: "orange",
            badge: 7,
            title: "Worker topology",
            diagram: (
              <Diagram>
                <NodeChip>workers = 1</NodeChip>
                <MiniArrow />
                <NodeChip accent>workers &gt; 1</NodeChip>
              </Diagram>
            ),
            children:
              "One worker shares the coordinator runtime; workers > 1 run OS threads with current_thread + LocalSet.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Phase runtime", path: "rust/aiperf/src/phase_runtime.rs" },
          { label: "Scheduled bridge", path: "rust/aiperf/src/scheduled.rs" },
          { label: "Sharded workers", path: "rust/aiperf/src/runner_protocol/sharded_scheduled.rs" },
          { label: "Turn placement", path: "rust/aiperf/src/runner_protocol/turn_execution.rs" },
        ]}
      />
    </div>
  );
}
