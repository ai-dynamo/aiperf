/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the ScheduledView page: the paced workload path.

const nodes: Node[] = [
  bandHeader("b-prepare", "Prepare data and policy", 0, 0),
  panel("input", "dataset input", "synthetic · file · public", 0, 60),
  card("loader", "loader", undefined, "Dataset + conversations", 280, 60),
  panel("sampler", "sampler", "sequential · shuffle · random", 560, 60),
  card("spec", "NativeRunSpec", undefined, "phases · limits · arrival · endpoint profiles", 840, 60),

  bandHeader("b-drive", "Drive phases and arrivals", 0, 200),
  card("orchestrator", "PhaseOrchestrator", undefined, "warmup → profiling", 0, 260),
  card("policy", "workload policy", undefined, "request-rate · user-centric · fixed", 280, 260),
  panel("arrival", "arrival schedule", "constant · Poisson · Gamma · burst", 560, 260),
  card("slotpool", "SlotPool + StopChecker", undefined, "admission · request/duration bounds", 840, 260),

  bandHeader("b-place", "Place and dispatch", 0, 400),
  panel("turn", "PreparedTurn", "materialized conversation turn", 0, 460),
  card("dispatcher", "TurnDispatcher", undefined, "placement abstraction", 280, 460),
  panel("table", "worker-local endpoint table", "prepare_worker once", 560, 460),
  card("sink", "RequestSink<R>", undefined, "HTTP · gRPC · DirectRequest", 840, 460),

  bandHeader("b-topology", "Worker topology", 0, 600),
  panel("w1", "workers = 1", "coordinator current-thread runtime", 0, 660),
  card("wn", "workers > 1", undefined, "OS threads · current_thread + LocalSet", 300, 660),
];

const edges: Edge[] = [
  flow("input", "loader"),
  flow("loader", "sampler"),
  flow("sampler", "spec"),
  flow("orchestrator", "policy"),
  flow("policy", "arrival"),
  flow("arrival", "slotpool"),
  flow("turn", "dispatcher"),
  flow("dispatcher", "table"),
  flow("table", "sink"),
  dashed("sink", "w1"),
  flow("sink", "wn"),
];

/** ScheduledView: the paced workload path from dataset lowering to worker topology. */
export function ScheduledPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Paced workload path">
        The scheduled workload lowers datasets into conversations, applies one arrival policy, admits work through
        bounded slots, and dispatches prepared turns over HTTP, gRPC, or DynoSim. Phases carry ramp, cancellation,
        grace, and drain; continuations receive FIFO priority.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={600} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Same workload ID">
          Transport selection does not create separate HTTP, gRPC, and DynoSim workload registrations.
        </Callout>
        <Callout tone="info" title="Accuracy">
          Static accuracy is configuration on the scheduled path, with canonical Python evaluators behind a subprocess
          seam.
        </Callout>
        <Callout tone="success" title="Local hot path">
          Each worker co-locates scheduler, prepared endpoints, transport, and observers without per-token cross-thread
          locking.
        </Callout>
      </Grid>

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
