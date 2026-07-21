/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { card, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the SeamsView page: extension internals. Two side-by-side sub-diagrams
// (compile-time extension universe, execution substitution) plus a wide cellular-scaling diagram.

const extensionNodes: Node[] = [
  card("extension", "AIPerfExtension", undefined, "transactional registration", 0, 0),
  card("registry", "AIPerfRegistry", undefined, "frozen once per executable image", 0, 130),
  panel("datasets", "datasets", "loaders + samplers", 300, 0),
  panel("endpoints", "endpoints", "body + response", 300, 100),
  panel("exporters", "exporters", "report sinks", 300, 200),
  panel("transports", "transports", "HTTP · gRPC · DynoSim", 300, 300),
  panel("workloads", "workloads", "scheduled · graph…", 300, 400),
];

const extensionEdges: Edge[] = [
  flow("extension", "registry"),
  flow("registry", "datasets"),
  flow("registry", "endpoints"),
  flow("registry", "exporters"),
  flow("registry", "transports"),
  flow("registry", "workloads"),
];

const substitutionNodes: Node[] = [
  card("executor", "Workload / graph executor", undefined, "transport-neutral orchestration", 0, 180),
  card("clock", "Clock", undefined, "RealClock | SimClock", 320, 60),
  card("requestsink", "RequestSink<R>", undefined, "transport-native R", 320, 300),
  panel("http", "HTTP / SSE", "Hyper", 600, 0),
  panel("grpc", "gRPC", "Tonic", 600, 120),
  panel("dynosim", "DynoSim", "DirectRequest", 600, 240),
  card("observer", "RequestObserver event stream", undefined, undefined, 900, 180),
];

const substitutionEdges: Edge[] = [
  flow("executor", "clock"),
  flow("executor", "requestsink"),
  flow("requestsink", "http"),
  flow("requestsink", "grpc"),
  flow("requestsink", "dynosim"),
  flow("http", "observer"),
  flow("grpc", "observer"),
  flow("dynosim", "observer"),
];

const cellularNodes: Node[] = [
  card("controller", "controller process", undefined, "slice budgets + distribute envelope", 0, 160),
  card("cell0", "cell 0", undefined, "ordinary execute path", 340, 0),
  card("cell1", "cell 1", undefined, "ordinary execute path", 340, 160),
  card("celln", "cell N", undefined, "ordinary execute path", 340, 320),
  card("aggregators", "optional aggregators", undefined, "merge folded stores", 700, 160),
  card("final", "final report", undefined, "controller commit", 1000, 160),
];

const cellularEdges: Edge[] = [
  flow("controller", "cell0"),
  flow("controller", "cell1"),
  flow("controller", "celln"),
  flow("cell0", "aggregators"),
  flow("cell1", "aggregators"),
  flow("celln", "aggregators"),
  flow("aggregators", "final"),
];

/** SeamsView: compile-time composition and execution-path substitution around one run core. */
export function SeamsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Extension internals">
        The architecture stays open in two directions: compile-time product composition at startup, and transport/clock
        substitution on the execution path. Cellular mode scales around the same single-run core.
      </PageIntro>

      <Grid columns="1fr 1fr" gap={16}>
        <div>
          <h3 className="mb-2 text-base font-semibold">Compile-time extension universe</h3>
          <DeckDiagram nodes={extensionNodes} edges={extensionEdges} height={420} />
        </div>
        <div>
          <h3 className="mb-2 text-base font-semibold">Execution substitution</h3>
          <DeckDiagram nodes={substitutionNodes} edges={substitutionEdges} height={420} />
        </div>
      </Grid>

      <Divider />

      <div>
        <h3 className="mb-2 text-base font-semibold">Cellular scaling wraps the same run core</h3>
        <DeckDiagram nodes={cellularNodes} edges={cellularEdges} height={360} />
      </div>

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="No runtime plugin discovery">
          Extensions are statically linked and duplicate names fail registration transactionally.
        </Callout>
        <Callout tone="info" title="No pair matrix">
          Transport and workload registries are independent; workloads resolve an execution factory from the prepared
          transport.
        </Callout>
        <Callout tone="warning" title="Cellular gate">
          Cross-process cells use the opt-in <code>velo</code> feature. Lean builds preserve <code>cells=1</code> and
          reject larger runs.
        </Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "Extension registry", path: "rust/aiperf/src/extensions/mod.rs" },
          { label: "Cell controller", path: "rust/aiperf/src/runner_protocol/cellular_controller.rs" },
          { label: "Observer implementation", path: "rust/loadgen-core/src/observer.rs" },
        ]}
      />
    </div>
  );
}
