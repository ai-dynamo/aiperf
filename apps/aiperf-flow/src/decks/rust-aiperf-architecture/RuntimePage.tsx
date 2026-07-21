/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the RuntimeView page: one request end-to-end. Four bands from author/bootstrap
// through validate/prepare, run/dispatch, and reduce/commit.

const nodes: Node[] = [
  bandHeader("b1", "1 · Author and bootstrap", 0, 0),
  panel("config", "Config v2 / flags", undefined, 0, 60),
  card("spec", "AuthoredRunSpecV2", undefined, "serialized to execution-child stdin", 280, 60),
  card("app", "RunnerApplication::stock", undefined, "freeze registries + resolvers + factories", 580, 60),

  bandHeader("b2", "2 · Validate and prepare", 0, 200),
  panel("coordinator", "Coordinator", "resolve IDs; fail closed", 0, 260),
  card("workload-f", "Workload factory", undefined, "scheduled · graph", 280, 260),
  card("transport-f", "Transport factory", undefined, "http · grpc · dynosim", 560, 260),
  card("prepared-op", "Prepared operation", undefined, "one-shot executable", 840, 260),

  bandHeader("b3", "3 · Run phases and dispatch requests", 0, 400),
  card("phase", "Phase runtime", undefined, "warmup → profiling", 0, 460),
  card("driver", "Workload driver", undefined, "scheduled or graph", 280, 460),
  panel("admission", "Admission + pacing", "SlotPool · arrivals · stop", 560, 460),
  card("endpoint", "Prepared endpoint", undefined, "request body + parser", 840, 460),
  card("clock", "Clock", undefined, "RealClock or SimClock", 0, 600),
  card("dispatch", "RequestSink<R>::dispatch", undefined, "HTTP · gRPC · DirectRequest", 340, 600),
  card("observer", "RequestObserver", undefined, "arrival · admit · token · usage · terminal", 700, 600),

  bandHeader("b4", "4 · Reduce, join side channels, commit", 0, 740),
  panel("capture", "Per-worker capture", "records or t-digest sketch", 0, 800),
  card("accumulator", "Metrics accumulator", undefined, "merge once after drain", 300, 800),
  card("sidechannels", "Side channels", undefined, "GPU · server · network", 580, 800),
  card("exporters", "Native exporters", undefined, "commit report + artifacts", 860, 800),
];

const edges: Edge[] = [
  flow("config", "spec"),
  flow("spec", "app"),
  flow("coordinator", "workload-f"),
  flow("workload-f", "transport-f"),
  flow("transport-f", "prepared-op"),
  flow("app", "coordinator"),
  flow("phase", "driver"),
  flow("driver", "admission"),
  flow("admission", "endpoint"),
  dashed("admission", "dispatch"),
  flow("endpoint", "dispatch"),
  flow("dispatch", "observer"),
  flow("observer", "capture"),
  flow("capture", "accumulator"),
  flow("sidechannels", "exporters"),
  flow("accumulator", "exporters"),
];

/** RuntimeView: the one-run hot path from frozen registries to committed report. */
export function RuntimePage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One request, end to end">
        This is the one-run hot path. Startup uses frozen registries and strict DTOs; request execution then stays on
        transport-native request types and local observer graphs. The final commit writes{" "}
        <code>native-v2.json</code> plus compatibility exports.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={640} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Startup vs hot path">
          Type erasure and registry lookups happen during validation/preparation, not per token.
        </Callout>
        <Callout tone="info" title="Timing authority">
          Arrival, admission, token, cancellation, and phase timing come from the injected <code>Clock</code>.
        </Callout>
        <Callout tone="success" title="Lock avoidance">
          Worker-local <code>Rc/RefCell</code> observer state avoids an <code>Arc/Mutex</code> on each token.
        </Callout>
      </Grid>

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
