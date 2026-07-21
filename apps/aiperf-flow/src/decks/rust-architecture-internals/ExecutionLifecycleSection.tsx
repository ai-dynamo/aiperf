/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 02 — coordinator stages surround a separate phase clock. The argv-selected
//! operation (validate | execute) enters one validation chain; execute continues into the
//! Clock-driven execution band. Ported from `ExecutionLifecycle` in the canvas source.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Legend } from "../../prose/Legend.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  headerNode,
  cardNode,
  panelNode,
  flowEdge,
  plainEdge,
  rank,
  type Detail,
} from "./parts.js";

type Operation = "validate" | "execute";

function buildNodes(operation: Operation): Node[] {
  const executes = operation === "execute";
  const nodes: Node[] = [
    headerNode("band-stdio", 0, 0, "Stdio boundary"),
    cardNode("argv", 30, 40, "argv operation", executes ? "OperationV2::Execute" : "OperationV2::Validate"),
    cardNode("stdin", 30, 130, "stdin to EOF", "BenchmarkRunWireV2"),
    cardNode("stderr", 30, 220, "stderr", "live diagnostics"),
    cardNode("stdout", 30, 310, "stdout", executes ? "run_terminal" : "run_validation"),
    panelNode("exit", 30, 400, "process exit code"),

    headerNode("band-coord", 240, 0, "Coordinator envelope stages"),
    cardNode("stage-outer", 240, 60, "outer + authored", "protocol / serde"),
    cardNode("stage-selection", 420, 60, "selection", "transport + workload"),
    cardNode("stage-validation", 600, 60, "run validation", "profiles + sidecars"),
    cardNode(
      "stage-final",
      780,
      60,
      executes ? "prepare" : "return",
      executes ? "PreparedRunnerOperation" : "deferred_checks",
    ),

    panelNode("failure-stages", 420, 470, "FAILURE STAGES", "protocol · validation · preparation · execution · reporting"),
  ];

  if (executes) {
    nodes.push(
      headerNode("band-clock", 240, 200, "Execution clock · only for --execute"),
      cardNode("validate-plan", 240, 250, "validate_plan", "second execution gate"),
      cardNode("clock-drive", 420, 250, "clock.drive", "RealClock | SimClock"),
      cardNode("phase-orch", 600, 250, "phase orchestrator", "scheduled | graph"),
      cardNode("run-outcome", 780, 250, "run outcome", "uncommitted report"),
    );
  }
  return nodes;
}

function buildEdges(operation: Operation): Edge[] {
  const executes = operation === "execute";
  const edges: Edge[] = [
    plainEdge("e-argv-stdin", "argv", "stdin"),
    plainEdge("e-stdout-exit", "stdout", "exit"),
    flowEdge("e-stdin-outer", "stdin", "stage-outer", { speed: "slow" }),
    flowEdge("e-outer-selection", "stage-outer", "stage-selection"),
    flowEdge("e-selection-validation", "stage-selection", "stage-validation"),
    flowEdge("e-validation-final", "stage-validation", "stage-final"),
  ];
  if (executes) {
    edges.push(
      flowEdge("e-final-plan", "stage-final", "validate-plan"),
      flowEdge("e-plan-clock", "validate-plan", "clock-drive"),
      flowEdge("e-clock-phase", "clock-drive", "phase-orch"),
      flowEdge("e-phase-outcome", "phase-orch", "run-outcome"),
      flowEdge("e-outcome-stdout", "run-outcome", "stdout", { speed: "slow" }),
    );
  } else {
    edges.push(flowEdge("e-final-stdout", "stage-final", "stdout", { speed: "slow" }));
  }
  return edges;
}

/** Section 02 diagram + the five Clock-driven lifecycle stages and failure-stage footer. */
export function ExecutionLifecycleSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [operation, setOperation] = useState<Operation>("execute");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="02"
          title="Coordinator stages surround a separate phase clock"
          subtitle="The argv-selected operation enters one validation chain; execute continues into preparation, Clock-driven phases, persistence, and a terminal envelope."
        />
        <Segmented
          ariaLabel="Operation"
          value={operation}
          onChange={setOperation}
          options={[
            { id: "validate", label: "--validate" },
            { id: "execute", label: "--execute" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(operation)} edges={buildEdges(operation)} height={520} />

      <div>
        <p className={`mb-2 text-xs font-bold uppercase tracking-wide ${inkClassName("secondary")}`}>
          Lifecycle stages
        </p>
        <Legend
          entries={[
            { color: "blue", label: "sending" },
            { color: "green", label: "grace return" },
            { color: "orange", label: "cancel inflight" },
            { color: "purple", label: "drain" },
            { color: "red", label: "force" },
          ]}
        />
        <p className={`mt-2 text-xs ${inkClassName("tertiary")}`}>
          cancellation during sending can skip grace · seamless non-final phases may drain in background
        </p>
      </div>

      <Grid columns={3} gap={12}>
        <Callout tone="info" title="Argv-selected operation">
          The child derives validate versus execute from argv, then wraps the bare run in <Code inline>EnvelopeV2</Code>.
        </Callout>
        <Callout tone="info" title="Two validation layers">
          Static coordinator validation runs before <Code inline>validate_plan</Code> inside the execution driver.
        </Callout>
        <Callout tone="warning" title="Commit ordering">
          The report file is written before exporters; the optional commit hook runs after the exporter registry completes.
        </Callout>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          <Code inline>ClockPhaseRunner::finish_returning</Code> distinguishes grace timeout, cancellation, successful
          drain, and forced completion. These phase outcomes are separate from protocol failure stages.
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "coordinator", path: "rust/runtime/src/engine/coordinator.rs" },
          { label: "execution driver", path: "rust/runtime/src/engine/execute.rs" },
          { label: "phase runner", path: "rust/runtime/src/timing/phase/runner.rs" },
          { label: "stdio mode", path: "rust/cli/src/execute_mode.rs" },
        ]}
      />
    </SectionShell>
  );
}
