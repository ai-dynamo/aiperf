/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the ProtocolView page: one child lifecycle across the process boundary.

const nodes: Node[] = [
  bandHeader("b-parent", "Parent process", 0, 0),
  panel("profile", "profile / sweep / search", undefined, 0, 60),
  card("request", "RunnerRequest v2", undefined, "operation + AuthoredRunSpecV2", 280, 60),
  panel("resolve", "exec_bin::resolve", "override or current_exe", 580, 60),
  card("spawn", "spawn child", undefined, "--execute", 840, 60),

  bandHeader("b-child", "Execution child", 0, 220),
  card("stdin", "stdin to EOF", undefined, "strict JSON decode", 0, 280),
  panel("bootstrap", "bootstrap", "protocol=2 · validate|execute", 280, 280),
  card("app", "RunnerApplication", undefined, "frozen distribution universe", 560, 280),
  card("coordinator", "Coordinator", undefined, "validate → prepare", 840, 280),
  card("validate-op", "validate operation", undefined, "side-effect-free result", 300, 420),
  card("execute-op", "execute operation", undefined, "run + commit report", 700, 420),

  bandHeader("b-terminal", "Terminal contract", 0, 560),
  panel("stderr", "stderr", "diagnostics and lifecycle only", 0, 620),
  card("stdout", "stdout", undefined, "exactly one typed JSONL envelope", 300, 620),
  panel("parent-parse", "parent parses terminal", "success · report_path · error", 640, 620),
];

const edges: Edge[] = [
  flow("profile", "request"),
  flow("request", "resolve"),
  flow("resolve", "spawn"),
  flow("spawn", "stdin"),
  flow("stdin", "bootstrap"),
  flow("bootstrap", "app"),
  flow("app", "coordinator"),
  dashed("coordinator", "validate-op"),
  flow("coordinator", "execute-op"),
  flow("execute-op", "stdout"),
  flow("stdout", "parent-parse"),
];

/** ProtocolView: fresh process boundary per benchmark via self re-exec. */
export function ProtocolPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One child lifecycle">
        Each benchmark gets a fresh process boundary without a second product binary: the entry point resolves{" "}
        <code>current_exe()</code>, starts <code>aiperf --execute</code>, writes one protocol-v2 envelope, and waits for
        one terminal JSON line.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={600} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Isolation">
          Signals and panics are contained in the child; the parent remains the presentation shell.
        </Callout>
        <Callout tone="info" title="Capabilities">
          The linked catalog is composed in-process; it is not a public CLI subcommand.
        </Callout>
        <Callout tone="warning" title="Override">
          <code>AIPERF_EXEC_BIN</code> may point development runs at a differently featured executable.
        </Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "Self-exec resolver", path: "rust/cli/src/exec_bin.rs" },
          { label: "Parent protocol client", path: "rust/cli/src/execute.rs" },
          { label: "Execution child", path: "rust/cli/src/execute_mode.rs" },
          { label: "Coordinator", path: "rust/aiperf/src/runner_protocol/coordinator.rs" },
        ]}
      />
    </div>
  );
}
