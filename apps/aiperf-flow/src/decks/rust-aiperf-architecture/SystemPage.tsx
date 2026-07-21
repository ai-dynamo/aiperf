/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the SystemView page of the Rust AIPerf architecture Cursor canvas. Four bands:
// author/launch, execute-one-run, dispatch target, and artifacts/integrations.

const nodes: Node[] = [
  bandHeader("b-author", "Author and launch", 0, 0),
  panel("user", "User / automation", "Config v2 + CLI flags", 0, 60),
  card("aiperf", "aiperf", undefined, "native aiperf-cli entry point", 300, 60),
  panel("peripheral", "Peripheral commands", "native or delegated to Python", 620, 60),

  bandHeader("b-execute", "Execute one run", 0, 200),
  card("execute", "aiperf --execute", undefined, "same binary re-exec · strict protocol v2", 300, 260),
  card("cell", "aiperf --cell", undefined, "optional cells > 1 over velo", 620, 260),

  bandHeader("b-dispatch", "Dispatch target", 0, 420),
  card("real", "Real inference server", undefined, "OpenAI · Anthropic · KServe · Riva", 0, 480),
  card("mock", "aiperf-mock-server", undefined, "standalone online test target", 300, 480),
  card("dyno", "Dynamo SteppableReplay", undefined, "in-process dynosim feature", 620, 480),

  bandHeader("b-artifacts", "Artifacts and integrations", 0, 640),
  card("report", "native-v2 report", undefined, "typed schema", 0, 700),
  panel("files", "JSON / CSV / Parquet", "per-record + summary", 300, 700),
  panel("integrations", "OTLP · MLflow · W&B", "network exporters", 620, 700),
];

const edges: Edge[] = [
  flow("user", "aiperf"),
  dashed("aiperf", "peripheral"),
  flow("aiperf", "execute"),
  dashed("execute", "cell"),
  flow("execute", "real"),
  flow("execute", "mock"),
  dashed("execute", "dyno"),
  flow("execute", "report"),
  flow("report", "files"),
  dashed("files", "integrations"),
];

/** SystemView: AIPerf's product landscape — one native binary, re-exec'd per run. */
export function SystemPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One binary, two roles">
        AIPerf's Rust product is one native binary with two roles: the entry point re-execs itself as{" "}
        <code>aiperf --execute</code> for each benchmark. The execution child owns load dispatch; the mock server is
        only an independently launched test target.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={620} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Product boundary">
          The entry-point process authors and launches. The same <code>aiperf</code> binary, re-exec'd in internal
          execute mode, is the only process that dispatches benchmark load.
        </Callout>
        <Callout tone="info" title="Same online path">
          Real and mock online runs use the same HTTP/gRPC clients; only the target address changes.
        </Callout>
        <Callout tone="warning" title="Feature gate">
          DynoSim is compiled through the execution binary's <code>dynosim</code> feature; it is not a separate command.
        </Callout>
      </Grid>

      <EvidenceRow
        items={[
          { label: "CLI routing", path: "rust/cli/src/dispatch.rs" },
          { label: "Execution mode", path: "rust/cli/src/execute_mode.rs" },
          { label: "Workspace crates", path: "Cargo.toml" },
        ]}
      />
    </div>
  );
}
