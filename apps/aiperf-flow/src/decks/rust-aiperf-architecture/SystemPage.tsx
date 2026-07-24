/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, RoundNode, DbNode, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of AIPerf's product landscape: one native binary, re-exec'd per run,
// dispatching load to a target and emitting typed artifacts. Each spoke is one beat of that story.

/** SystemView: AIPerf's product landscape — one native binary, re-exec'd per run. */
export function SystemPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One binary, two roles">
        AIPerf's Rust product is one native binary with two roles: the entry point re-execs itself as{" "}
        <code>aiperf --execute</code> for each benchmark. The execution child owns load dispatch; the mock server is
        only an independently launched test target.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · ONE BINARY",
          title: "What runs the benchmark?",
          body: "One native binary, re-exec'd as --execute for each run.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Author and launch",
            diagram: (
              <Diagram>
                <NodeChip>CONFIG</NodeChip>
                <MiniArrow />
                <NodeChip accent>aiperf</NodeChip>
              </Diagram>
            ),
            children: "Config v2 + CLI flags resolve into one launch of the native aiperf-cli entry point.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Re-exec per run",
            diagram: (
              <Diagram>
                <NodeChip>aiperf</NodeChip>
                <MiniArrow />
                <NodeChip accent>--execute</NodeChip>
              </Diagram>
            ),
            children: "The same binary re-execs itself over stdio in strict protocol-v2 execute mode.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Cells scale out",
            diagram: (
              <Diagram>
                <RoundNode>1</RoundNode>
                <RoundNode accent>2</RoundNode>
                <RoundNode>n</RoundNode>
                <MiniArrow />
                <NodeChip>velo</NodeChip>
              </Diagram>
            ),
            children: "With cells > 1 the controller fans work across cell processes over the velo transport.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Dispatch target",
            diagram: (
              <Diagram>
                <NodeChip>EXEC</NodeChip>
                <MiniArrow />
                <DbNode accent>HTTP</DbNode>
              </Diagram>
            ),
            children: "Real server, aiperf-mock-server, or in-process Dynamo — same client, only the address changes.",
          },
          {
            accent: "red",
            badge: 5,
            title: "Dispatch owns measurement",
            diagram: (
              <Diagram>
                <NodeChip accent>SINK</NodeChip>
                <MiniArrow />
                <MiniBars heights={[38, 72, 100, 82]} />
              </Diagram>
            ),
            children: "The execution child is the only process that dispatches load and measures each request.",
          },
          {
            accent: "yellow",
            badge: 6,
            title: "Typed artifacts",
            diagram: (
              <Diagram>
                <NodeChip accent>REPORT</NodeChip>
                <MiniArrow />
                <NodeChip>JSON · CSV</NodeChip>
              </Diagram>
            ),
            children: "A typed native-v2 report plus per-record JSON / CSV / Parquet outputs.",
          },
          {
            accent: "cyan",
            badge: 7,
            title: "Network exporters",
            diagram: (
              <Diagram>
                <NodeChip>OTLP</NodeChip>
                <MiniArrow />
                <NodeChip accent>W&B</NodeChip>
              </Diagram>
            ),
            children: "OTLP, MLflow, and W&B sinks stream results to external systems.",
          },
        ]}
      />

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
