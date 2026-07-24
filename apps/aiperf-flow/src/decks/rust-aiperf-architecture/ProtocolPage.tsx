/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of one child lifecycle: the parent resolves current_exe(), spawns
// aiperf --execute, writes one protocol-v2 envelope, and waits for one terminal JSON line. Each
// spoke is one beat of the parent → child → terminal-contract story from the old ProtocolView.

/** ProtocolView: fresh process boundary per benchmark via self re-exec. */
export function ProtocolPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="One child lifecycle">
        Each benchmark gets a fresh process boundary without a second product binary: the entry point resolves{" "}
        <code>current_exe()</code>, starts <code>aiperf --execute</code>, writes one protocol-v2 envelope, and waits for
        one terminal JSON line.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · SELF RE-EXEC",
          title: "How is a run isolated?",
          body: "Parent spawns one --execute child, one envelope in, one terminal line out.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Parent request",
            diagram: (
              <Diagram>
                <NodeChip>profile</NodeChip>
                <MiniArrow />
                <NodeChip accent>RunnerRequest v2</NodeChip>
              </Diagram>
            ),
            children: "profile / sweep / search build a RunnerRequest v2: operation + AuthoredRunSpecV2.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Resolve and spawn",
            diagram: (
              <Diagram>
                <NodeChip>resolve</NodeChip>
                <MiniArrow />
                <NodeChip accent>--execute</NodeChip>
              </Diagram>
            ),
            children: "exec_bin::resolve picks the override or current_exe, then spawns the child with --execute.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "stdin and bootstrap",
            diagram: (
              <Diagram>
                <NodeChip>stdin EOF</NodeChip>
                <MiniArrow />
                <NodeChip accent>protocol=2</NodeChip>
              </Diagram>
            ),
            children: "The child reads stdin to EOF, strict-JSON-decodes, and bootstraps in validate|execute mode.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Frozen application",
            diagram: (
              <Diagram>
                <NodeChip accent>RunnerApplication</NodeChip>
                <MiniArrow />
                <NodeChip>Coordinator</NodeChip>
              </Diagram>
            ),
            children: "RunnerApplication freezes the distribution universe; the Coordinator runs validate → prepare.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "Validate or execute",
            diagram: (
              <Diagram>
                <NodeChip>validate</NodeChip>
                <MiniArrow />
                <NodeChip accent>execute</NodeChip>
              </Diagram>
            ),
            children: "A side-effect-free validate operation, or an execute operation that runs and commits the report.",
          },
          {
            accent: "orange",
            badge: 6,
            title: "stdout envelope",
            diagram: (
              <Diagram>
                <NodeChip>execute</NodeChip>
                <MiniArrow />
                <NodeChip accent>JSONL</NodeChip>
              </Diagram>
            ),
            children: "stdout carries exactly one typed JSONL envelope; stderr is diagnostics and lifecycle only.",
          },
          {
            accent: "red",
            badge: 7,
            title: "Terminal contract",
            diagram: (
              <Diagram>
                <NodeChip>stdout</NodeChip>
                <MiniArrow />
                <NodeChip accent>parent parse</NodeChip>
              </Diagram>
            ),
            children: "The parent parses the terminal line: success · report_path · error, containing child signals and panics.",
          },
        ]}
      />

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
