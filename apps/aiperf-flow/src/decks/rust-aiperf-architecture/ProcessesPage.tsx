/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of the crate topology: compile-time dependency direction
// (aiperf-cli → aiperf → loadgen-core) plus the runtime network / optional-feature boundaries.
// Each spoke is one executable role, library, or external boundary from the old ProcessesView.

/** ProcessesView: crate topology and the compile-time vs runtime boundaries. */
export function ProcessesPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Crates and boundaries">
        Solid arrows are compile-time dependencies or self re-exec. Dashed arrows are runtime network or optional
        feature paths. The large <code>aiperf</code> library absorbs the former multi-crate runtime modules.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · CRATE TOPOLOGY",
          title: "How is the workspace wired?",
          body: "aiperf-cli → aiperf → loadgen-core, with dashed runtime boundaries.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "aiperf entry point",
            diagram: (
              <Diagram>
                <NodeChip>CONFIG</NodeChip>
                <MiniArrow />
                <NodeChip accent>aiperf</NodeChip>
              </Diagram>
            ),
            children: "The aiperf-cli binary: profile · config · chat · validate and the other native commands.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "aiperf --execute",
            diagram: (
              <Diagram>
                <NodeChip>aiperf</NodeChip>
                <MiniArrow />
                <NodeChip accent>--execute</NodeChip>
              </Diagram>
            ),
            children: "The same binary, re-exec'd as an isolated child over stdio protocol v2.",
          },
          {
            accent: "green",
            badge: 3,
            title: "aiperf library",
            diagram: (
              <Diagram>
                <NodeChip accent>aiperf</NodeChip>
                <MiniArrow />
                <NodeChip>16 modules</NodeChip>
              </Diagram>
            ),
            children: "Runtime composition + runner_protocol, absorbing the 16 former multi-crate runtime modules.",
          },
          {
            accent: "purple",
            badge: 4,
            title: "loadgen-core",
            diagram: (
              <Diagram>
                <NodeChip accent>Dispatchable</NodeChip>
                <MiniArrow />
                <NodeChip>RequestSink</NodeChip>
              </Diagram>
            ),
            children: "The dispatch seam: Dispatchable · RequestSink · RequestObserver, depended on by aiperf.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "aiperf-mock-server",
            diagram: (
              <Diagram>
                <NodeChip>HTTP/SSE</NodeChip>
                <MiniArrow />
                <NodeChip accent>gRPC · TLS · UDS</NodeChip>
              </Diagram>
            ),
            children: "Standalone inference target → aiperf; execute mode and mock do not depend on each other.",
          },
          {
            accent: "orange",
            badge: 6,
            title: "Packaging and tests",
            diagram: (
              <Diagram>
                <NodeChip accent>e2e</NodeChip>
              </Diagram>
            ),
            children: "e2e drives product integration.",
          },
          {
            accent: "red",
            badge: 7,
            title: "External boundaries",
            diagram: (
              <Diagram>
                <NodeChip accent>HTTP/gRPC</NodeChip>
                <MiniArrow />
                <NodeChip>mocker · pyeval</NodeChip>
              </Diagram>
            ),
            children: "Dashed runtime edges reach HTTP/gRPC servers, the Dynamo mocker, and Python evaluators.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "aiperf modules", path: "rust/aiperf/src/lib.rs" },
          { label: "Executable features", path: "rust/cli/Cargo.toml" },
          { label: "Library features", path: "rust/aiperf/Cargo.toml" },
        ]}
      />
    </div>
  );
}
