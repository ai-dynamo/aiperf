/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow, MiniBars } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of AIPerf's Cargo feature composition: a lean base executable ringed
// by orthogonal feature branches, each statically linked and fail-closed. Ported from FeaturesView.

/** FeaturesView: the executable's feature set defines its implementation universe. */
export function FeaturesPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Feature composition">
        The executable's feature set defines the available implementation universe. The lean CLI remains sibling-free;
        optional features add persistence, scale-out, embedded Python, or Dynamo integration.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · CARGO FEATURES",
          title: "What can this image do?",
          body: "A lean base executable, extended by orthogonal statically-linked features.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Lean base",
            diagram: (
              <Diagram>
                <NodeChip accent>default = []</NodeChip>
                <MiniArrow />
                <NodeChip>runner-protocol</NodeChip>
              </Diagram>
            ),
            children: "aiperf-cli default = [] is the lean entry point; runner-protocol is always on. HTTP · gRPC · scheduled · graph.",
          },
          {
            accent: "green",
            badge: 2,
            title: "parquet",
            diagram: (
              <Diagram>
                <NodeChip accent>parquet</NodeChip>
                <MiniArrow />
                <MiniBars heights={[40, 68, 100, 76]} />
              </Diagram>
            ),
            children: "Columnar datasets and per-record artifacts. Parquet output requires the parquet feature.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "velo",
            diagram: (
              <Diagram>
                <NodeChip accent>velo</NodeChip>
                <MiniArrow />
                <NodeChip>controller · cell</NodeChip>
              </Diagram>
            ),
            children: "Cross-process scale-out: controller · cell · aggregator. cells > 1 requires velo; lean builds reject it.",
          },
          {
            accent: "orange",
            badge: 4,
            title: "dynosim",
            diagram: (
              <Diagram>
                <NodeChip accent>dynosim</NodeChip>
                <MiniArrow />
                <NodeChip>dynamo-full</NodeChip>
              </Diagram>
            ),
            children: "Dynamo mocker integration; dynamo-full adds router · ZMQ · KV · AIC. Needs the sibling checkout.",
          },
          {
            accent: "cyan",
            badge: 5,
            title: "pyo3-embed",
            diagram: (
              <Diagram>
                <NodeChip accent>pyo3-embed</NodeChip>
                <MiniArrow />
                <NodeChip>Python</NodeChip>
              </Diagram>
            ),
            children: "In-process Python delegation; the delegation policy runs embedded or as a Python subprocess.",
          },
          {
            accent: "yellow",
            badge: 6,
            title: "search-pyo3",
            diagram: (
              <Diagram>
                <NodeChip accent>search-pyo3</NodeChip>
                <MiniArrow />
                <NodeChip>scipy · optuna</NodeChip>
              </Diagram>
            ),
            children: "Adaptive-search planners backed by scipy + optuna, layered on the pyo3 embed.",
          },
          {
            accent: "red",
            badge: 7,
            title: "full",
            diagram: (
              <Diagram>
                <NodeChip accent>full</NodeChip>
                <MiniArrow />
                <NodeChip>everything</NodeChip>
              </Diagram>
            ),
            children: "full = dynosim + parquet + velo. Authored capabilities missing from the image fail closed at validation.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Executable features", path: "rust/cli/Cargo.toml" },
          { label: "Library features", path: "rust/aiperf/Cargo.toml" },
          { label: "Capability composition", path: "rust/aiperf/src/runner_protocol/application.rs" },
          { label: "Wheel bundling", path: "Makefile" },
        ]}
      />
    </div>
  );
}
