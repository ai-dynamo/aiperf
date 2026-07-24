/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import { Stack } from "../../layout/Stack.js";
import { Callout } from "../../prose/Callout.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `LaunchPage`: the four-step
//! preflight chain (project -> registry -> re-exec -> handle_v2) with a single reject branch off
//! the registry gate.

function nodes(maint: boolean): Node[] {
  return [
    {
      id: "project",
      type: "card",
      position: { x: 0, y: 0 },
      data: { title: "project", subtitle: maint ? "load.rs → BenchmarkRun" : "native Config v2" },
    },
    {
      id: "registry",
      type: "card",
      position: { x: 260, y: 0 },
      data: { title: "registry", subtitle: maint ? "dynosim_offline registered" : "transport exists?" },
    },
    {
      id: "re-exec",
      type: "card",
      position: { x: 520, y: 0 },
      data: { title: "re-exec", subtitle: maint ? "aiperf --execute stdin" : "same binary child" },
    },
    {
      id: "handle-v2",
      type: "card",
      position: { x: 780, y: 0 },
      data: { title: "handle_v2", subtitle: maint ? "offline_execution" : "run + report" },
    },
    {
      id: "reject",
      type: "panel",
      position: { x: 260, y: 150 },
      data: { title: "reject", detail: maint ? "transport unregistered" : "unsupported", strokeRole: "secondary" },
    },
  ];
}

const edges: Edge[] = [
  { id: "e-project-registry", source: "project", target: "registry", type: "flow" },
  { id: "e-registry-reexec", source: "registry", target: "re-exec", type: "flow" },
  { id: "e-reexec-handle", source: "re-exec", target: "handle-v2", type: "flow" },
  { id: "e-registry-reject", source: "registry", target: "reject" },
];

/**
 * Launch & preflight page.
 *
 * Ports `LaunchPage` from `docs/canvases/dynosim-offline-flow.canvas.tsx`: the native CLI
 * projects one protocol-v2 execute envelope and re-execs itself in `--execute` mode. The frozen
 * `RunnerApplication` registry must already include `dynosim_offline` (feature-gated at compile
 * time). Fail-closed, no fallback.
 */
export function LaunchPage({ level }: { level: Level }): React.JSX.Element {
  const maint = atLeast(level, "maintainer");
  const dev = atLeast(level, "developer");
  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Launch & preflight</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The native CLI projects one protocol-v2 execute envelope and re-execs itself in{" "}
          <strong>--execute</strong> mode. The frozen <strong>RunnerApplication</strong> registry
          must already include <strong>dynosim_offline</strong> (feature-gated at compile time).
          Fail-closed, no fallback.
        </p>
      </div>

      <AutoLayoutFlow key={String(maint)} nodes={nodes(maint)} edges={edges} layout={{ direction: "RIGHT" }} height={260} />

      {dev && (
        <Callout tone="warning" title="What fails closed">
          Base build without the <strong>dynosim</strong> Cargo feature (transport absent from
          registry) · non-v2 envelope · authored <strong>required_features</strong> the linked
          image wasn&apos;t compiled with · unregistered transport or workload id.
        </Callout>
      )}
    </Stack>
  );
}
