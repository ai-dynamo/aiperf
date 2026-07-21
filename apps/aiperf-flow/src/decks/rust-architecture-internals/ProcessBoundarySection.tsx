/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 01 — one run crosses one child process boundary. Parent process
//! (profile → BenchmarkRun → spawn → wait) over the fresh execution child
//! (stdin → Application → execute → stdout), with a typed failure path.
//! Ported from `ProcessBoundary` in the canvas source.

import type { Edge, Node } from "@xyflow/react";
import { Stack } from "../../layout/Stack.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Divider } from "../../layout/Divider.js";
import { inkClassName } from "../../theme/tokens.js";
import {
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

function buildNodes(detail: Detail): Node[] {
  const engineering = rank(detail) > 0;
  return [
    headerNode("band-parent", 0, 0, "Parent process"),
    cardNode("profile", 40, 60, "profile", "flags / config"),
    cardNode(
      "benchmarkrun",
      240,
      60,
      "BenchmarkRun",
      engineering ? "CLI model" : "one run",
      engineering ? "strict child wire decode" : undefined,
    ),
    cardNode("spawn", 440, 60, "spawn", "override | self"),
    cardNode("wait", 620, 60, "wait", "terminal"),

    headerNode("band-child", 0, 200, "Fresh execution child"),
    cardNode("stdin", 40, 260, "stdin to EOF", "bare BenchmarkRun", "default --execute"),
    cardNode("application", 240, 260, "Application", "handle protocol v2"),
    cardNode("execute", 440, 260, "execute", "run + commit"),
    cardNode("stdout", 620, 260, "stdout", "one JSONL"),

    panelNode("failure", 240, 400, "typed failure", "protocol · validation · preparation · execution · reporting", "primary"),
  ];
}

const edges: Edge[] = [
  flowEdge("e-profile-run", "profile", "benchmarkrun"),
  flowEdge("e-run-spawn", "benchmarkrun", "spawn"),
  flowEdge("e-spawn-wait", "spawn", "wait"),
  flowEdge("e-spawn-stdin", "spawn", "stdin", { speed: "slow" }),
  flowEdge("e-stdin-app", "stdin", "application"),
  flowEdge("e-app-execute", "application", "execute"),
  flowEdge("e-execute-stdout", "execute", "stdout"),
  flowEdge("e-stdout-wait", "stdout", "wait", { speed: "slow" }),
  plainEdge("e-app-failure", "application", "failure"),
];

/** Section 01 diagram + narrative on the single child-process boundary each run crosses. */
export function ProcessBoundarySection({ detail }: { detail: Detail }): React.JSX.Element {
  return (
    <SectionShell>
      <SectionHeading
        number="01"
        title="One run crosses one child process boundary"
        subtitle="Each profile run starts an aiperf --execute child and exchanges one request and one terminal envelope over stdio."
      />
      <Grid columns="1.35fr .65fr" gap={18} align="stretch">
        <FlowFrame nodes={buildNodes(detail)} edges={edges} height={440} />
        <Stack gap={13}>
          <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>
            Profile spawns one aiperf --execute child per run
          </h3>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            The parent forwards SIGINT to the child PID; child panics become typed protocol failures on stdout.
          </p>
          <Divider />
          <p className={`text-sm font-semibold ${inkClassName("primary")}`}>Channel discipline</p>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            For default <Code inline>--execute</Code>, <Code inline>stdin</Code> carries one request,{" "}
            <Code inline>stderr</Code> carries live diagnostics, and <Code inline>stdout</Code> carries exactly one
            terminal envelope.
          </p>
          {rank(detail) > 0 && (
            <Callout tone="info" title="Before clap">
              Internal modes are intercepted in <Code inline>main.rs</Code> before ordinary command parsing.
            </Callout>
          )}
        </Stack>
      </Grid>
      <SourcesRow
        detail={detail}
        paths={[
          { label: "main.rs", path: "rust/cli/src/main.rs" },
          { label: "execute.rs", path: "rust/cli/src/execute.rs" },
          { label: "execute_mode.rs", path: "rust/cli/src/execute_mode.rs" },
        ]}
      />
    </SectionShell>
  );
}
