/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 12 — the architecture in seven concrete mechanisms. A numbered list of implemented
//! types, call paths, and persistence boundaries, plus a closing construction callout. Ported
//! from `Mechanisms` in the canvas source.

import clsx from "clsx";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName, surfaceClassName, strokeClassName } from "../../theme/tokens.js";
import { SectionHeading, SourcesRow, SectionShell, type Detail } from "./parts.js";

const MECHANISMS: Array<[string, string]> = [
  ["Self re-exec", "The profile parent starts aiperf --execute and writes one BenchmarkRun to child stdin."],
  ["Frozen composition", "Factories register at startup, then the application becomes immutable."],
  ["Clock and observers", "Transport sinks hold an injected Clock and emit RequestObserver lifecycle callbacks."],
  ["Worker locality", "Scheduling, endpoint preparation, transport, and metrics stay co-located."],
  ["Prepared selection", "Coordinator resolves transport and workload IDs, then prepares one executable operation."],
  [
    "Shared prepared reduction",
    "Prepared HTTP endpoint dispatch and gRPC dispatch call reduce_parsed_response and use WorkerMeasurement.",
  ],
  ["Report persistence", "Side channels join before the atomic native-v2.json write; exporter registry execution follows."],
];

/** Section 12: the closing seven-mechanism summary of the architecture. */
export function MechanismsSection({ detail }: { detail: Detail }): React.JSX.Element {
  return (
    <SectionShell>
      <SectionHeading
        number="12"
        title="The architecture in seven concrete mechanisms"
        subtitle="Each statement below names an implemented type, call path, or persistence boundary."
      />
      <div className={clsx("border-l-4 pl-4", "border-l-category-blue")}>
        <Grid columns="1fr 1fr" gap={16}>
          {MECHANISMS.map(([title, body], index) => (
            <Row key={title} gap={10} align="start">
              <span
                className={clsx(
                  "shrink-0 rounded-md border px-2 py-0.5 text-xs font-bold shadow-sm",
                  surfaceClassName("elevated"),
                  strokeClassName("secondary"),
                  inkClassName("secondary"),
                )}
              >
                {String(index + 1).padStart(2, "0")}
              </span>
              <Stack gap={3}>
                <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</span>
                <span className={`text-sm ${inkClassName("secondary")}`}>{body}</span>
              </Stack>
            </Row>
          ))}
        </Grid>
      </div>
      <Callout tone="info" title="Startup and worker construction">
        <Code inline>AIPerfRegistryFactory::build()</Code> freezes transports and workloads at startup;{" "}
        <Code inline>ExecutionSinkBuilder::build_sink()</Code> constructs each worker-local sink.
      </Callout>
      <SourcesRow
        detail={detail}
        paths={[
          { label: "application.rs", path: "rust/runtime/src/engine/application.rs" },
          { label: "sink builder", path: "rust/runtime/src/engine/sink.rs" },
        ]}
      />
    </SectionShell>
  );
}
