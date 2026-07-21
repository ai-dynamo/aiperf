/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Stack } from "../../layout/Stack.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the ProcessesView page: crates and boundaries. Solid arrows are compile-time
// dependencies or self re-exec; dashed arrows are runtime network or optional feature paths.

const nodes: Node[] = [
  bandHeader("b-roles", "Executable process roles", 0, 0),
  card("entry", "aiperf entry point", undefined, "profile · config · chat · validate…", 0, 60),
  card("execute", "aiperf --execute", undefined, "same binary · stdio v2 · isolated child", 300, 60),
  card("mock", "aiperf-mock-server", undefined, "HTTP/SSE · gRPC · TLS · UDS", 620, 60),

  bandHeader("b-libs", "Libraries", 0, 220),
  card("aiperf", "aiperf", undefined, "runtime composition + runner_protocol + 16 absorbed modules", 0, 280),
  card("loadgen", "loadgen-core", undefined, "Dispatchable · RequestSink · RequestObserver", 360, 280),
  panel("pyext", "pyext", "packaging-only pyo3 cdylib", 0, 400),
  panel("e2e", "e2e harness", "product-level integration tests", 360, 400),

  bandHeader("b-external", "External runtime boundaries", 0, 540),
  card("http", "HTTP / gRPC servers", undefined, undefined, 0, 600),
  card("mocker", "Dynamo mocker", undefined, undefined, 300, 600),
  card("pyeval", "Python evaluators", undefined, undefined, 600, 600),
];

const edges: Edge[] = [
  flow("entry", "execute"),
  flow("execute", "aiperf"),
  flow("entry", "aiperf"),
  flow("mock", "aiperf"),
  flow("aiperf", "loadgen"),
  dashed("e2e", "aiperf"),
  dashed("aiperf", "http"),
  dashed("aiperf", "mocker"),
  dashed("aiperf", "pyeval"),
];

/** ProcessesView: crate topology and the compile-time vs runtime boundaries. */
export function ProcessesPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Crates and boundaries">
        Solid arrows are compile-time dependencies or self re-exec. Dashed arrows are runtime network or optional
        feature paths. The large <code>aiperf</code> library absorbs the former multi-crate runtime modules.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={560} />

      <Grid columns="1.2fr 1fr" gap={16}>
        <Stack gap={8}>
          <div className="rounded-lg border border-stroke-secondary px-4 py-3 text-sm shadow-sm">
            <div className="font-semibold">Dependency direction</div>
            <p className="mt-1 text-ink-secondary">
              <code>aiperf-cli</code> → <code>aiperf</code> → <code>loadgen-core</code>. The entry point re-execs the
              current <code>aiperf</code> binary with <code>--execute</code>. <code>aiperf-mock-server</code> →{" "}
              <code>aiperf</code>; execute mode and mock do not depend on each other.
            </p>
          </div>
        </Stack>
        <Callout tone="info" title="Packaging">
          <code>pyext</code> is the wheel's compiled binding target and is not in the execution path. The workspace
          still contains and packages the older <code>aiperf-runner</code> crate, but the default CLI path self
          re-execs.
        </Callout>
      </Grid>

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
