/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hero — title, tagline, the one-directional Config v2 → aiperf → --execute → runtime → report
//! pipeline spine, three framing callouts, and the global detail-level control. Ported from
//! `Hero` in `docs/canvases/rust-architecture-internals.canvas.tsx`.

import type { Edge, Node } from "@xyflow/react";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SourcesRow,
  FlowFrame,
  cardNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

export type HeroSectionProps = {
  detail: Detail;
  onDetailChange: (detail: Detail) => void;
};

function buildNodes(detail: Detail): Node[] {
  const engineering = rank(detail) > 0;
  return [
    cardNode("config", 0, 80, "Config v2", "flags + YAML"),
    cardNode("aiperf", 200, 80, "aiperf", "native parent", engineering ? "aiperf-cli" : undefined),
    cardNode("execute", 420, 80, "--execute", "OS child process", engineering ? "self binary by default" : undefined),
    cardNode("runtime", 640, 80, "runtime", "schedule + dispatch", engineering ? "aiperf-runtime" : undefined),
    cardNode("report", 860, 80, "report", "commit + export"),
  ];
}

const edges: Edge[] = [
  flowEdge("e-config-aiperf", "config", "aiperf", { label: "author" }),
  flowEdge("e-aiperf-execute", "aiperf", "execute", { label: "validate + prepare" }),
  flowEdge("e-execute-runtime", "execute", "runtime", { label: "execute + observe" }),
  flowEdge("e-runtime-report", "runtime", "report", { label: "persist" }),
];

/** Opening section: the single directional spine of one native run, with the detail control. */
export function HeroSection({ detail, onDetailChange }: HeroSectionProps): React.JSX.Element {
  return (
    <Stack gap={18}>
      <Row gap={16} align="center" justify="space-between" wrap>
        <Stack gap={5} className="min-w-[300px]">
          <h1 className={`text-3xl font-bold ${inkClassName("primary")}`}>Inside Rust AIPerf</h1>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            A continuous journey through one native run—from authored Config v2 to worker-local
            transport, measurement, and final artifacts.
          </p>
        </Stack>
        <Segmented
          ariaLabel="Detail level"
          value={detail}
          onChange={onDetailChange}
          options={[
            { id: "orientation", label: "Orientation" },
            { id: "engineering", label: "Engineering" },
            { id: "source", label: "Source" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(detail)} edges={edges} height={300} />

      <p className={`text-center text-xs ${inkClassName("tertiary")}`}>
        Config v2 → aiperf-cli → aiperf-runtime → loadgen-core → artifacts
      </p>

      <Grid columns="1.3fr 1fr 1fr" gap={14}>
        <Callout tone="info" title="One product binary">
          The human entry point and execution engine are two modes of the same <Code inline>aiperf</Code> executable.
        </Callout>
        <Callout tone="info" title="Small hot-path core">
          <Code inline>loadgen-core</Code> owns <Code inline>Dispatchable</Code>, <Code inline>RequestSink</Code>,{" "}
          <Code inline>RequestObserver</Code>, and the trace collector.
        </Callout>
        <Callout tone="warning" title="Standalone mock target">
          <Code inline>aiperf-mock-server</Code> listens as its own HTTP/gRPC process; tests and operators launch it
          before profiling.
        </Callout>
      </Grid>

      <SourcesRow
        detail={detail}
        paths={[
          { label: "workspace", path: "Cargo.toml" },
          { label: "CLI entry", path: "rust/cli/src/main.rs" },
          { label: "runtime root", path: "rust/runtime/src/lib.rs" },
        ]}
      />
    </Stack>
  );
}
