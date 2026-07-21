/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the FeaturesView page: feature composition and build variants.

const nodes: Node[] = [
  bandHeader("b-base", "Base executable", 0, 0),
  card("cli", "aiperf-cli default = []", undefined, "lean entry point + execution child", 0, 60),
  card("runnerproto", "runner-protocol", undefined, "always enabled on aiperf dependency", 320, 60),
  panel("baseexec", "base execution", "HTTP · gRPC · scheduled · graph", 640, 60),

  bandHeader("b-branches", "Orthogonal feature branches", 0, 200),
  card("parquet", "parquet", undefined, "columnar datasets + artifacts", 0, 260),
  card("velo", "velo", undefined, "controller · cell · aggregator", 240, 260),
  card("dynosim", "dynosim", undefined, "Dynamo mocker sibling checkout", 480, 260),
  card("pyo3", "pyo3-embed", undefined, "in-process Python delegation", 720, 260),
  panel("searchpyo3", "search-pyo3", "scipy + optuna planners", 0, 380),
  panel("dynamofull", "dynamo-full", "router · ZMQ · KV · AIC", 280, 380),
  card("full", "full", undefined, "dynosim + parquet + velo", 560, 380),

  bandHeader("b-capability", "Runtime capability result", 0, 520),
  panel("transportcat", "transport catalog", "http · grpc · optional dynosim", 0, 580),
  panel("cellpolicy", "cell count policy", "cells > 1 requires velo", 280, 580),
  panel("artifactpolicy", "artifact policy", "Parquet requires parquet", 560, 580),
  panel("delegationpolicy", "delegation policy", "embedded or Python subprocess", 840, 580),

  bandHeader("b-packaging", "Packaging transition", 0, 720),
  card("wheel", "wheel entry point", undefined, "interned aiperf-native", 0, 780),
  panel("legacy", "legacy packaged runner", "still present; default CLI self re-execs", 320, 780),
];

const edges: Edge[] = [
  flow("cli", "runnerproto"),
  flow("runnerproto", "baseexec"),
  dashed("pyo3", "searchpyo3"),
  dashed("dynosim", "dynamofull"),
  flow("dynosim", "full"),
  dashed("wheel", "legacy"),
];

/** FeaturesView: the executable's feature set defines its implementation universe. */
export function FeaturesPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Feature composition">
        The executable's feature set defines the available implementation universe. The lean CLI remains sibling-free;
        optional features add persistence, scale-out, embedded Python, or Dynamo integration.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={620} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Fail closed">
          Authored transports, artifacts, or cells unavailable in the current image are rejected during validation.
        </Callout>
        <Callout tone="info" title="No runtime discovery">
          Capabilities describe statically linked factories; enabling a config value cannot load missing code.
        </Callout>
        <Callout tone="warning" title="DynoSim dependency">
          DynoSim and Dynamo-full builds require the sibling <code>dynamo-aiperf-native</code> checkout.
        </Callout>
      </Grid>

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
