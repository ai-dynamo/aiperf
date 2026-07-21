/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 03 — a frozen universe, composed once. The AIPerfRegistry hub with seven
//! satellite factory registries and the COORDINATOR resolution node. Ported from
//! `CompositionConstellation` in the canvas source.

import type { Edge, Node } from "@xyflow/react";
import { Stack } from "../../layout/Stack.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  cardNode,
  panelNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

const SATELLITES: Array<{ id: string; x: number; y: number; title: string }> = [
  { id: "endpoints", x: 40, y: 20, title: "endpoints" },
  { id: "dataset-formats", x: 300, y: 0, title: "dataset formats" },
  { id: "samplers", x: 560, y: 20, title: "samplers" },
  { id: "transports", x: 0, y: 200, title: "transports" },
  { id: "workloads", x: 600, y: 200, title: "workloads" },
  { id: "exporters", x: 120, y: 360, title: "exporters" },
  { id: "actuators", x: 480, y: 360, title: "actuators" },
];

function buildNodes(detail: Detail): Node[] {
  const nodes: Node[] = [
    cardNode(
      "registry",
      300,
      180,
      "AIPerfRegistry",
      "transactional → frozen",
      rank(detail) > 1 ? "BuiltinAIPerfRegistryFactory" : undefined,
      "primary",
    ),
    panelNode("coordinator", 300, 300, "COORDINATOR", "validate* → prepare → execute", "primary"),
  ];
  for (const sat of SATELLITES) {
    nodes.push(cardNode(sat.id, sat.x, sat.y, sat.title, rank(detail) > 0 ? "factory registry" : undefined));
  }
  return nodes;
}

const edges: Edge[] = [
  ...SATELLITES.map((sat) => flowEdge(`e-${sat.id}-registry`, sat.id, "registry")),
  flowEdge("e-registry-coordinator", "registry", "coordinator"),
];

/** Section 03 diagram: the frozen registry hub and its satellite factory registries. */
export function CompositionConstellationSection({ detail }: { detail: Detail }): React.JSX.Element {
  return (
    <SectionShell>
      <SectionHeading
        number="03"
        title="A frozen universe, composed once"
        subtitle="Factories register transactionally at startup; authored IDs resolve against frozen registries during validation and preparation."
      />
      <Grid columns=".9fr 1.1fr" gap={20} align="stretch">
        <Stack gap={12}>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            <Code inline>Application::stock()</Code> builds the linked implementation universe once, owns its
            coordinator, and routes each protocol-v2 envelope through it.
          </p>
          <Callout tone="success" title="Prepared execution selection">
            The coordinator resolves transport and workload IDs from frozen registries, then workload preparation
            receives the validated transport configuration.
          </Callout>
          <Callout tone="info" title="Registry validation">
            Duplicate registration returns an extension error; unknown authored IDs return a validation diagnostic
            before preparation.
          </Callout>
          {rank(detail) > 0 && (
            <p className={`text-sm ${inkClassName("tertiary")}`}>
              Each worker calls <Code inline>PreparedEndpointTableFactory::prepare_worker()</Code> before dispatch.
            </p>
          )}
        </Stack>
        <FlowFrame nodes={buildNodes(detail)} edges={edges} height={460} />
      </Grid>
      <SourcesRow
        detail={detail}
        paths={[
          { label: "application.rs", path: "rust/runtime/src/engine/application.rs" },
          { label: "registry.rs", path: "rust/runtime/src/engine/registry.rs" },
          { label: "extensions", path: "rust/runtime/src/extensions/mod.rs" },
        ]}
      />
    </SectionShell>
  );
}
