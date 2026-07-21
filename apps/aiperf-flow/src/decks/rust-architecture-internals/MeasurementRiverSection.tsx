/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 10 — facts become artifacts in one direction. Observer callbacks flow into the
//! worker-local storage policy band, then the post-drain join and commit band, then five
//! artifact output boxes. Ported from `MeasurementRiver` in the canvas source.

import type { Edge, Node } from "@xyflow/react";
import { Stack } from "../../layout/Stack.js";
import { Grid } from "../../layout/Grid.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  headerNode,
  cardNode,
  chipNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

const OBSERVER = ["arrival", "admit", "token", "usage", "terminal"] as const;

function buildNodes(detail: Detail): Node[] {
  const nodes: Node[] = [];
  OBSERVER.forEach((label, index) => {
    nodes.push(chipNode(`obs-${label}`, index * 170, index % 2 === 0 ? 0 : 40, rank(detail) > 0 ? `on_${label}` : label));
  });

  nodes.push(
    headerNode("band-storage", 0, 130, "Worker-local storage policy"),
    cardNode("exact-storage", 60, 180, "Exact storage", "retain rows | exact-fold"),
    cardNode("accumulator", 360, 180, "MetricsAccumulator", "119-row catalog + derived", undefined, "primary"),
    cardNode("sketch-fold", 660, 180, "Sketch fold", "t-digest + drop"),

    headerNode("band-commit", 0, 300, "Post-drain join and commit"),
    cardNode("side-channels", 40, 350, "side channels", "GPU · server · network"),
    cardNode("native-report", 280, 350, "NativeReport", "typed schema v2"),
    cardNode("native-json", 520, 350, "native-v2.json", "durable commit point", undefined, "primary"),
    cardNode("fanout", 760, 350, "fan-out", "exporters"),

    headerNode("band-artifacts", 0, 460, "ARTIFACT OUTPUTS"),
  );

  const artifacts = [
    { id: "art-summary", title: "summary JSON / CSV" },
    { id: "art-perrecord", title: "per-record files" },
    { id: "art-console", title: "console / accuracy" },
    { id: "art-timeslices", title: "timeslices (exact)" },
    { id: "art-otlp", title: "OTLP · MLflow · W&B" },
  ];
  artifacts.forEach((a, index) => {
    nodes.push(cardNode(a.id, index * 190, 510, a.title));
  });
  return nodes;
}

const edges: Edge[] = [
  ...OBSERVER.slice(0, -1).map((label, i) => flowEdge(`e-obs-${i}`, `obs-${label}`, `obs-${OBSERVER[i + 1]}`)),
  flowEdge("e-token-acc", "obs-token", "accumulator"),
  flowEdge("e-exact-acc", "exact-storage", "accumulator"),
  flowEdge("e-sketch-acc", "sketch-fold", "accumulator"),
  flowEdge("e-side-report", "side-channels", "native-report"),
  flowEdge("e-report-json", "native-report", "native-json"),
  flowEdge("e-json-fanout", "native-json", "fanout"),
  flowEdge("e-acc-report", "accumulator", "native-report"),
  flowEdge("e-fanout-otlp", "fanout", "art-otlp"),
  flowEdge("e-json-perrecord", "native-json", "art-perrecord"),
];

/** Section 10 diagram: the observation-to-artifact river and its exact/sketch modes. */
export function MeasurementRiverSection({ detail }: { detail: Detail }): React.JSX.Element {
  return (
    <SectionShell>
      <SectionHeading
        number="10"
        title="Facts become artifacts in one direction"
        subtitle="Transport observations flow into worker-local storage, then worker drain, accumulator merge, side-channel joins, one durable commit, and exporter fan-out."
      />
      <FlowFrame nodes={buildNodes(detail)} edges={edges} height={620} />
      <p className={`text-xs ${inkClassName("tertiary")}`}>
        scheduler arrival → dispatch admit → streaming token callbacks → usage immediately before terminal
      </p>

      <Grid columns="1fr 1fr" gap={16}>
        <Stack gap={8}>
          <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>Exact mode</h3>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Exact retain enables per-record artifacts; retain and exact-fold preserve exact aggregate percentiles and
            timeslices.
          </p>
        </Stack>
        <Stack gap={8}>
          <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>Sketch mode</h3>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Rows fold and drop continuously; counts and extrema remain exact while percentiles become approximate.
          </p>
        </Stack>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          RecordArtifactLane writes per-record sidecars during capture; NativeReport carries aggregate schemas
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "metrics store", path: "rust/runtime/src/metrics_core/store.rs" },
          { label: "report commit", path: "rust/runtime/src/report.rs" },
          { label: "record writers", path: "rust/runtime/src/engine/records.rs" },
          { label: "export plane", path: "rust/runtime/src/export/mod.rs" },
        ]}
      />
    </SectionShell>
  );
}
