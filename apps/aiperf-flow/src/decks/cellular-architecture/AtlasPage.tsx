/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Interactive recipe/route explorer over the full cellular atlas. The fidelity recipe
//! (T0/T1/T2/T3) plus work unit, storage, start policy, and roadmap toggle drive `deriveRoute`,
//! which highlights the selected route through the React Flow node/edge graph and reports the
//! derived fidelity model. Ported from `RecipeStrip` / `SystemAtlas` / `FullAtlasPage`.

import { useState } from "react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Select } from "../../prose/Select.js";
import { Toggle } from "../../prose/Toggle.js";
import { Stat } from "../../prose/Stat.js";
import { Callout } from "../../prose/Callout.js";
import { Legend } from "../../prose/Legend.js";
import { Code } from "../../prose/Code.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName } from "../../theme/tokens.js";
import { Inspector } from "./Inspector.js";
import { buildAtlasGraph } from "./atlasGraph.js";
import {
  NODES,
  EDGES,
  deriveRoute,
  type RecipeId,
  type WorkloadKind,
  type StorageMode,
  type StartMode,
  type TruthMode,
} from "./data.js";

const RECIPES: ReadonlyArray<readonly [RecipeId, string]> = [
  ["t0", "T0 Exact"],
  ["t1", "T1 Bounded"],
  ["t2", "T2 Hierarchical"],
  ["t3", "T3 External sink"],
];

function CellCrossSection({
  storage,
  workload,
}: {
  storage: StorageMode;
  workload: WorkloadKind;
}): React.JSX.Element {
  const positions = Array.from({ length: 12 }, (_, index) => index);
  const result =
    storage === "retain"
      ? "Vec<CapturedRecord>"
      : storage === "exact-fold"
        ? "Exact ColumnStore"
        : "TagSketch · t-digest";
  return (
    <Stack gap={10}>
      <Row justify="space-between" align="center">
        <h3 className={`text-sm font-semibold ${inkClassName("primary")}`}>Inside one cell</h3>
        <span className="rounded-none border border-stroke-secondary px-2 py-0.5 text-xs font-semibold text-ink-secondary">
          {workload === "graph" ? "whole traces" : "global request slots"}
        </span>
      </Row>
      <div className="overflow-x-auto">
        <div
          className="items-center gap-1"
          style={{ display: "grid", gridTemplateColumns: "130px repeat(12, minmax(24px, 1fr))", minWidth: 560 }}
        >
          <span className={`text-xs ${inkClassName("tertiary")}`}>Global position</span>
          {positions.map((position) => (
            <span key={position} className={`text-center text-xs font-mono ${inkClassName("secondary")}`}>
              {position}
            </span>
          ))}
          {[0, 1, 2].map((cell) => (
            <div key={cell} style={{ display: "contents" }}>
              <span className={`text-xs font-semibold ${inkClassName("primary")}`}>Cell {cell} owns</span>
              {positions.map((position) => {
                const owns = position % 3 === cell;
                return (
                  <div
                    key={`${cell}-${position}`}
                    className={
                      "h-5 border text-center text-[10px] leading-5 " +
                      (owns
                        ? "border-category-green bg-surface-panel text-ink-primary"
                        : "border-stroke-tertiary text-ink-tertiary")
                    }
                  >
                    {owns ? position : "·"}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <Divider />
      <Row gap={10} align="center" wrap>
        <span className="rounded-none border border-stroke-secondary px-3 py-2 text-sm text-ink-secondary">
          worker 0 · local 0,2,…
        </span>
        <span className={inkClassName("tertiary")}>→</span>
        <span className="rounded-none border border-stroke-secondary px-3 py-2 text-sm text-ink-secondary">
          worker 1 · local 1,3,…
        </span>
        <span className={inkClassName("tertiary")}>→</span>
        <span className="rounded-none border border-category-purple px-3 py-2 text-sm font-semibold text-ink-primary">
          {result}
        </span>
      </Row>
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        Cell ownership is interleaved across the global space; worker ownership is a second partition
        inside the cell. Each worker folds at record completion when exact-fold or sketch mode is active.
      </p>
    </Stack>
  );
}

/**
 * Recipe/route explorer for the cellular atlas. Self-contained; takes no required props. Selecting
 * a fidelity recipe or any selector re-derives the highlighted route and its fidelity readouts.
 */
export function AtlasPage(): React.JSX.Element {
  const [recipe, setRecipe] = useState<RecipeId>("t1");
  const [workload, setWorkload] = useState<WorkloadKind>("scheduled");
  const [storage, setStorage] = useState<StorageMode>("sketch");
  const [start, setStart] = useState<StartMode>("synchronized");
  const [truth, setTruth] = useState<TruthMode>("full");
  const [selectedId, setSelectedId] = useState("controller");

  const route = deriveRoute(recipe, workload, storage, start);

  const selectRecipe = (next: RecipeId) => {
    setRecipe(next);
    if (next === "t0") {
      setStorage("retain");
      setStart("synchronized");
    } else if (next === "t3") {
      setStorage("sketch");
      setStart("synchronized");
      setTruth("full");
    } else {
      setStorage("sketch");
      setStart("synchronized");
    }
  };

  const setRoadmap = (show: boolean) => {
    setTruth(show ? "full" : "built-only");
    if (!show && recipe === "t3") {
      setRecipe("t1");
      setStorage("exact-fold");
      setStart("synchronized");
    }
  };

  const visibleNodeIds = new Set(
    NODES.filter((node) => truth === "full" || node.status === "built").map((node) => node.id),
  );
  const visibleEdgeIds = new Set(
    EDGES.filter(
      (edge) =>
        visibleNodeIds.has(edge.from) &&
        visibleNodeIds.has(edge.to) &&
        (truth === "full" || edge.status === "built"),
    ).map((edge) => edge.id),
  );

  const { nodes, edges } = buildAtlasGraph({
    visibleNodeIds,
    visibleEdgeIds,
    activeNodeIds: route.nodeIds,
    activeEdgeIds: route.edgeIds,
    selectedId,
  });

  const readouts: ReadonlyArray<readonly [string, string]> = [
    ["Per-cell retention", route.memory],
    ["Percentiles", route.percentiles],
    ["Exact aggregates", route.exactAggregates],
    ["Record artifacts", route.artifacts],
    ["Result topology", route.topology],
  ];

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
          One benchmark. Many autonomous cells. One measurement contract.
        </h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          Trace the authored run through deterministic ownership, nested worker shards, fold or
          retain, flat or hierarchical merge, and the final report. Pick a fidelity recipe to
          highlight its route; select any node for source-grounded evidence.
        </p>
      </div>

      <Legend
        entries={[
          { color: "yellow", label: "Control plane" },
          { color: "blue", label: "Data plane" },
          { color: "green", label: "Execution plane" },
          { color: "purple", label: "Results plane" },
        ]}
      />

      <div className="flex flex-col gap-3 border border-stroke-secondary px-4 py-3 lg:flex-row lg:items-end">
        <Stack gap={5} className="lg:flex-1">
          <Eyebrow>Fidelity recipe</Eyebrow>
          <Row gap={6} wrap>
            {RECIPES.map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={recipe === id}
                onClick={() => selectRecipe(id)}
                className={
                  "rounded-none border px-3 py-1.5 text-xs font-semibold transition-colors " +
                  (recipe === id
                    ? "border-accent-primary bg-accent-primary text-white"
                    : "border-stroke-secondary text-ink-secondary")
                }
              >
                {label}
              </button>
            ))}
          </Row>
        </Stack>
        <Select
          label="Work unit"
          value={workload}
          onChange={(value) => setWorkload(value as WorkloadKind)}
          options={[
            { value: "scheduled", label: "Scheduled" },
            { value: "graph", label: "Graph trace" },
          ]}
        />
        <Select
          label="Storage"
          value={storage}
          onChange={(value) => setStorage(value as StorageMode)}
          options={[
            { value: "retain", label: "Retain rows" },
            { value: "exact-fold", label: "Exact fold" },
            { value: "sketch", label: "Sketch" },
          ]}
        />
        <Select
          label="Start"
          value={start}
          onChange={(value) => setStart(value as StartMode)}
          options={[
            { value: "synchronized", label: "Synchronized" },
            { value: "phaser", label: "Phaser · opt-in" },
            { value: "barrier-free", label: "Barrier-free" },
          ]}
        />
        <Toggle checked={truth === "full"} onChange={setRoadmap} label="Roadmap" />
      </div>

      <Grid columns="repeat(5, minmax(0, 1fr))" gap={10}>
        {readouts.map(([label, value]) => (
          <Stat key={label} label={label} value={value} />
        ))}
      </Grid>

      {route.warning ? (
        <Callout tone="warning">{route.warning}</Callout>
      ) : null}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="border border-stroke-secondary" style={{ height: 520 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={edges}
            onNodeClick={(_event, node) => setSelectedId(node.id)}
            onEdgeClick={(_event, edge) => setSelectedId(edge.id)}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
        <Inspector selectedId={selectedId} />
      </div>

      <CellCrossSection storage={storage} workload={workload} />

      <Divider />
      <Row gap={12} align="center" wrap>
        <Code inline>{workload === "graph" ? "PartitionedGraphTraceSource" : "CellularAutonomousIssuer"}</Code>
        <span className={`text-sm ${inkClassName("secondary")}`}>
          {workload === "graph"
            ? "Whole traces are cell-local; retain merge concatenates and renumbers."
            : "Global ordinal = phase_base + within_phase_local × cell_count + cell_id."}
        </span>
      </Row>
    </Stack>
  );
}
