/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The 20-page cellular storyboard as a live `useStepSimulator` walkthrough. Each step reveals
//! more of the atlas node/edge graph (progressive `storyVisibility`), with a source-grounded
//! evidence margin, a reduction-mode strip on the Reduce pages, and a fidelity ladder on page 19.
//! Ported from `CellularArchitectureAtlas` / `StoryPage` in the source canvas.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Code } from "../../prose/Code.js";
import { Divider } from "../../layout/Divider.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName } from "../../theme/tokens.js";
import { Inspector } from "./Inspector.js";
import { buildAtlasGraph } from "./atlasGraph.js";
import {
  NODES,
  EDGES,
  STORY_STEPS,
  storyVisibility,
  REDUCTION_COPY,
  type ReductionMode,
  type StoryStep,
} from "./data.js";

const STORY_CHAPTERS = [
  { label: "Launch", pages: [1, 2, 3, 4, 5] },
  { label: "Distribute", pages: [6, 7, 8, 9, 10] },
  { label: "Execute", pages: [11, 12, 13, 14, 15] },
  { label: "Reduce", pages: [16, 17, 18] },
  { label: "Scale", pages: [19, 20] },
] as const;

const REDUCTION_LABELS = ["INPUT", "OPERATION", "WIRE OUTPUT", "FIDELITY"] as const;

const SCALE_TIERS: ReadonlyArray<readonly [string, string, string]> = [
  ["T0 Exact", "Retain · flat merge", "Built"],
  ["T1 Bounded", "Sketch · flat merge", "Built"],
  ["T2 Hierarchical", "Fold · local tree built (e2e) · k8s wiring partial", "Built local / Partial k8s"],
  ["T3 External sink", "No-central-merge streaming is planned; barrier-free is a separate built START mode", "Planned"],
];

function StoryRail({ page, onPage }: { page: number; onPage: (page: number) => void }): React.JSX.Element {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {STORY_CHAPTERS.map((chapter) => (
        <Stack key={chapter.label} gap={6}>
          <Eyebrow>{chapter.label}</Eyebrow>
          <Row gap={5} wrap>
            {chapter.pages.map((number) => {
              const isCurrent = page === number;
              const isPast = number < page;
              return (
                <button
                  key={number}
                  type="button"
                  aria-pressed={isCurrent}
                  aria-label={`Page ${number}: ${STORY_STEPS[number - 1].title}`}
                  title={`Page ${number}: ${STORY_STEPS[number - 1].title}`}
                  onClick={() => onPage(number)}
                  className={
                    "h-6 w-6 rounded-md border text-xs font-semibold shadow-sm transition-colors " +
                    (isCurrent
                      ? "border-accent-primary bg-accent-primary text-white shadow-md"
                      : isPast
                        ? "border-category-green text-category-green"
                        : "border-stroke-secondary text-ink-secondary")
                  }
                >
                  {number}
                </button>
              );
            })}
          </Row>
        </Stack>
      ))}
    </div>
  );
}

function ReductionSimulation({ mode }: { mode: ReductionMode }): React.JSX.Element {
  return (
    <div className="grid grid-cols-1 border-y border-stroke-secondary sm:grid-cols-4">
      {REDUCTION_COPY[mode].map((value, index) => (
        <div
          key={REDUCTION_LABELS[index]}
          className={
            "px-3 py-2.5 " +
            (index === 0 ? "" : "sm:border-l sm:border-stroke-tertiary ") +
            (index === 2 ? "bg-surface-panel" : "")
          }
        >
          <Eyebrow>{REDUCTION_LABELS[index]}</Eyebrow>
          <div className={`mt-1 text-sm font-semibold ${inkClassName("primary")}`}>{value}</div>
        </div>
      ))}
    </div>
  );
}

function ScaleBoundaryStrip(): React.JSX.Element {
  return (
    <Stack gap={8}>
      <Row gap={10} align="center" wrap>
        <Eyebrow>Cellular fidelity ladder</Eyebrow>
        <span className="rounded-md border border-category-yellow px-2 py-0.5 text-xs font-semibold text-category-yellow shadow-sm">
          Scheduled duration/unbounded rejected · graph duration built
        </span>
      </Row>
      <div className="grid grid-cols-1 border-y border-stroke-secondary sm:grid-cols-4">
        {SCALE_TIERS.map(([title, detail, status], index) => (
          <div
            key={title}
            className={"px-3 py-2.5 " + (index === 0 ? "" : "sm:border-l sm:border-stroke-tertiary")}
          >
            <div className={`text-sm font-semibold ${inkClassName("primary")}`}>{title}</div>
            <div className={`mt-1 text-sm ${inkClassName("secondary")}`}>{detail}</div>
            <div
              className={
                "mt-1 text-sm font-semibold " +
                (status === "Built" ? "text-category-green" : "text-category-yellow")
              }
            >
              {status}
            </div>
          </div>
        ))}
      </div>
    </Stack>
  );
}

function EvidenceMargin({ step, selectedId }: { step: StoryStep; selectedId: string }): React.JSX.Element {
  return (
    <Stack gap={10} className="border-l border-stroke-secondary pl-4">
      <Eyebrow>Invariant</Eyebrow>
      <span className={`text-sm ${inkClassName("primary")}`}>{step.invariant}</span>
      <Divider />
      <Eyebrow>Source evidence</Eyebrow>
      <Code>{step.symbol}</Code>
      <span className={`text-sm ${inkClassName("secondary")}`}>{step.path}</span>
      <span className={`text-xs ${inkClassName("tertiary")}`}>{step.proof}</span>
      <Divider />
      <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Introduced on this page</span>
      <span className={`text-sm ${inkClassName("secondary")}`}>{step.change}</span>
      <Divider />
      <Inspector selectedId={selectedId} />
    </Stack>
  );
}

function buildStoryGraph(step: StoryStep, selectedId: string): { nodes: Node[]; edges: Edge[] } {
  if (step.fullAtlas) {
    const allNodes = new Set(NODES.map((node) => node.id));
    const allEdges = new Set(EDGES.map((edge) => edge.id));
    return buildAtlasGraph({
      visibleNodeIds: allNodes,
      visibleEdgeIds: allEdges,
      activeNodeIds: allNodes,
      activeEdgeIds: allEdges,
      selectedId,
    });
  }
  const visibility = storyVisibility(step.page);
  return buildAtlasGraph({
    visibleNodeIds: visibility.nodeIds,
    visibleEdgeIds: visibility.edgeIds,
    activeNodeIds: new Set(step.addedNodeIds),
    activeEdgeIds: new Set(step.addedEdgeIds),
    selectedId,
  });
}

/**
 * Cellular storyboard walkthrough. Steps through the 20 story pages via {@link useStepSimulator},
 * revealing the atlas graph progressively. Self-contained; takes no required props.
 */
export function StoryPage(): React.JSX.Element {
  const sim = useStepSimulator(STORY_STEPS, { autoPlayMs: 2600 });
  const [selectedId, setSelectedId] = useState("");
  const step: StoryStep = sim.current ?? STORY_STEPS[0];
  const page = step.page;

  // `useStepSimulator` has no absolute seek. Each `next()`/`back()` uses a functional state
  // updater, so N synchronous calls compose to a jump of N — a bounded, deterministic seek.
  const goTo = (target: number) => {
    const clamped = Math.max(1, Math.min(20, target));
    const delta = clamped - page;
    if (delta > 0) for (let i = 0; i < delta; i++) sim.next();
    else for (let i = 0; i < -delta; i++) sim.back();
  };

  const { nodes, edges } = buildStoryGraph(step, selectedId);

  return (
    <Stack gap={16}>
      <div>
        <Eyebrow>
          Cellular story · {step.chapter} · Page {page} of 20
        </Eyebrow>
        <h2 className={`mt-1 text-lg font-semibold ${inkClassName("primary")}`}>{step.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>{step.thesis}</p>
      </div>

      <Row gap={10} align="center" wrap>
        <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>
          Back
        </Button>
        <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
          Next
        </Button>
        <Button variant="secondary" onClick={sim.togglePlay}>
          {sim.isPlaying ? "Pause" : "Play"}
        </Button>
        <Button variant="ghost" onClick={sim.reset}>
          Reset
        </Button>
        <Button variant="ghost" onClick={() => goTo(20)}>
          Jump to full atlas
        </Button>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>{page} / 20</span>
      </Row>

      <StoryRail page={page} onPage={goTo} />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="border border-stroke-secondary" style={{ height: 480 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={edges}
            onNodeClick={(_event, node) => setSelectedId(node.id)}
            onEdgeClick={(_event, edge) => setSelectedId(edge.id)}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
        <EvidenceMargin step={step} selectedId={selectedId} />
      </div>

      {step.simulation ? <ReductionSimulation mode={step.simulation} /> : null}
      {page === 19 ? <ScaleBoundaryStrip /> : null}
    </Stack>
  );
}
