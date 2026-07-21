/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared rendering pieces for the Mock Foundry (mock-server-architecture) deck. Kept local to
//! this deck folder so no shared `src/` primitive is edited. `SignatureFlow` renders one page's
//! node chain as a real interactive React Flow diagram driven by `useStepSimulator`; `CatalogTable`
//! renders a chapter's catalog entries verbatim; `ChapterIntro` renders the heading + framing copy.

import { useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import {
  EVIDENCE_LABEL,
  STATUS_LABEL,
  SPECIMEN_STAGE,
  type Chapter,
  type FeaturePage,
} from "./catalog.js";

/** Heading + framing copy for a chapter page. */
export function ChapterIntro({
  chapter,
  lead,
}: {
  chapter: Chapter;
  lead: string;
}): React.JSX.Element {
  return (
    <div>
      <div className={`text-[11px] font-bold uppercase tracking-wide ${inkClassName("tertiary")}`}>
        {chapter.short} · {chapter.world}
      </div>
      <h2 className={`mt-1 text-lg font-semibold ${inkClassName("primary")}`}>{chapter.title}</h2>
      <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>{lead}</p>
      <p className={`mt-1 text-xs ${inkClassName("tertiary")}`}>
        Specimen at this stage: <code>{SPECIMEN_STAGE[chapter.id]}</code>
      </p>
    </div>
  );
}

// The active node in the walkthrough gets an accent border; nodes already crossed get a muted
// solid border; not-yet-reached nodes stay in their default secondary stroke. These are all
// literal class strings so the Tailwind JIT scanner keeps them (never build them dynamically).
const NODE_ACTIVE = "border-accent-primary bg-accent-tint";
const NODE_DONE = "border-stroke-primary";

/**
 * Renders a single catalog page's `nodes` list as a left-to-right React Flow chain and drives a
 * step-through highlight over it with {@link useStepSimulator}. Step `i` highlights node `i` and
 * marks earlier nodes as crossed. When the page carries a bespoke `steps` array (e.g. Accept →
 * Route → Budget), that label is shown alongside the node label.
 */
export function SignatureFlow({ page }: { page: FeaturePage }): React.JSX.Element {
  const sim = useStepSimulator(page.nodes, { autoPlayMs: 1100 });
  const active = sim.index;

  const { nodes, edges } = useMemo(() => {
    const built: Node[] = page.nodes.map((label, i) => {
      const state = i === active ? NODE_ACTIVE : i < active ? NODE_DONE : undefined;
      const isEndpoint = i === 0 || i === page.nodes.length - 1;
      return {
        id: `n-${i}`,
        type: isEndpoint ? "panel" : "chip",
        position: { x: i * 210, y: (i % 2) * 96 },
        data: isEndpoint
          ? { title: label, detail: i === 0 ? "input" : "terminal", className: state }
          : { label, className: state },
      };
    });
    const built_edges: Edge[] = page.nodes.slice(1).map((_, i) => ({
      id: `e-${i}`,
      source: `n-${i}`,
      target: `n-${i + 1}`,
      type: "flow",
      data: { speed: i < active ? ("fast" as const) : ("slow" as const) },
    }));
    return { nodes: built, edges: built_edges };
  }, [page.nodes, active]);

  const stepLabel = page.steps[Math.min(active, page.steps.length - 1)];

  return (
    <Stack gap={10}>
      <Row gap={10} align="center" wrap>
        <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
          Advance specimen
        </Button>
        <Button variant="secondary" onClick={sim.togglePlay} disabled={sim.isLast}>
          {sim.isPlaying ? "Pause" : "Play"}
        </Button>
        <Button variant="ghost" onClick={sim.reset}>
          Reset
        </Button>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
          {active + 1}/{page.nodes.length} · <strong>{page.nodes[active]}</strong>
          {page.steps.length > 0 && stepLabel ? ` · phase: ${stepLabel}` : ""}
        </span>
      </Row>
      <div
        className={`rounded-lg border shadow-sm ${strokeClassName("secondary")} ${surfaceClassName("panel")}`}
        style={{ height: 320 }}
      >
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </div>
    </Stack>
  );
}

const CATALOG_COLUMNS: TableColumn[] = [
  { key: "title", label: "Feature" },
  { key: "status", label: "Status" },
  { key: "evidence", label: "Evidence" },
  { key: "source", label: "Source" },
  { key: "proof", label: "Strongest proof" },
  { key: "invariant", label: "Invariant" },
];

const STATUS_TONE: Record<string, TableRow["tone"]> = {
  built: "success",
  partial: "warning",
  boundary: "danger",
};

/**
 * Renders every catalog entry of a chapter as a table — title, status, evidence tier, source and
 * proof file paths, and the invariant sentence — so all of that chapter's facts are visible even
 * though only a signature page is rendered as an interactive diagram.
 */
export function CatalogTable({ pages }: { pages: readonly FeaturePage[] }): React.JSX.Element {
  const rows: TableRow[] = pages.map((entry) => ({
    title: entry.title,
    status: STATUS_LABEL[entry.status],
    evidence: EVIDENCE_LABEL[entry.evidence],
    source: <code className="text-xs">{entry.source}</code>,
    proof: <code className="text-xs">{entry.proof}</code>,
    invariant: entry.invariant,
    tone: STATUS_TONE[entry.status],
  }));
  return <Table columns={CATALOG_COLUMNS} rows={rows} />;
}

/**
 * Full chapter page: intro, the signature page rendered as an interactive React Flow walkthrough,
 * the chapter's mode vocabulary, and the verbatim catalog table. Each per-chapter file is a thin
 * wrapper that supplies its chapter, lead copy, and which page is the signature diagram.
 */
export function ChapterPage({
  chapter,
  lead,
  signature,
  pages,
}: {
  chapter: Chapter;
  lead: string;
  signature: FeaturePage;
  pages: readonly FeaturePage[];
}): React.JSX.Element {
  return (
    <Stack gap={16}>
      <ChapterIntro chapter={chapter} lead={lead} />
      <div>
        <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>
          Signature walkthrough — {signature.title}
        </h3>
        <SignatureFlow page={signature} />
        <p className={`mt-2 text-xs ${inkClassName("tertiary")}`}>
          Invariant: {signature.invariant}
        </p>
      </div>
      <div>
        <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>
          Chapter feature catalog ({pages.length})
        </h3>
        <CatalogTable pages={pages} />
      </div>
      <div>
        <h3 className={`mb-2 text-sm font-semibold ${inkClassName("secondary")}`}>Mode vocabulary</h3>
        <ModeList pages={pages} />
      </div>
    </Stack>
  );
}

/** Compact modes/steps chips for a page, so per-page mode vocabulary stays visible. */
export function ModeList({ pages }: { pages: readonly FeaturePage[] }): React.JSX.Element {
  return (
    <Stack gap={6}>
      {pages
        .filter((entry) => entry.modes.length > 0)
        .map((entry) => (
          <div key={entry.id} className="text-xs">
            <span className={`font-semibold ${inkClassName("secondary")}`}>{entry.title}:</span>{" "}
            <span className={inkClassName("tertiary")}>{entry.modes.join(" · ")}</span>
          </div>
        ))}
    </Stack>
  );
}
