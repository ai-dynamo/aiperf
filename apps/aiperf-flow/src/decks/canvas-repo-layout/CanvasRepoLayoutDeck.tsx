/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Pill } from "../../prose/Pill.js";
import { Table } from "../../prose/Table.js";
import type { CategoryRole } from "../../theme/tokens.js";

// Ported from docs/canvases/canvas-repo-layout.canvas.tsx (a real, hand-authored Cursor
// Canvas). Single-view canvas: no PageTabs union in the source, so this is one component.
// Explains how docs/canvases/*.canvas.tsx source is bridged into Cursor's per-machine
// managed canvases directory via a symlink, keeping git history on the real repo files.

type FlowStepTone = "built" | "local" | "git";

type FlowStep = {
  id: string;
  title: string;
  detail: string;
  tone: FlowStepTone;
};

const FLOW_STEPS: FlowStep[] = [
  {
    id: "edit",
    title: "Edit in repo",
    detail: "Source files live in docs/canvases/*.canvas.tsx and are versioned with the project.",
    tone: "git",
  },
  {
    id: "symlink",
    title: "IDE bridge",
    detail:
      "Each file is symlinked into ~/.cursor/projects/<workspace>/canvases/ so Cursor can compile and preview it beside chat.",
    tone: "built",
  },
  {
    id: "sidecar",
    title: "Local runtime state",
    detail: "*.canvas.data.json and *.canvas.status.json stay in the managed directory and are gitignored.",
    tone: "local",
  },
];

// Every FlowStepTone maps to a literal category role string (see the Tailwind-JIT-trap note
// in SKILL.md) — never interpolate a category into a class name at runtime.
const TONE_CATEGORY: Record<FlowStepTone, CategoryRole> = {
  built: "green",
  local: "yellow",
  git: "blue",
};

const flowNodes: Node[] = FLOW_STEPS.map((step, index) => ({
  id: step.id,
  type: "panel",
  position: { x: index * 340, y: 0 },
  data: { title: step.title, detail: step.detail },
}));

const flowEdges: Edge[] = [
  { id: "e-edit-symlink", source: "edit", target: "symlink", type: "flow" },
  { id: "e-symlink-sidecar", source: "symlink", target: "sidecar", type: "flow" },
];

type CanvasEntry = {
  name: string;
  topic: string;
};

const CANVASES: CanvasEntry[] = [
  { name: "cellular-algorithm-workbook", topic: "Cellular algorithm workbook" },
  { name: "cellular-architecture", topic: "Cellular controller / cell topology" },
  { name: "dynosim-offline-flow", topic: "Dynosim offline replay flow" },
  { name: "mock-server-architecture", topic: "aiperf-mock-server surface map" },
  { name: "rust-aiperf-architecture", topic: "Rust product architecture" },
  { name: "segment-pools-and-body-plans", topic: "Segment pools and body plans" },
  { name: "velo-in-aiperf", topic: "Velo transport in cellular mode" },
];

const DIRECTORY_MAP = `repo/
  docs/canvases/
    *.canvas.tsx          # committed source (edit here)
    .gitignore            # ignores runtime sidecars if they land here

managed (per machine)/
  ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-rust/canvases/
    *.canvas.tsx -> repo/docs/canvases/*.canvas.tsx
    *.canvas.data.json    # local UI state (not committed)
    tsconfig.json         # IDE tooling only`;

const SYMLINK_COMMAND = `ln -s "$PWD/docs/canvases/my-topic.canvas.tsx" \\
  ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-rust/canvases/`;

function TonePill({ tone }: { tone: FlowStepTone }): React.JSX.Element {
  return <Pill tone={TONE_CATEGORY[tone]}>{tone}</Pill>;
}

/**
 * Ports `docs/canvases/canvas-repo-layout.canvas.tsx` — a meta-canvas documenting how
 * `docs/canvases/*.canvas.tsx` source files are bridged into Cursor's per-machine managed
 * canvases directory via a symlink. Single view: intro, why-symlinks callout, an edit ->
 * symlink -> sidecar flow diagram, the directory map, the table of committed canvases, and
 * how to add a new one.
 */
export function CanvasRepoLayoutDeck(): React.JSX.Element {
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 bg-surface-page px-10 py-8">
      <Stack gap={8}>
        <h1 className="text-2xl font-semibold text-ink-primary">Canvas repo layout</h1>
        <p className="max-w-3xl text-sm text-ink-secondary">
          Committed canvases for the AIPerf Rust workspace. Source of truth is in the repo; Cursor still previews
          through its managed canvases directory.
        </p>
        <Row gap={8} wrap>
          <Pill tone="green">7 canvases migrated</Pill>
          <Pill tone="blue">docs/canvases/</Pill>
          <Pill tone="gray">symlink bridge</Pill>
        </Row>
      </Stack>

      <Callout tone="info" title="Why not commit only to the repo path?">
        Cursor detects canvases only when they appear as direct children of{" "}
        <Code inline>~/.cursor/projects/&lt;workspace&gt;/canvases/</Code>. Symlinks satisfy that rule while keeping
        git history on the real files under <Code inline>docs/canvases/</Code>.
      </Callout>

      <Stack gap={10}>
        <div style={{ height: 220 }}>
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={flowNodes}
            edges={flowEdges}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        </div>
        <Grid columns={3} gap={12}>
          {FLOW_STEPS.map((step) => (
            <div key={step.id} className="rounded-none border border-stroke-secondary px-4 py-3">
              <Row justify="space-between" align="center">
                <div className="text-sm font-semibold text-ink-primary">{step.title}</div>
                <TonePill tone={step.tone} />
              </Row>
              <p className="mt-2 text-sm text-ink-secondary">{step.detail}</p>
            </div>
          ))}
        </Grid>
      </Stack>

      <Stack gap={10}>
        <h2 className="text-lg font-semibold text-ink-primary">Directory map</h2>
        <Code>{DIRECTORY_MAP}</Code>
      </Stack>

      <Stack gap={10}>
        <h2 className="text-lg font-semibold text-ink-primary">Committed canvases</h2>
        <p className="text-xs text-ink-tertiary">Source: docs/canvases/ in the AIPerf Rust repo</p>
        <Table
          columns={[
            { key: "file", label: "File" },
            { key: "topic", label: "Topic" },
          ]}
          rows={CANVASES.map((canvas) => ({
            key: canvas.name,
            file: `${canvas.name}.canvas.tsx`,
            topic: canvas.topic,
          }))}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <h2 className="text-lg font-semibold text-ink-primary">Adding a new canvas</h2>
        <Grid columns={2} gap={12}>
          <div className="rounded-none border border-stroke-secondary px-4 py-3">
            <div className="text-sm font-semibold text-ink-primary">1. Create source in repo</div>
            <p className="mt-2 text-sm text-ink-secondary">
              Add docs/canvases/my-topic.canvas.tsx. Import only from cursor/canvas and default-export one component.
            </p>
          </div>
          <div className="rounded-none border border-stroke-secondary px-4 py-3">
            <div className="text-sm font-semibold text-ink-primary">2. Bridge to Cursor</div>
            <div className="mt-2">
              <Code>{SYMLINK_COMMAND}</Code>
            </div>
          </div>
        </Grid>
      </Stack>

      <div className="rounded-none border border-stroke-secondary bg-surface-elevated px-4 py-3">
        <h3 className="text-base font-semibold text-ink-primary">Companion planning docs</h3>
        <p className="mt-1.5 text-sm text-ink-secondary">
          Markdown storyboards for some canvases already live under docs/superpowers/plans/. Keep narrative/planning
          text there; keep interactive architecture views here as .canvas.tsx files.
        </p>
      </div>
    </div>
  );
}
