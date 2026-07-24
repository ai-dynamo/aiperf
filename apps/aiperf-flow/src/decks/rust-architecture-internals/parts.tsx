/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared one-off primitives for the `rust-architecture-internals` deck, ported from the
//! helper components in `docs/canvases/rust-architecture-internals.canvas.tsx` (`Segmented`,
//! `SectionHeading`, `Sources`, and the SVG `Node`/`Band`/`Seam` boxes). Diagram boxes are
//! re-authored as real React Flow nodes; these helpers only cover the prose chrome and the
//! node/edge builders. Scoped to this deck folder to avoid touching shared `src/layout`.

import type { ReactNode } from "react";
import type { Edge, Node } from "@xyflow/react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  ReactFlowProvider,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { Row } from "../../layout/Row.js";
import {
  surfaceClassName,
  strokeClassName,
  inkClassName,
  type StrokeRole,
} from "../../theme/tokens.js";

/** Detail level shared across every section (the canvas's global `rust-internals-v2.detail`). */
export type Detail = "orientation" | "engineering" | "source";

/** Numeric rank of a detail level, mirroring the canvas `rank()` helper. */
export function rank(detail: Detail): number {
  return { orientation: 0, engineering: 1, source: 2 }[detail];
}

// Source SVG coordinates are scaled up so the min-width React Flow nodes do not overlap.
const SCALE_X = 1.5;
const SCALE_Y = 1.7;

function pos(x: number, y: number): { x: number; y: number } {
  return { x: x * SCALE_X, y: y * SCALE_Y };
}

/** Header (band label) node. */
export function headerNode(
  id: string,
  x: number,
  y: number,
  title: string,
  caption?: string,
): Node {
  return { id, type: "header", position: pos(x, y), data: { title, caption } };
}

/** Panel node (title + optional detail line). */
export function panelNode(
  id: string,
  x: number,
  y: number,
  title: string,
  detail?: string,
  strokeRole?: StrokeRole,
): Node {
  return { id, type: "panel", position: pos(x, y), data: { title, detail, strokeRole } };
}

/** Card node (title + subtitle + optional detail). */
export function cardNode(
  id: string,
  x: number,
  y: number,
  title: string,
  subtitle?: string,
  detail?: string,
  strokeRole?: StrokeRole,
): Node {
  return {
    id,
    type: "card",
    position: pos(x, y),
    data: { title, subtitle, detail, strokeRole },
  };
}

/** Chip node (bare label, no handles). */
export function chipNode(id: string, x: number, y: number, label: string): Node {
  return { id, type: "chip", position: pos(x, y), data: { label } };
}

/** Animated `flow` edge signalling data/request movement. */
export function flowEdge(
  id: string,
  source: string,
  target: string,
  opts?: { label?: string; speed?: "slow" | "normal" | "fast" },
): Edge {
  return {
    id,
    source,
    target,
    type: "flow",
    label: opts?.label,
    data: opts?.speed ? { speed: opts.speed } : undefined,
  };
}

/** Plain static connector. */
export function plainEdge(id: string, source: string, target: string): Edge {
  return { id, source, target };
}

export type FlowFrameProps = {
  nodes: Node[];
  edges: Edge[];
  height?: number;
  /**
   * Optional ELK auto-layout. When set, node positions are computed from graph structure and
   * measured sizes (the authored `position` hints are ignored); omit it to keep the legacy
   * manual-position behavior unchanged. Options should be a stable (module-level) object.
   */
  layout?: ElkOptions;
};

/**
 * Fixed-height React Flow canvas frame with the deck's standard dotted background and
 * disabled interaction chrome. Every diagram in this deck renders through here.
 */
export function FlowFrame({ nodes, edges, height = 420, layout }: FlowFrameProps): React.JSX.Element {
  return (
    <div style={{ height }} className={clsx("border", strokeClassName("secondary"))}>
      <ReactFlowProvider>
        {layout ? (
          <FlowFrameAutoLaid nodes={nodes} edges={edges} layout={layout} />
        ) : (
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            nodesDraggable={false}
            nodesConnectable={false}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        )}
      </ReactFlowProvider>
    </div>
  );
}

/** Inner canvas: runs the ELK hook (inside the provider) and renders the auto-laid-out nodes. */
function FlowFrameAutoLaid({
  nodes: inputNodes,
  edges,
  layout,
}: {
  nodes: Node[];
  edges: Edge[];
  layout: ElkOptions;
}): React.JSX.Element {
  const { nodes, laidOut } = useElkLayout(inputNodes, edges, layout);
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.12 }}
      nodesDraggable={false}
      nodesConnectable={false}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

export type SegmentedOption<T extends string> = { id: T; label: string };

export type SegmentedProps<T extends string> = {
  value: T;
  onChange: (value: T) => void;
  options: ReadonlyArray<SegmentedOption<T>>;
  ariaLabel?: string;
};

/**
 * Row of mutually exclusive view-mode buttons — the deck's replacement for the canvas
 * `Segmented` control. Plain `useState` in each section drives the selection.
 */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
}: SegmentedProps<T>): React.JSX.Element {
  return (
    <Row gap={6} wrap align="center">
      <div role="group" aria-label={ariaLabel} className="flex flex-wrap gap-1.5">
        {options.map((option) => {
          const selected = option.id === value;
          return (
            <button
              key={option.id}
              type="button"
              aria-pressed={selected}
              onClick={() => onChange(option.id)}
              className={clsx(
                "rounded-md border px-3 py-1.5 text-xs font-medium shadow-sm transition-colors",
                selected
                  ? clsx("bg-accent-primary text-white", "border-accent-primary")
                  : clsx(surfaceClassName("elevated"), strokeClassName("secondary"), inkClassName("secondary")),
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </Row>
  );
}

export type SectionHeadingProps = {
  number: string;
  title: string;
  subtitle: string;
};

/** Numbered section heading — a compact rounded index badge, the title, and a subtitle. */
export function SectionHeading({ number, title, subtitle }: SectionHeadingProps): React.JSX.Element {
  return (
    <Row gap={12} align="start">
      <div
        className={clsx(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-md border text-[11px] font-bold shadow-sm",
          surfaceClassName("elevated"),
          strokeClassName("primary"),
          inkClassName("tertiary"),
        )}
      >
        {number}
      </div>
      <div className="flex flex-col gap-1">
        <h2 className={clsx("text-xl font-bold", inkClassName("primary"))}>{title}</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>{subtitle}</p>
      </div>
    </Row>
  );
}

export type SourceRef = { label: string; path: string };

export type SourcesRowProps = {
  detail: Detail;
  paths: SourceRef[];
};

/**
 * Source-evidence links, rendered only at the `source` detail level (matching the canvas
 * `Sources` component). There is no file-opening host here, so each path is a static,
 * titled reference chip rather than an action button.
 */
export function SourcesRow({ detail, paths }: SourcesRowProps): React.JSX.Element | null {
  if (detail !== "source") {
    return null;
  }
  return (
    <Row gap={8} wrap align="center">
      <span className={clsx("text-xs", inkClassName("tertiary"))}>source evidence</span>
      {paths.map((item) => (
        <span
          key={item.path}
          title={item.path}
          className={clsx(
            "rounded-md border px-2 py-1 font-mono text-[11px] shadow-sm",
            surfaceClassName("panel"),
            strokeClassName("secondary"),
            inkClassName("secondary"),
          )}
        >
          {item.label}
        </span>
      ))}
    </Row>
  );
}

/** Convenience wrapper for a section body with consistent vertical rhythm. */
export function SectionShell({ children }: { children: ReactNode }): React.JSX.Element {
  return <div className="flex flex-col gap-4">{children}</div>;
}
