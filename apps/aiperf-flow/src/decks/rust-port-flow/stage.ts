/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deck-local contract every `rust-port-flow` stage module implements, plus the pure builders that
//! assemble the 9 stage definitions into the overview React Flow diagram and the `ZoomTree` the
//! shared `ZoomStage` navigates. Stage agents own one `stages/<id>.ts` file each; they never touch
//! the deck shell — they just fill in their stage's `subgraph`/`leaves`/`evidence`.

import type { Edge, Node } from "@xyflow/react";
import { categoryBgTintClassName } from "../../theme/tokens.js";
import type { CategoryRole } from "../../theme/tokens.js";

// ────────────────────────────────────────────────────────────────────────────────────────────────
// Semantic node roles: drill-down boxes are colored by WHAT THEY ARE, not by their stage, so the
// same kind of thing reads the same everywhere. A stage tags each node with a `NodeRole`; the helper
// paints a matching category tint + a colored left border (the `!` overrides Card's default accent).
// ────────────────────────────────────────────────────────────────────────────────────────────────

/** What a drill-down box represents (drives its color). */
export type NodeRole =
  | "storage" // data at rest: stores, arenas, segments, handles
  | "compute" // work/execution: workers, threads, hot path, reducers
  | "transport" // network/IO: HTTP/gRPC, dispatch, sinks, the wire
  | "control" // orchestration: scheduler, clock, phase, coordination, config
  | "media" // out-of-band media: content_server, sidecars, publishers
  | "server" // external inference server / endpoint
  | "neutral"; // plain junctions/labels with no strong category

const ROLE_COLOR: Record<NodeRole, CategoryRole> = {
  storage: "blue",
  compute: "green",
  transport: "cyan",
  control: "purple",
  media: "orange",
  server: "yellow",
  neutral: "gray",
};

// Static literal maps so Tailwind's scanner emits every class (never interpolate — see SKILL.md).
// A soft FULL border in the role color (all four sides, not a hard left strip)…
const ROLE_BORDER: Record<NodeRole, string> = {
  storage: "!border-category-blue/60",
  compute: "!border-category-green/60",
  transport: "!border-category-cyan/60",
  control: "!border-category-purple/60",
  media: "!border-category-orange/60",
  server: "!border-category-yellow/60",
  neutral: "!border-category-gray/60",
};

// …plus a soft drop shadow: a dark elevation shadow for the 3D float, and a low, diffuse glow in the
// role color so each box reads as a shadowed COLOR box (not a color strip). Values are single
// literal strings (underscore = space) so Tailwind emits them verbatim.
const ROLE_SHADOW: Record<NodeRole, string> = {
  storage: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-blue)]",
  compute: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-green)]",
  transport: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-cyan)]",
  control: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-purple)]",
  media: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-orange)]",
  server: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-yellow)]",
  neutral: "!shadow-[0_10px_28px_-12px_rgba(0,0,0,0.72),0_2px_16px_-6px_var(--color-category-gray)]",
};

/**
 * Class string for a node of the given semantic role: a soft-tinted fill, a full soft colored
 * border (no left strip), and a soft dark-elevation + role-colored drop shadow — a floating 3D
 * "shadowed color box".
 */
export function roleClassName(role: NodeRole): string {
  return `${categoryBgTintClassName(ROLE_COLOR[role])} ${ROLE_BORDER[role]} ${ROLE_SHADOW[role]}`;
}

/**
 * Legend rows for the deck to render a color key. Order is deliberate: it separates the visually
 * closest hues (green↔cyan, orange↔yellow) so no two adjacent swatches are hard to tell apart —
 * this ordering passes the colorblind + normal-vision adjacency checks (validated with the dataviz
 * palette validator on the dark surface). Every swatch is also text-labeled (secondary encoding).
 */
export const NODE_ROLE_LEGEND: ReadonlyArray<{ role: NodeRole; label: string; color: CategoryRole }> = [
  { role: "storage", label: "Storage / data", color: ROLE_COLOR.storage },
  { role: "compute", label: "Compute / workers", color: ROLE_COLOR.compute },
  { role: "media", label: "Media / sidecar", color: ROLE_COLOR.media },
  { role: "control", label: "Control / orchestration", color: ROLE_COLOR.control },
  { role: "transport", label: "Transport / network", color: ROLE_COLOR.transport },
  { role: "server", label: "Inference server", color: ROLE_COLOR.server },
];
import type {
  Lane,
  LaneId,
  RequestPath,
  SeamFrame,
  StageRegion,
  TimelineEvent,
  ZoomTree,
} from "../../interactive/index.js";

/** One verified source anchor (real `file:line`) shown in a stage's level-1 evidence row. */
export interface StageEvidence {
  label: string;
  path: string;
}

/** A stage's internal (level-1) React Flow subgraph. */
export interface StageSubgraph {
  nodes: Node[];
  edges: Edge[];
  /** Child ids (keys of `StageDef.leaves`) that a node in this subgraph can drill into. */
  children?: string[];
}

/**
 * The typed contract a stage module implements. The deck skeleton ships a stub `StageDef` per
 * spine stage (id/order/label/caption/tone only); a stage agent fleshes out `subgraph`, optional
 * `leaves` (a third zoom level), and `evidence` in that stage's own file.
 */
export interface StageDef {
  /** Stable id — the overview node id AND the `ZoomTree` key for this stage. */
  id: string;
  /** 0-8 ordinal in the narrative spine; drives overview layout + edge wiring. */
  order: number;
  /** Short label on the overview stage node and the breadcrumb. */
  label: string;
  /** One-line "what this stage does" caption, grounded in the spec's narrative spine. */
  caption: string;
  /** Category color accent for the stage. */
  tone: CategoryRole;
  /**
   * v2 swimlane-timeline: which lane this stage's region sits in (one of {@link RUST_PORT_LANE_IDS}).
   * A stage agent picks the lane its subsystem lives in.
   */
  lane: LaneId;
  /**
   * v2 swimlane-timeline: the ordered event points the request line passes through for this stage.
   * `atOrder` places the event on the (evenly-spaced) virtual axis; `realOffsetMs` its wall-ms offset
   * for the RealClock scale. An event's `laneId` is usually `lane`, but may differ (e.g. an
   * aggregate-lane stage that also emits export-lane events). Derived from the stage's real captions
   * and type names — no placeholders. A lane agent enriches ONE stage's events here.
   */
  events: TimelineEvent[];
  /** Level-1 subgraph. Omit in a stub; the deck synthesizes a single-node seed from the caption. */
  subgraph?: StageSubgraph;
  /** Level-2+ subgraphs keyed by child id, for a stage that earns a deeper drill (e.g. Transport → HTTP). */
  leaves?: Record<string, { label: string; nodes: Node[]; edges: Edge[] }>;
  /** Real `file:line` source anchors for this stage. */
  evidence?: StageEvidence[];
}

/** Id of the synthetic root node that holds the 9-stage overview diagram. */
export const OVERVIEW_ID = "overview";

// ────────────────────────────────────────────────────────────────────────────────────────────────
// v2 swimlane-timeline model (the OVERVIEW is now a `TimelineTrack`, not a React Flow node graph).
// ────────────────────────────────────────────────────────────────────────────────────────────────

/** The 6 rust-port swimlane ids, top→bottom in request-touch order (spec §"Lanes"). */
export const RUST_PORT_LANE_IDS = [
  "dataset",
  "scheduler",
  "transport",
  "server",
  "aggregate",
  "export",
] as const;

/** Union of the 6 rust-port lane ids (a `LaneId` refinement for stage authors). */
export type RustPortLaneId = (typeof RUST_PORT_LANE_IDS)[number];

/** The 6 swimlanes rendered by `TimelineTrack`, top→bottom (spec §"Lanes"). */
export const LANES: readonly Lane[] = [
  { id: "dataset", label: "Dataset" },
  { id: "scheduler", label: "Scheduler / Workload" },
  { id: "transport", label: "Transport" },
  { id: "server", label: "Server" },
  { id: "aggregate", label: "Aggregate" },
  { id: "export", label: "Export" },
];

/**
 * The 3 nested seam frames (spec §"Seam frames"). `clock` spans the whole axis (its unit/scale is
 * clock-mode dependent); `workload` frames the scheduler admission segment; `transport` frames the
 * dispatch→server→reduce segment. `spanOrder` values reference the stages' event `atOrder`s.
 */
export const SEAM_FRAMES: readonly SeamFrame[] = [
  { id: "clock", label: "Clock" },
  { id: "workload", label: "Workload", spanLaneIds: ["scheduler"], spanOrder: [5, 6] },
  { id: "transport", label: "Transport", spanLaneIds: ["transport", "server"], spanOrder: [8, 11] },
];

/** Everything a `TimelineTrack` needs, assembled from the stages' lane/event metadata. */
export interface TimelineModel {
  lanes: readonly Lane[];
  regions: StageRegion[];
  events: TimelineEvent[];
  seamFrames: readonly SeamFrame[];
  requestPath: RequestPath;
}

/**
 * Assembles the swimlane-timeline model from the stage registry: one `StageRegion` per stage (its
 * lane + the order-span of its events), the flat event list, the fixed lanes/seam-frames, and the
 * `requestPath` (all events in ascending `atOrder`). A stage with no events contributes no region.
 */
export function buildTimelineModel(stages: readonly StageDef[]): TimelineModel {
  const events: TimelineEvent[] = [];
  const regions: StageRegion[] = [];
  for (const stage of stages) {
    if (stage.events.length === 0) {
      continue;
    }
    for (const event of stage.events) {
      events.push(event);
    }
    const orders = stage.events.map((e) => e.atOrder);
    regions.push({
      id: stage.id,
      laneId: stage.lane,
      label: stage.label,
      startOrder: Math.min(...orders),
      endOrder: Math.max(...orders),
    });
  }
  const requestPath: RequestPath = [...events]
    .sort((a, b) => a.atOrder - b.atOrder)
    .map((e) => e.id);
  return { lanes: LANES, regions, events, seamFrames: SEAM_FRAMES, requestPath };
}

const COL = 300;
const ROW = 380;

/** Boustrophedon position for a stage by its spine ordinal, so edges read as one connected pipeline. */
function overviewPosition(order: number): { x: number; y: number } {
  // Row 0: orders 0-3 left→right; row 1: orders 4-7 right→left; row 2: order 8 at the left.
  if (order <= 3) {
    return { x: order * COL, y: 0 };
  }
  if (order <= 7) {
    return { x: (7 - order) * COL, y: ROW };
  }
  return { x: 0, y: ROW * 2 };
}

/** The 9 overview nodes: one `card` per stage, tinted by its tone. Node id === stage id (so a click drills). */
export function overviewNodes(stages: readonly StageDef[]): Node[] {
  return stages.map((stage) => ({
    id: stage.id,
    type: "card",
    position: overviewPosition(stage.order),
    data: {
      title: stage.label,
      subtitle: `Stage ${stage.order}`,
      detail: stage.caption,
      className: categoryBgTintClassName(stage.tone),
    },
  }));
}

/** Sequential `flow` edges wiring stage order 0→1→…→8. */
export function overviewEdges(stages: readonly StageDef[]): Edge[] {
  const ordered = [...stages].sort((a, b) => a.order - b.order);
  const edges: Edge[] = [];
  for (let i = 0; i < ordered.length - 1; i++) {
    const source = ordered[i]!;
    const target = ordered[i + 1]!;
    edges.push({ id: `e-${source.id}-${target.id}`, source: source.id, target: target.id, type: "flow" });
  }
  return edges;
}

/** A one-node subgraph synthesized for a stub stage so its level-1 canvas is real, never empty. */
function seedSubgraph(stage: StageDef): StageSubgraph {
  return {
    nodes: [
      {
        id: `${stage.id}__seed`,
        type: "card",
        position: { x: 0, y: 0 },
        data: {
          title: stage.label,
          subtitle: `Stage ${stage.order}`,
          detail: stage.caption,
          className: categoryBgTintClassName(stage.tone),
        },
      },
    ],
    edges: [],
  };
}

/**
 * Assembles the navigable `ZoomTree`: a synthetic overview root whose children are the 9 stages,
 * each stage node carrying its (real or seeded) subgraph and its `StageDef` as `data`, plus any
 * leaf nodes a stage defines for a deeper zoom level.
 */
export function buildZoomTree(stages: readonly StageDef[]): ZoomTree<StageDef> {
  const tree: ZoomTree<StageDef> = {
    [OVERVIEW_ID]: {
      label: "Big-picture request lifecycle",
      nodes: overviewNodes(stages),
      edges: overviewEdges(stages),
      children: [...stages].sort((a, b) => a.order - b.order).map((s) => s.id),
    },
  };

  for (const stage of stages) {
    const sub = stage.subgraph ?? seedSubgraph(stage);
    const leafIds = Object.keys(stage.leaves ?? {});
    tree[stage.id] = {
      label: stage.label,
      nodes: sub.nodes,
      edges: sub.edges,
      children: sub.children ?? (leafIds.length > 0 ? leafIds : undefined),
      data: stage,
    };
    for (const [leafId, leaf] of Object.entries(stage.leaves ?? {})) {
      tree[leafId] = { label: leaf.label, nodes: leaf.nodes, edges: leaf.edges };
    }
  }

  return tree;
}
