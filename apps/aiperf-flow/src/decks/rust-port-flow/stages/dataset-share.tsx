/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 3 — Sharing the dataset with workers.
//!
//! The frozen `SegmentStore` is interned *once*: bytes live exactly once, and conversations/turns
//! carry dense integer `Handle`s (not bytes). The `Dataset` owns a single `Arc<dyn SegmentStore>`,
//! so each worker thread shares the same arena by cloning a pointer — zero-copy across threads.
//! The level-2 leaf makes the key correction explicit: `content_server` is a SEPARATE run-owned
//! HTTP **media** delivery sidecar, NOT the dataset-text sharing mechanism.
//!
//! Verified source anchors (`rust/…`, read against real code, not the spec):
//!   - `runtime/src/dataset/segment.rs:238`  `pub trait SegmentStore: Send + Sync`
//!   - `runtime/src/dataset/segment.rs:514`  `SegmentPool::freeze -> InMemorySegmentStore`
//!   - `runtime/src/dataset/segment.rs:531`  "Frozen in-memory arena shared across worker threads"
//!   - `runtime/src/dataset/segment.rs:28`   `pub struct Handle(u32)` (dense opaque arena index)
//!   - `runtime/src/dataset/dataset.rs:48`   `segments: Arc<dyn SegmentStore>` on the shared `Dataset`
//!   - `runtime/src/dataset/model.rs:216`    `pub struct Turn` ("every large value is a segment handle")
//!   - `runtime/src/content_server/mod.rs:4` content_server = run-owned HTTP media serving
//!   - `cli/src/model/telemetry.rs:229`      `pub struct ContentServerSidecar` (run-owned HTTP sidecar)

import type { Edge, Node } from "@xyflow/react";
import { categoryBgTintClassName } from "../../../theme/tokens.js";
import type { CategoryRole } from "../../../theme/tokens.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";

const TONE: CategoryRole = "cyan";

/** A `card` node tinted by a category role — the emphasized boxes in this stage's subgraph. */
function card(
  id: string,
  title: string,
  subtitle: string | undefined,
  detail: string,
  x: number,
  y: number,
  tint: CategoryRole = TONE,
): Node {
  return {
    id,
    type: "card",
    position: { x, y },
    data: { title, subtitle, detail, className: categoryBgTintClassName(tint) },
  };
}

/** A `panel` node — a plainer worker/step box. */
function panel(id: string, title: string, detail: string, x: number, y: number): Node {
  return { id, type: "panel", position: { x, y }, data: { title, detail } };
}

/** A `chip` node — a compact label-only tag. */
function chip(id: string, label: string, x: number, y: number): Node {
  return { id, type: "chip", position: { x, y }, data: { label } };
}

/** A `header` node — a grouping heading. */
function header(id: string, title: string, caption: string, x: number, y: number): Node {
  return { id, type: "header", position: { x, y }, data: { title, caption } };
}

/** A solid, animated primary-path edge. */
function flow(source: string, target: string, label?: string): Edge {
  return { id: `e-${source}-${target}`, source, target, type: "flow", label };
}

/** A dashed, muted "separate concern" edge (rendered slow, tertiary color). */
function dashed(source: string, target: string, label?: string): Edge {
  return {
    id: `e-${source}-${target}`,
    source,
    target,
    type: "flow",
    label,
    data: { speed: "slow", color: "var(--color-stroke-tertiary)" },
  };
}

/** The id the level-1 `content_server` node drills into (also its `leaves` key + `ZoomTree` key). */
const CONTENT_SERVER_LEAF = "sharing-content-server";

// Level-1 subgraph: bytes-once → Arc → Handles-not-bytes across W worker threads, with the
// content_server media sidecar sitting alongside as an explicitly disjoint concern.
const subgraphNodes: Node[] = [
  card(
    "share-store",
    "InMemorySegmentStore",
    "frozen arena",
    "Box<[Segment]> interned once — bytes live exactly once (content-addressed, prefix-folded, deduped).",
    0,
    80,
  ),
  card(
    "share-arc",
    "Arc<dyn SegmentStore>",
    "one shared owner",
    "The Dataset holds a single Arc; each worker clones the pointer — zero-copy, never a byte copy.",
    300,
    80,
  ),
  card(
    "share-turns",
    "Conversation · Turn",
    "carries Handles",
    "Per-turn dispatch data: every potentially large value is a dense Handle (u32), not inline bytes.",
    600,
    80,
  ),
  panel(
    "share-w0",
    "Worker thread 0",
    "Turn { … : Handle } — resolves bytes on demand via build_body(handles) at dispatch.",
    900,
    -30,
  ),
  panel(
    "share-w1",
    "Worker thread 1",
    "Same frozen store via Arc::clone — shared read-only, no per-worker copy.",
    900,
    80,
  ),
  panel(
    "share-wn",
    "Worker thread W-1",
    "SegmentStore is Send + Sync, so the one arena is safely shared across every sub-cell thread.",
    900,
    190,
  ),
  chip("share-run", "Run-owned sidecar", 300, 320),
  card(
    CONTENT_SERVER_LEAF,
    "content_server",
    "MEDIA sidecar — NOT dataset sharing",
    "Separate run-owned HTTP server: streams generated images/videos as URLs. Dataset TEXT never flows here — click to compare.",
    600,
    320,
    "orange",
  ),
];

const subgraphEdges: Edge[] = [
  flow("share-store", "share-arc", "frozen"),
  flow("share-arc", "share-turns", "Arc::clone"),
  flow("share-turns", "share-w0"),
  flow("share-turns", "share-w1"),
  flow("share-turns", "share-wn"),
  dashed("share-run", CONTENT_SERVER_LEAF, "media URLs"),
];

// Level-2 leaf: the content_server media sidecar, contrasted with dataset-text sharing so the two
// mechanisms are visibly disjoint.
const contentServerNodes: Node[] = [
  panel(
    "cs-cfg",
    "ContentServerSidecar",
    "cfg.sidecars.content_server → a run-owned native HTTP endpoint (media URLs instead of inline base64).",
    0,
    60,
  ),
  card(
    "cs-server",
    "ContentServerRuntime",
    "run-owned HTTP",
    "One server streams a path-confined directory, exposes /healthz, and retains bounded request records.",
    0,
    180,
    "orange",
  ),
  card(
    "cs-pub",
    "ContentServerMediaPublisher",
    "synthetic-media seam",
    "Publishes generated images/videos to disk and rewrites payload media into small HTTP URLs; audio stays inline base64.",
    0,
    300,
    "orange",
  ),
  header("cs-vs", "Two disjoint concerns", "text sharing vs media delivery", 380, 0),
  card(
    "cs-text",
    "Dataset TEXT sharing",
    "in-process",
    "SegmentStore Handles + Arc<dyn SegmentStore> — zero-copy across threads. No HTTP, no bytes copied.",
    380,
    90,
    "cyan",
  ),
  card(
    "cs-media",
    "MEDIA delivery",
    "out-of-band",
    "content_server HTTP URLs — a delivery transport for large binary media, entirely separate from text.",
    380,
    220,
    "orange",
  ),
];

const contentServerEdges: Edge[] = [
  flow("cs-cfg", "cs-server", "spawns"),
  flow("cs-server", "cs-pub", "publishes URLs"),
];

/**
 * Stage 3 definition. Authored as `dataset-share.tsx`; the deck's `STAGES` registry imports this
 * `StageDef` for spine ordinal 3. Keeps the stub's stable `id`/`order`/`label`/`tone` so the
 * overview layout, edge wiring, and the deck's `STAGE_LABELS` test stay unchanged.
 */
export const datasetShareStage: StageDef = {
  id: "sharing",
  order: 3,
  label: "Sharing the dataset",
  caption:
    "The frozen SegmentStore: bytes live exactly once; turns carry Handles not bytes for zero-copy sharing across worker threads (content_server is a separate media sidecar).",
  tone: TONE,
  // v2 timeline: zero-copy Handle sharing stays in the Dataset lane — bytes live once, turns carry Handles.
  lane: "dataset",
  events: [{ id: "sh-handles", label: "Handles", laneId: "dataset", atOrder: 4, realOffsetMs: 44 }],
  subgraph: {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    children: [CONTENT_SERVER_LEAF],
  },
  leaves: {
    [CONTENT_SERVER_LEAF]: {
      label: "content_server media sidecar",
      nodes: contentServerNodes,
      edges: contentServerEdges,
    },
  },
  evidence: [
    { label: "SegmentStore trait (Send + Sync)", path: "runtime/src/dataset/segment.rs:238" },
    { label: "SegmentPool::freeze", path: "runtime/src/dataset/segment.rs:514" },
    { label: "InMemorySegmentStore (frozen, shared)", path: "runtime/src/dataset/segment.rs:533" },
    { label: "Handle (dense u32)", path: "runtime/src/dataset/segment.rs:28" },
    { label: "Dataset — Arc<dyn SegmentStore>", path: "runtime/src/dataset/dataset.rs:48" },
    { label: "Turn (carries Handles)", path: "runtime/src/dataset/model.rs:216" },
    { label: "content_server (media sidecar)", path: "runtime/src/content_server/mod.rs:4" },
    { label: "ContentServerSidecar", path: "cli/src/model/telemetry.rs:229" },
  ],
};

/**
 * The play-layer step fragment for this stage: the request particle traverses the sharing path
 * (frozen store → Arc → Handle-carrying turns → worker threads) and then names content_server as
 * the disjoint media sidecar. Node ids reference this stage's own level-1 subgraph.
 */
export const datasetShareSteps: readonly FlowStep[] = [
  {
    nodeId: "share-store",
    caption:
      "Bytes are interned once into the frozen InMemorySegmentStore — content-addressed, deduplicated.",
  },
  {
    nodeId: "share-arc",
    caption:
      "The Dataset holds one Arc<dyn SegmentStore>; each worker clones the pointer — zero-copy, no byte copy.",
  },
  {
    nodeId: "share-turns",
    caption: "Conversations and Turns carry dense Handle (u32) values, never inline bytes.",
  },
  {
    nodeId: "share-w0",
    caption:
      "A worker thread resolves bytes on demand via build_body(handles) at dispatch — the arena stays shared.",
  },
  {
    nodeId: CONTENT_SERVER_LEAF,
    caption:
      "content_server is a SEPARATE run-owned media sidecar — it delivers image/video URLs, not dataset text.",
  },
];
