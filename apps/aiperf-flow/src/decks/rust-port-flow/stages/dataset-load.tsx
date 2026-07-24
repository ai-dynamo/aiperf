/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stage 2 — Dataset loading (detail module).
//!
//! Tells the load-time content-lowering story for one benchmark dataset:
//!   `DatasetLoader::load` → `SegmentPool` (the mutable interner) → prefix-folded BLAKE3 `intern`
//!   with dedup → `freeze()` into an immutable `InMemorySegmentStore` (dense arena) → dense integer
//!   `Handle`s → a frozen `Turn` whose large values are all `Handle`s, not bytes.
//!
//! Level-1 subgraph = the load pipeline. Two level-2 leaves drill deeper: `domains` (the six
//! disjoint BLAKE3 content domains) and `hashing` (prefix-folded `payload_id` + dedup → dense
//! `Handle`). Every anchor in `evidence` was pinned against the real `rust/runtime/src/dataset`
//! source (verified `file:line`, not the spec markdown).

import type { Edge, Node } from "@xyflow/react";
import { roleClassName } from "../stage.js";
import type { NodeRole } from "../stage.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";

/** A tinted `card` node for a level-1/level-2 subgraph, colored by its semantic node role. */
function card(
  id: string,
  position: { x: number; y: number },
  title: string,
  subtitle: string,
  detail: string,
  role: NodeRole,
): Node {
  return {
    id,
    type: "card",
    position,
    data: { title, subtitle, detail, className: roleClassName(role) },
  };
}

/** An animated data-movement edge along the load pipeline. */
function flowEdge(source: string, target: string): Edge {
  return { id: `e-${source}-${target}`, source, target, type: "flow" };
}

/** A plain (non-animated) classification/reference connector. */
function linkEdge(source: string, target: string): Edge {
  return { id: `e-${source}-${target}`, source, target };
}

// ---------------------------------------------------------------------------
// Level-1 subgraph: the load pipeline.
// ---------------------------------------------------------------------------

const subgraphNodes: Node[] = [
  card(
    "loaders",
    { x: 0, y: 40 },
    "DatasetLoader::load",
    "loaders",
    "Each loader (HF, trace, synthetic, raw) reads RawRows.",
    "compute",
  ),
  card(
    "pool",
    { x: 250, y: 40 },
    "SegmentPool",
    "mutable interner",
    "Write side: arena Vec<Segment> + SegmentId→Handle map.",
    "storage",
  ),
  card(
    "intern",
    { x: 500, y: 40 },
    "intern(parent, payload)",
    "content-address",
    "Interns a Payload under a prefix parent → dense Handle.",
    "compute",
  ),
  card(
    "domains",
    { x: 360, y: 210 },
    "Six BLAKE3 domains",
    "SegmentDomain",
    "message · text · raw · token-ids · media · trace-hash — disjoint.",
    "storage",
  ),
  card(
    "hashing",
    { x: 620, y: 210 },
    "Prefix-folded hashing",
    "payload_id + dedup",
    "Child hash folds parent's content hash; repeats dedup.",
    "compute",
  ),
  card(
    "freeze",
    { x: 760, y: 40 },
    "freeze()",
    "seal the arena",
    "Drops the write map, hands back an immutable store.",
    "compute",
  ),
  card(
    "store",
    { x: 1010, y: 40 },
    "InMemorySegmentStore",
    "frozen arena",
    "Dense Box<[Segment]>; bytes live once, shared read-only.",
    "storage",
  ),
  card(
    "turn",
    { x: 1010, y: 210 },
    "Turn",
    "body: SmallVec<[Handle; 1]>",
    "Large values are Handles into the store — indices, not bytes.",
    "storage",
  ),
];

const subgraphEdges: Edge[] = [
  flowEdge("loaders", "pool"),
  flowEdge("pool", "intern"),
  flowEdge("intern", "freeze"),
  flowEdge("freeze", "store"),
  linkEdge("intern", "domains"),
  linkEdge("intern", "hashing"),
  linkEdge("store", "turn"),
];

// ---------------------------------------------------------------------------
// Level-2 leaf: the six disjoint BLAKE3 content domains (SegmentDomain variants).
// ---------------------------------------------------------------------------

const domainLeafNodes: Node[] = [
  card("dom-message", { x: 0, y: 0 }, "message", "SegmentDomain::Message", "Pre-serialized endpoint message; formats as an array.", "storage"),
  card("dom-text-only", { x: 300, y: 0 }, "text-only", "SegmentDomain::TextOnly", "Plain text field; spliced verbatim at dispatch.", "storage"),
  card("dom-raw", { x: 600, y: 0 }, "raw", "SegmentDomain::Raw", "Prebuilt body; leading raw handle bypasses formatting.", "storage"),
  card("dom-token-ids", { x: 0, y: 150 }, "token-ids", "SegmentDomain::TokenIds", "Pre-tokenized input IDs — token-native dispatch.", "storage"),
  card("dom-media", { x: 300, y: 150 }, "media", "SegmentDomain::Media", "Binary/encoded multimodal content, folded into identity.", "media"),
  card("dom-trace-hash-ids", { x: 600, y: 150 }, "trace-hash-ids", "SegmentDomain::TraceHashIds", "Source-trace block ids on a Turn's trace_hash_ids.", "storage"),
];

// ---------------------------------------------------------------------------
// Level-2 leaf: prefix-folded hashing + dedup → dense Handle.
// ---------------------------------------------------------------------------

const hashingLeafNodes: Node[] = [
  card(
    "hash-parent",
    { x: 0, y: 40 },
    "hash_parent(parent)",
    "prefix fold",
    "Folds parent's content hash in; shared prefixes → shared ids.",
    "compute",
  ),
  card(
    "hash-payload",
    { x: 280, y: 40 },
    "payload_id(parent, payload)",
    "BLAKE3",
    "Hashes version + domain tag + parent hash + payload → SegmentId.",
    "compute",
  ),
  card(
    "hash-dedup",
    { x: 560, y: 40 },
    "push_interned → ids map",
    "dedup",
    "Known SegmentId returns its Handle; else append a dense index.",
    "compute",
  ),
  card(
    "hash-handle",
    { x: 840, y: 40 },
    "Handle(u32)",
    "dense index",
    "Public address is a dense arena index; map lives until freeze().",
    "storage",
  ),
];

const hashingLeafEdges: Edge[] = [
  flowEdge("hash-parent", "hash-payload"),
  flowEdge("hash-payload", "hash-dedup"),
  flowEdge("hash-dedup", "hash-handle"),
];

/**
 * The play-layer fragment for this stage: an animated particle traverses the load pipeline, each
 * step naming the real type/fn on the active node. Node ids match the level-1 subgraph.
 */
export const datasetFlowSteps: FlowStep[] = [
  {
    nodeId: "loaders",
    caption: "DatasetLoader::load reads RawRows from the source (HF, trace, synthetic, raw-payload).",
    timingMs: 400,
  },
  {
    nodeId: "pool",
    caption: "Rows are interned into a mutable SegmentPool — an arena Vec<Segment> plus a SegmentId→Handle map.",
    timingMs: 400,
  },
  {
    nodeId: "intern",
    caption:
      "intern() hashes each payload with its parent's content folded in (prefix-folded BLAKE3); a repeated SegmentId dedups to the existing Handle.",
    timingMs: 500,
  },
  {
    nodeId: "freeze",
    caption: "freeze() drops the write-side SegmentId→Handle map and returns an immutable InMemorySegmentStore.",
    timingMs: 400,
  },
  {
    nodeId: "store",
    caption: "The frozen InMemorySegmentStore arena holds each unique segment's bytes exactly once, shared read-only across workers.",
    timingMs: 400,
  },
  {
    nodeId: "turn",
    caption: "Each large value on a Turn is a dense Handle into the store — body: SmallVec<[Handle; 1]> — so turns carry indices, not bytes.",
    timingMs: 400,
  },
];

/**
 * Stage 2 — Dataset loading. Drop-in `StageDef` for the `rust-port-flow` deck registry (the deck
 * shell imports this symbol). Fills in the level-1 `subgraph`, two level-2 `leaves`, and verified
 * `evidence` anchors on top of the stub's id/order/label/caption/tone.
 */
export const datasetStage: StageDef = {
  id: "dataset",
  order: 2,
  label: "Dataset loading",
  caption:
    "Loaders → SegmentStore (six disjoint BLAKE3 content domains, prefix-folded hashing) → dense integer Handles → Turn/body freeze.",
  tone: "green",
  // v2 timeline: dataset load/freeze in the Dataset lane — the SegmentStore freeze point.
  lane: "dataset",
  events: [{ id: "ds-freeze", label: "freeze", laneId: "dataset", atOrder: 3, realOffsetMs: 38 }],
  subgraph: {
    nodes: subgraphNodes,
    edges: subgraphEdges,
    children: ["domains", "hashing"],
  },
  leaves: {
    domains: {
      label: "Six BLAKE3 content domains",
      nodes: domainLeafNodes,
      edges: [],
    },
    hashing: {
      label: "Prefix-folded hashing & dedup",
      nodes: hashingLeafNodes,
      edges: hashingLeafEdges,
    },
  },
  evidence: [
    { label: "SegmentStore seam", path: "runtime/src/dataset/segment.rs:238" },
    { label: "SegmentPool interner", path: "runtime/src/dataset/segment.rs:278" },
    { label: "SegmentPool::intern (dedup)", path: "runtime/src/dataset/segment.rs:319" },
    { label: "prefix-folded payload_id", path: "runtime/src/dataset/segment.rs:554" },
    { label: "SegmentDomain (6 domains)", path: "runtime/src/dataset/segment.rs:168" },
    { label: "freeze → InMemorySegmentStore", path: "runtime/src/dataset/segment.rs:514" },
    { label: "dense Handle(u32)", path: "runtime/src/dataset/segment.rs:28" },
    { label: "Turn.body: [Handle]", path: "runtime/src/dataset/model.rs:282" },
    { label: "DatasetLoader::load", path: "runtime/src/dataset/loader/mod.rs:299" },
  ],
};
