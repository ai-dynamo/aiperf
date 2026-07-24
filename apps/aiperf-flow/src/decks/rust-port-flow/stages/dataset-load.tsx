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
import { categoryBgTintClassName } from "../../../theme/tokens.js";
import type { CategoryRole } from "../../../theme/tokens.js";
import type { FlowStep } from "../../../interactive/index.js";
import type { StageDef } from "../stage.js";

/** A tinted `card` node for a level-1/level-2 subgraph, colored by a category role via the static helper. */
function card(
  id: string,
  position: { x: number; y: number },
  title: string,
  subtitle: string,
  detail: string,
  tone: CategoryRole,
): Node {
  return {
    id,
    type: "card",
    position,
    data: { title, subtitle, detail, className: categoryBgTintClassName(tone) },
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
    "Each loader (HF, trace, synthetic, raw-payload, …) reads RawRows from its source.",
    "green",
  ),
  card(
    "pool",
    { x: 250, y: 40 },
    "SegmentPool",
    "mutable interner",
    "The write side: arena Vec<Segment> plus a SegmentId→Handle map, filled as rows are lowered.",
    "green",
  ),
  card(
    "intern",
    { x: 500, y: 40 },
    "intern(parent, payload)",
    "content-address",
    "Interns one Payload under an optional prefix parent, returning a dense Handle.",
    "green",
  ),
  card(
    "domains",
    { x: 360, y: 210 },
    "Six BLAKE3 domains",
    "SegmentDomain",
    "message · text-only · raw · token-ids · media · trace-hash-ids — disjoint content domains.",
    "blue",
  ),
  card(
    "hashing",
    { x: 620, y: 210 },
    "Prefix-folded hashing",
    "payload_id + dedup",
    "A child hash folds the parent's content hash (not its index); a repeated SegmentId dedups.",
    "purple",
  ),
  card(
    "freeze",
    { x: 760, y: 40 },
    "freeze()",
    "seal the arena",
    "Drops the write-only SegmentId→Handle map and hands back an immutable store.",
    "green",
  ),
  card(
    "store",
    { x: 1010, y: 40 },
    "InMemorySegmentStore",
    "frozen arena",
    "The dense Box<[Segment]> arena — each unique segment's bytes live exactly once, shared read-only across workers.",
    "cyan",
  ),
  card(
    "turn",
    { x: 1010, y: 210 },
    "Turn",
    "body: SmallVec<[Handle; 1]>",
    "Every potentially large per-turn value is a Handle into the store; turns carry dense indices, not bytes.",
    "orange",
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
  card("dom-message", { x: 0, y: 0 }, "message", "SegmentDomain::Message", "A pre-serialized endpoint message object; these handles format as an array.", "blue"),
  card("dom-text-only", { x: 300, y: 0 }, "text-only", "SegmentDomain::TextOnly", "Plain text for non-message endpoint fields; spliced verbatim at dispatch.", "blue"),
  card("dom-raw", { x: 600, y: 0 }, "raw", "SegmentDomain::Raw", "A complete prebuilt request body — a leading raw handle bypasses endpoint formatting.", "blue"),
  card("dom-token-ids", { x: 0, y: 150 }, "token-ids", "SegmentDomain::TokenIds", "Exact pre-tokenized input IDs — the token-native dispatch path.", "blue"),
  card("dom-media", { x: 300, y: 150 }, "media", "SegmentDomain::Media", "Binary or encoded multimodal content, folded into the segment identity.", "blue"),
  card("dom-trace-hash-ids", { x: 600, y: 150 }, "trace-hash-ids", "SegmentDomain::TraceHashIds", "Authored source-trace block identities carried by a Turn's trace_hash_ids.", "blue"),
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
    "Feeds the parent segment's content hash into BLAKE3, so shared prefixes converge to shared ids.",
    "purple",
  ),
  card(
    "hash-payload",
    { x: 280, y: 40 },
    "payload_id(parent, payload)",
    "BLAKE3",
    "Hashes HASH_VERSION + the domain tag + the folded parent hash + the payload content into a SegmentId.",
    "purple",
  ),
  card(
    "hash-dedup",
    { x: 560, y: 40 },
    "push_interned → ids map",
    "dedup",
    "A SegmentId already in the map returns its existing Handle; otherwise a fresh dense index is appended.",
    "purple",
  ),
  card(
    "hash-handle",
    { x: 840, y: 40 },
    "Handle(u32)",
    "dense index",
    "The public address is a dense arena index; the SegmentId→Handle map only exists until freeze().",
    "green",
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
