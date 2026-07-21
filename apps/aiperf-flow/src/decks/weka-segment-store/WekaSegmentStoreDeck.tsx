/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Pill as Badge } from "../../prose/Pill.js";
import { TopBar } from "../../shell/TopBar.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";

//! Ported from docs/canvases/weka-segment-store.canvas.tsx (single-view Cursor canvas, no
//! internal page tabs). Content plane: content-addressed segment pool dedup, the unified
//! on-disk store (content.idx/content.blob + nodes.idx/nodes.blob), and A1-vs-A2 format
//! detection at read time (rust/aiperf-graph segment store, WEKA_UNIFIED_STORE).

/** One-off local segment-id chip, scoped to this deck (ported from the canvas's `SegChip`). */
function SegChip({ id, shared }: { id: string; shared?: boolean }): React.JSX.Element {
  return (
    <div
      className={clsx(
        "min-w-[40px] rounded-md border px-2.5 py-1.5 text-center shadow-sm",
        strokeClassName(shared ? "primary" : "secondary"),
        shared ? surfaceClassName("elevated") : surfaceClassName("panel"),
      )}
    >
      <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{id}</span>
    </div>
  );
}

/** One-off local labeled row of segment-id chips, scoped to this deck (ported from the canvas's `PathRow`). */
function PathRow({
  label,
  ids,
}: {
  label: string;
  ids: Array<{ id: string; shared?: boolean }>;
}): React.JSX.Element {
  return (
    <Row gap={10} align="center" wrap>
      <div className="min-w-[78px]">
        <span className={clsx("text-sm font-medium", inkClassName("tertiary"))}>{label}</span>
      </div>
      <Row gap={6} align="center" wrap>
        {ids.map((s, i) => (
          <Row key={s.id} gap={6} align="center">
            <SegChip id={s.id} shared={s.shared} />
            {i < ids.length - 1 && <span className={clsx("text-sm", inkClassName("quaternary"))}>-&gt;</span>}
          </Row>
        ))}
      </Row>
    </Row>
  );
}

/** One-off local branch card, scoped to this deck (ported from the canvas's `BranchCard`). */
function BranchCard({
  tag,
  title,
  children,
}: {
  tag: string;
  title: string;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className={clsx("rounded-lg border px-4 py-3 shadow-sm", strokeClassName("secondary"), surfaceClassName("elevated"))}>
      <Row align="center" justify="space-between" gap={10}>
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{title}</span>
        <Badge>{tag}</Badge>
      </Row>
      <div className="mt-2">{children}</div>
    </div>
  );
}

const unifiedStoreNodes: Node[] = [
  {
    id: "unified-dir",
    type: "header",
    position: { x: 0, y: 0 },
    data: { title: "aiperf_graph_segments_<benchmark_id>/" },
  },
  {
    id: "content-idx",
    type: "panel",
    position: { x: 0, y: 70 },
    data: { title: "content.idx", detail: "hex map (A1)  |  packed 'Q' handles (A2)" },
  },
  {
    id: "content-blob",
    type: "panel",
    position: { x: 0, y: 190 },
    data: { title: "content.blob", detail: "{role,content} JSON blobs, concatenated" },
  },
  {
    id: "nodes-idx",
    type: "panel",
    position: { x: 340, y: 70 },
    data: { title: "nodes.idx", detail: "trace_id -> ordinal:variant -> [off,size]" },
  },
  {
    id: "nodes-blob",
    type: "panel",
    position: { x: 340, y: 190 },
    data: { title: "nodes.blob", detail: "per-node envelopes, concatenated" },
  },
];

const unifiedStoreEdges: Edge[] = [
  {
    id: "e-content-idx-blob",
    source: "content-idx",
    target: "content-blob",
    type: "flow",
    label: "span [off,size]",
  },
  {
    id: "e-nodes-idx-blob",
    source: "nodes-idx",
    target: "nodes-blob",
    type: "flow",
    label: "envelope [off,size]",
  },
];

/**
 * Weka content plane and unified segment store explainer deck.
 *
 * Ports `WekaSegmentStore` (default export) from
 * `docs/canvases/weka-segment-store.canvas.tsx` — a single-view Cursor canvas with no internal
 * page tabs — onto aiperf-flow's node/edge/prose vocabulary. Three sections: the
 * content-addressed segment pool that deduplicates shared conversation prefixes, the unified
 * on-disk store (one directory, four mmap'd files), and the A1-vs-A2 format detection branch
 * a reader takes based on the first byte of `content.idx`.
 */
export function WekaSegmentStoreDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Segment Store" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-5xl bg-surface-page px-10 py-8">
          <Stack gap={26}>
            <Stack gap={10}>
              <Row align="center" gap={10} wrap>
                <h1 className={clsx("text-2xl font-bold", inkClassName("primary"))}>
                  Weka content plane and the unified segment store
                </h1>
                <Badge active>WEKA_UNIFIED_STORE</Badge>
              </Row>
              <p className={clsx("text-sm", inkClassName("secondary"))}>
                The content plane deduplicates prompt/response segments by content-addressed id. The
                unified store folds the content pool and the graph-delta addressing store into one
                directory a worker opens as a single client.
              </p>
            </Stack>

            <Divider />

            <Stack gap={10}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                Content-addressed segment pool
              </h2>
              <p className={clsx("text-sm", inkClassName("secondary"))}>
                Each segment id is a <strong>prefix-dependent</strong> blake2b hash of (parent_id, role,
                token-ids). Shared conversation prefixes hash to the same ids, so two turns that share
                history point at <strong>the same pool entries</strong> — the pool stores each unique
                segment once.
              </p>
              <div className={clsx("rounded-lg border shadow-sm", strokeClassName("secondary"), surfaceClassName("elevated"))}>
                <Row
                  align="center"
                  justify="space-between"
                  gap={10}
                  className={clsx("border-b px-4 py-3", strokeClassName("secondary"))}
                >
                  <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>
                    Two turns sharing a prefix
                  </span>
                  <Badge>accent = shared</Badge>
                </Row>
                <div className="px-4 py-3">
                  <Stack gap={12}>
                    <PathRow
                      label="r2 prompt"
                      ids={[
                        { id: "s0", shared: true },
                        { id: "s1", shared: true },
                        { id: "s2", shared: true },
                      ]}
                    />
                    <PathRow
                      label="r4 prompt"
                      ids={[
                        { id: "s0", shared: true },
                        { id: "s1", shared: true },
                        { id: "s2", shared: true },
                        { id: "s3" },
                        { id: "s4" },
                      ]}
                    />
                    <Divider />
                    <Row gap={10} align="center" wrap>
                      <div className="min-w-[78px]">
                        <span className={clsx("text-sm font-medium", inkClassName("tertiary"))}>SegmentPool</span>
                      </div>
                      <Row gap={6} wrap>
                        <SegChip id="s0" shared />
                        <SegChip id="s1" shared />
                        <SegChip id="s2" shared />
                        <SegChip id="s3" />
                        <SegChip id="s4" />
                      </Row>
                    </Row>
                  </Stack>
                </div>
              </div>
              <Callout tone="info" title="What a node carries">
                Each LlmNode stores <strong>metadata.trie.prompt_segment_ids</strong> (the path) and{" "}
                <strong>response_id</strong> (its assistant output segment). The worker rebuilds the
                request body from these ids/handles — never from predecessor channel values.
              </Callout>
            </Stack>

            <Stack gap={10}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                Unified store — one directory, four files
              </h2>
              <p className={clsx("text-sm", inkClassName("secondary"))}>
                <strong>GraphSegmentUnifiedClient</strong> duck-types both the delta-reader and the
                segment-client faces. Blobs are mmap&apos;d read-only so every worker shares one physical
                copy.
              </p>
              <div className={clsx("rounded-lg border shadow-sm", strokeClassName("secondary"), surfaceClassName("elevated"))}>
                <Row
                  align="center"
                  justify="space-between"
                  gap={10}
                  className={clsx("border-b px-4 py-3", strokeClassName("secondary"))}
                >
                  <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>On-disk layout</span>
                  <Badge>mmap ACCESS_READ</Badge>
                </Row>
                <div style={{ height: 320 }}>
                  <ReactFlow
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    nodes={unifiedStoreNodes}
                    edges={unifiedStoreEdges}
                    fitView
                    fitViewOptions={{ padding: 0.2 }}
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
                  </ReactFlow>
                </div>
              </div>
            </Stack>

            <Stack gap={10}>
              <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
                A1 vs A2 — detected from the first byte of content.idx
              </h2>
              <div
                className={clsx(
                  "self-start rounded-lg border px-3 py-2 shadow-sm",
                  strokeClassName("primary"),
                  surfaceClassName("elevated"),
                )}
              >
                <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>
                  _load_content_idx: first byte == &apos;&#123;&apos; ?
                </span>
              </div>
              <Grid columns={2} gap={16}>
                <BranchCard tag="_interned = False" title="A1 hex composition">
                  <Stack gap={6}>
                    <p className={clsx("text-sm", inkClassName("primary"))}>
                      content.idx is a JSON object{" "}
                      <strong>{"{ids: {hex_id: handle}, spans: [[off,size]]}"}</strong>. Byte-identical to
                      the legacy mmap content pool.
                    </p>
                    <p className={clsx("text-sm", inkClassName("tertiary"))}>
                      Envelopes carry hex <strong>prompt_segment_ids</strong>. Worker runs the dict
                      materialize path.
                    </p>
                  </Stack>
                </BranchCard>
                <BranchCard tag="_interned = True" title="A2 packed int handles">
                  <Stack gap={6}>
                    <p className={clsx("text-sm", inkClassName("primary"))}>
                      content.idx is a raw <strong>array(&apos;Q&apos;)</strong> of
                      [off0,size0,off1,size1,...] — mmap-friendly, no JSON parse.
                    </p>
                    <p className={clsx("text-sm", inkClassName("tertiary"))}>
                      Envelopes carry int <strong>handles</strong>. Worker restores pre-serialized bytes
                      from mmap slices.
                    </p>
                  </Stack>
                </BranchCard>
              </Grid>
              <Callout tone="warning" title="Worker branches on the store-level _interned flag">
                Selection is the reader&apos;s <strong>_interned</strong> flag (set when content.idx is
                packed) — not which envelope key is present, and not whether a materialize function
                returned non-None. An interned-store None is a genuine miss; a non-interned unified store
                falls through to the A1 dict path unchanged.
              </Callout>
            </Stack>

            <Callout tone="info" title="Phase A scope">
              The unified store is wired only into the eager local-file/dir trie build (segment_pool is
              not None). Combining it with WEKA_SEGMENT_TRIE_IR=False or an HF-id source is rejected at
              configure time, and a trie run never consults graph_delta_cache — the interned handles are
              insertion-index-local to one build (cross-cache handle binding is Phase B).
            </Callout>
          </Stack>
        </div>
      </div>
    </div>
  );
}
