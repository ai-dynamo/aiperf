/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";

// Ported from docs/canvases/segment-pools-and-body-plans.canvas.tsx `PagePrefix`
// (rust/aiperf/src/dataset/compose.rs, rust/aiperf/src/graph/recorded/trie/).
// A shared root chain (H0, H1) fans out into two conversation branches (C1, C2)
// that each reuse the shared prefix by handle instead of re-storing it.

const nodes: Node[] = [
  {
    id: "h0",
    type: "card",
    position: { x: 0, y: 140 },
    data: { title: "H0 · system", subtitle: "You are helpful.", detail: "shared once" },
  },
  {
    id: "h1",
    type: "card",
    position: { x: 260, y: 140 },
    data: { title: "H1 · user", subtitle: "What is 2+2?", detail: "shared once" },
  },
  {
    id: "h2",
    type: "card",
    position: { x: 560, y: 40 },
    data: { title: "H2 · assistant", subtitle: '"4"', detail: "novel" },
  },
  {
    id: "c1-request",
    type: "panel",
    position: { x: 820, y: 40 },
    data: { title: "C1 request", detail: "H0 · H1 · H2" },
  },
  {
    id: "h3",
    type: "card",
    position: { x: 560, y: 260 },
    data: { title: "H3 · assistant", subtitle: '"It equals four."', detail: "novel" },
  },
  {
    id: "c2-request",
    type: "panel",
    position: { x: 820, y: 260 },
    data: { title: "C2 request", detail: "H0 · H1 · H3" },
  },
];

const edges: Edge[] = [
  {
    id: "e-h0-h1",
    source: "h0",
    target: "h1",
    type: "flow",
    label: "shared prefix (interned once)",
  },
  { id: "e-h1-h2", source: "h1", target: "h2", type: "flow" },
  { id: "e-h1-h3", source: "h1", target: "h3", type: "flow" },
  { id: "e-h2-c1", source: "h2", target: "c1-request", type: "flow" },
  { id: "e-h3-c2", source: "h3", target: "c2-request", type: "flow" },
];

/**
 * Prefix trie / content-addressing page of the Segment Pools & Body Plans deck.
 *
 * Ports `PagePrefix` from `docs/canvases/segment-pools-and-body-plans.canvas.tsx`
 * onto aiperf-flow's node/edge vocabulary: the hand-drawn SVG boxes and animated
 * paths become `card`/`panel` nodes chained by `flow` edges. Shows two
 * conversations that share a system+user prefix (H0, H1), stored once, with
 * each branch (C1, C2) referencing the shared handles instead of re-storing
 * them — the same shape recorded traces get from the LCP trie over block
 * hashes in `graph/recorded/trie/`.
 */
export function PrefixPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Prefix chains &amp; LCP-trie lowering</h2>
        <p className="mt-1 max-w-3xl text-sm text-[var(--color-ink-secondary)]">
          Composers keep a running <code>parent: Option&lt;Handle&gt;</code> per conversation, so each turn extends a
          chain. Recorded traces (WEKA / Dynamo) go further: an LCP trie over block hashes finds the longest shared
          prefix, and the <code>PromptMessageCache</code> keyed on <code>(parent, role, block_hashes)</code> reuses
          the decoded, interned handle for any prefix two nodes share.
        </p>
      </div>

      <AutoLayoutFlow nodes={nodes} edges={edges} layout={{ direction: "RIGHT" }} height={480} />

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-[var(--color-stroke-secondary)] px-4 py-3 text-sm shadow-sm">
          <div className="font-semibold">resolve_content_parents</div>
          <p className="mt-1 text-[var(--color-ink-secondary)]">
            Walks each node's <code>hash_ids</code> along trie edges; the longest match yields the{" "}
            <code>content_parent</code>. Prefers the latest full-prefix terminal, else the earliest partial passer.
          </p>
          <div className="mt-2 text-xs text-[var(--color-ink-tertiary)]">graph/recorded/trie/parents.rs:18</div>
        </div>
        <div className="rounded-lg border border-[var(--color-stroke-secondary)] px-4 py-3 text-sm shadow-sm">
          <div className="font-semibold">rebase on context injection</div>
          <p className="mt-1 text-[var(--color-ink-secondary)]">
            When a system / user_context is injected after compose, <code>rebase_conversation_handles</code> re-interns
            every handle under the new root so blake3 ids reflect the shared prefix — content unchanged, identity
            refreshed.
          </p>
          <div className="mt-2 text-xs text-[var(--color-ink-tertiary)]">dataset/compose.rs:338</div>
        </div>
      </div>
    </div>
  );
}
