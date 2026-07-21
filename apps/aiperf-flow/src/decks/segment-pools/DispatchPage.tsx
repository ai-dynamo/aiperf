/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";

// Ported from docs/canvases/segment-pools-and-body-plans.canvas.tsx `PageDispatch`
// (rust/aiperf/src/dataset/model.rs, request.rs:288), with node shape and copy
// aligned to apps/explainers/decks-flow/segment-pools.flow's final slide,
// "Turn.body precedence is domain-driven". Turn.body fans out to three
// precedence-check nodes (Raw / TokenIds / Messages), which converge into a
// BodyPlan -> Bytes materializer.

const nodes: Node[] = [
  {
    id: "turn-body",
    type: "card",
    position: { x: 0, y: 140 },
    data: { title: "Turn.body", subtitle: "SmallVec<[Handle]>", detail: "dispatch precedence" },
  },

  {
    id: "raw",
    type: "panel",
    position: { x: 320, y: 0 },
    data: { title: "Raw handle first?", detail: "→ complete body" },
  },
  {
    id: "token-ids",
    type: "panel",
    position: { x: 320, y: 140 },
    data: { title: "TokenIds handle?", detail: "→ token-native" },
  },
  {
    id: "messages",
    type: "panel",
    position: { x: 320, y: 280 },
    data: { title: "Message handles", detail: "→ format as array" },
  },

  {
    id: "body-plan",
    type: "card",
    position: { x: 640, y: 140 },
    data: { title: "BodyPlan", subtitle: "raw · cached · format", detail: "merge_overrides" },
  },
  {
    id: "bytes",
    type: "card",
    position: { x: 900, y: 140 },
    data: { title: "Bytes", detail: "→ wire" },
  },
];

const edges: Edge[] = [
  { id: "e-body-raw", source: "turn-body", target: "raw", type: "flow" },
  { id: "e-body-token", source: "turn-body", target: "token-ids", type: "flow" },
  { id: "e-body-messages", source: "turn-body", target: "messages", type: "flow" },
  { id: "e-raw-plan", source: "raw", target: "body-plan", type: "flow" },
  { id: "e-token-plan", source: "token-ids", target: "body-plan", type: "flow" },
  { id: "e-messages-plan", source: "messages", target: "body-plan", type: "flow" },
  {
    id: "e-plan-bytes",
    source: "body-plan",
    target: "bytes",
    type: "flow",
    label: "materialize",
    data: { speed: "slow" },
  },
];

/**
 * Dispatch precedence page of the Segment Pools & Body Plans explainer deck.
 *
 * Ports `PageDispatch` from `docs/canvases/segment-pools-and-body-plans.canvas.tsx`
 * onto aiperf-flow's node/edge vocabulary, matching the node shape already
 * established for this content in `apps/explainers/decks-flow/segment-pools.flow`'s
 * final slide: `Turn.body` fans out to three precedence-check nodes (Raw,
 * TokenIds, Messages), which converge into a `BodyPlan` materializer that
 * emits wire `Bytes`.
 */
export function DispatchPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Dispatch — one precedence vector, domain-driven</h2>
        <p className="mt-1 max-w-3xl text-sm text-[var(--color-ink-secondary)]">
          A <code>Turn</code> stores large data only as handles. <code>Turn.body</code> is the single dispatch
          precedence vector; the <strong>domain</strong> of its first handle decides how the request body is built —
          replacing the old five-field precedence.
        </p>
      </div>

      <div style={{ height: 480 }}>
        <ReactFlow nodeTypes={nodeTypes} edgeTypes={edgeTypes} nodes={nodes} edges={edges} fitView>
        </ReactFlow>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-none border border-[var(--color-stroke-secondary)] px-4 py-3 text-sm">
          <div className="font-semibold">dispatch_body precedence</div>
          <pre className="mt-2 overflow-x-auto text-xs leading-[17px] text-[var(--color-ink-secondary)]">
            {`pub fn dispatch_body(
    raw_payload: Option<Handle>,
    raw_token_ids: Option<Handle>,
    messages: &[Handle],
) -> SmallVec<[Handle; 1]> {
    let mut body = SmallVec::new();
    if let Some(raw) = raw_payload { body.push(raw); }
    if let Some(tok) = raw_token_ids { body.push(tok); }
    if raw_payload.is_none()
        && raw_token_ids.is_none() {
        body.extend_from_slice(messages);
    }
    body
}`}
          </pre>
        </div>

        <div className="rounded-none border border-[var(--color-stroke-secondary)] px-4 py-3 text-sm">
          <div className="font-semibold">The two seams this feeds</div>
          <div className="mt-2 space-y-2 text-[var(--color-ink-secondary)]">
            <p>
              <code>RequestSink&lt;R&gt;::dispatch</code> drives materialized bytes to terminal, emitting arrival /
              token / usage through a <code>RequestObserver</code>.
            </p>
            <p>
              Graph HTTP dispatch skips <code>BodyPlan</code> entirely — it splices message wires directly via{" "}
              <code>build_message_body_from_wire_parts</code>. Same store, different splice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
