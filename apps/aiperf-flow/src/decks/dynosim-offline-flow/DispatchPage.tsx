/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import { Stack } from "../../layout/Stack.js";
import { Divider } from "../../layout/Divider.js";
import { inkClassName } from "../../theme/tokens.js";
import type { Level } from "./shared.js";
import { atLeast } from "./shared.js";

//! Ported from `docs/canvases/dynosim-offline-flow.canvas.tsx` `DispatchPage`: a turn becomes an
//! engine token array through a three-way priority (raw_token_ids > trace_hash_ids > text turn),
//! all converging on `dispatch_tokens` — no HTTP body is ever built. Below that, the observer
//! strip shows the callbacks the engine replies emit, same as a real HTTP request would.

function diagramNodes(maint: boolean): Node[] {
  return [
    {
      id: "raw-token-ids",
      type: "panel",
      position: { x: 0, y: 0 },
      data: { title: "raw_token_ids", detail: maint ? "resolve()" : "exact tokens" },
    },
    {
      id: "trace-hash-ids",
      type: "panel",
      position: { x: 0, y: 100 },
      data: { title: "trace_hash_ids", detail: maint ? "synthesize_tokens" : "trace blocks" },
    },
    {
      id: "text-turn",
      type: "panel",
      position: { x: 0, y: 200 },
      data: { title: "text turn", detail: maint ? "tiktoken encode" : "encode text" },
    },
    {
      id: "dispatch-tokens",
      type: "card",
      position: { x: 320, y: 100 },
      data: {
        title: "dispatch_tokens",
        subtitle: "no HTTP body",
        detail: maint ? "DirectRequest → submit" : undefined,
      },
    },
    {
      id: "engine",
      type: "card",
      position: { x: 620, y: 100 },
      data: { title: "engine" },
    },
  ];
}

const diagramEdges: Edge[] = [
  { id: "e-raw-dispatch", source: "raw-token-ids", target: "dispatch-tokens", type: "flow" },
  { id: "e-trace-dispatch", source: "trace-hash-ids", target: "dispatch-tokens", type: "flow" },
  { id: "e-text-dispatch", source: "text-turn", target: "dispatch-tokens", type: "flow" },
  { id: "e-dispatch-engine", source: "dispatch-tokens", target: "engine", type: "flow" },
];

function observerNodes(maint: boolean): Node[] {
  const labels = maint
    ? ["on_arrival", "on_admit", "on_token", "on_usage", "on_terminal"]
    : ["arrival", "admit", "token", "usage", "done"];
  return labels.map((label, idx) => ({
    id: `obs-${idx}`,
    type: "chip",
    position: { x: idx * 150, y: 0 },
    data: { label },
  }));
}

function observerEdges(): Edge[] {
  return [0, 1, 2, 3].map((idx) => ({
    id: `e-obs-${idx}`,
    source: `obs-${idx}`,
    target: `obs-${idx + 1}`,
    type: "flow",
  }));
}

/**
 * Request -> tokens -> engine page: the dispatch-priority diagram plus the observer-callback
 * strip, ported from `DispatchPage`/`ObserverStrip` in
 * `docs/canvases/dynosim-offline-flow.canvas.tsx`.
 */
export function DispatchPage({ level }: { level: Level }): React.JSX.Element {
  const maint = atLeast(level, "maintainer");
  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Request → tokens → engine</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          A turn becomes an engine token array through a three-way priority: exact token ids win,
          then recorded trace hashes, then plain text. All three converge on{" "}
          <strong>dispatch_tokens</strong> — no HTTP body is ever built.
        </p>
      </div>

      <AutoLayoutFlow key={String(maint)} nodes={diagramNodes(maint)} edges={diagramEdges} layout={{ direction: "RIGHT" }} height={320} />

      <Divider />
      <div>
        <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>What the run observes</h3>
        <p className={`mt-1 text-sm ${inkClassName("secondary")}`}>
          As the engine replies, each request emits the same callbacks a real HTTP request would —
          feeding the metrics accumulator.
        </p>
      </div>
      <AutoLayoutFlow key={String(maint)} nodes={observerNodes(maint)} edges={observerEdges()} layout={{ direction: "RIGHT" }} height={120} />
      {maint && (
        <p className={`text-center text-xs ${inkClassName("quaternary")}`}>
          on_arrival from ScheduledRuntime · the rest from DynosimSink
        </p>
      )}
    </Stack>
  );
}
