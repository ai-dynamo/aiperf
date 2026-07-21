/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { inkClassName } from "../../theme/tokens.js";

//! Ported from
//! ~/.cursor/projects/home-anthony-nvidia-projects-aiperf-ajc-weka-ir-v1/canvases/weka-timing-transforms.canvas.tsx
//! (a real, hand-authored Cursor Canvas with two SVG diagrams built by hand: `IdleWarp` and
//! `TStarChop`). Single-view deck, no `PageTabs` — the source canvas is one page. The hand-drawn
//! SVG bars/nodes become real React Flow `card`/`chip`/`header` nodes positioned on a time axis;
//! the hand-drawn arrows become `flow` edges.

// --- Idle-gap warp diagram ---------------------------------------------------------------
// x(t) = LEFT + t * PX mirrors the source canvas's own time-to-pixel scale function.
const IDLE_LEFT = 40;
const IDLE_PX = 22;
const idleX = (t: number): number => IDLE_LEFT + t * IDLE_PX;

const idleWarpNodes: Node[] = [
  { id: "idle-row-raw", type: "header", position: { x: -160, y: 30 }, data: { title: "raw" } },
  { id: "idle-row-warped", type: "header", position: { x: -160, y: 210 }, data: { title: "warped" } },

  // raw row: active intervals A-D, idle 26s gap between B and C
  { id: "idle-raw-a", type: "card", position: { x: idleX(0), y: 30 }, data: { title: "A", detail: "0s – 2s" } },
  { id: "idle-raw-b", type: "card", position: { x: idleX(3), y: 30 }, data: { title: "B", detail: "3s – 4s" } },
  {
    id: "idle-raw-gap",
    type: "chip",
    position: { x: idleX(15), y: -20 },
    data: { label: "idle 26s > cap", strokeRole: "tertiary" },
  },
  { id: "idle-raw-c", type: "card", position: { x: idleX(30), y: 30 }, data: { title: "C", detail: "30s – 32s" } },
  { id: "idle-raw-d", type: "card", position: { x: idleX(33), y: 30 }, data: { title: "D", detail: "33s – 34s" } },

  // warped row: same intervals, idle collapsed to the 5s cap
  { id: "idle-warp-a", type: "card", position: { x: idleX(0), y: 210 }, data: { title: "A", detail: "0s – 2s" } },
  { id: "idle-warp-b", type: "card", position: { x: idleX(3), y: 210 }, data: { title: "B", detail: "3s – 4s" } },
  {
    id: "idle-warp-cap",
    type: "chip",
    position: { x: idleX(6), y: 160 },
    data: { label: "cap 5s", strokeRole: "secondary" },
  },
  { id: "idle-warp-c", type: "card", position: { x: idleX(9), y: 210 }, data: { title: "C", detail: "9s – 11s" } },
  { id: "idle-warp-d", type: "card", position: { x: idleX(12), y: 210 }, data: { title: "D", detail: "12s – 13s" } },
];

const idleWarpEdges: Edge[] = [
  { id: "e-idle-raw-ab", source: "idle-raw-a", target: "idle-raw-b", type: "flow" },
  { id: "e-idle-raw-bc", source: "idle-raw-b", target: "idle-raw-c", type: "flow", data: { speed: "slow" } },
  { id: "e-idle-raw-cd", source: "idle-raw-c", target: "idle-raw-d", type: "flow" },
  { id: "e-idle-warp-ab", source: "idle-warp-a", target: "idle-warp-b", type: "flow" },
  { id: "e-idle-warp-bc", source: "idle-warp-b", target: "idle-warp-c", type: "flow" },
  { id: "e-idle-warp-cd", source: "idle-warp-c", target: "idle-warp-d", type: "flow" },
];

// --- t* snapshot chop diagram ------------------------------------------------------------
const CHOP_LEFT = 40;
const CHOP_PX = 22;
const chopX = (t: number): number => CHOP_LEFT + t * CHOP_PX;
const T_STAR = 10;
// arrival offsets (warped seconds) for n0..n5, matching the source canvas's `nodes` array
const CHOP_ARRIVALS = [0, 5, 9, 12, 20, 26];

const chopNodes: Node[] = [
  { id: "chop-row-before", type: "header", position: { x: -160, y: 30 }, data: { title: "before" } },
  { id: "chop-row-after", type: "header", position: { x: -160, y: 220 }, data: { title: "after" } },
  {
    id: "chop-tstar",
    type: "chip",
    position: { x: chopX(T_STAR), y: -20 },
    data: { label: "t* = 10s", strokeRole: "primary" },
  },
  ...CHOP_ARRIVALS.map((t, i) => ({
    id: `chop-before-n${i}`,
    type: "card",
    position: { x: chopX(t), y: 30 },
    data: {
      title: `n${i}`,
      detail: `arrival ${t}s`,
      strokeRole: t < T_STAR ? ("tertiary" as const) : ("primary" as const),
    },
  })),

  { id: "chop-after-start", type: "card", position: { x: 6, y: 220 }, data: { title: "START", detail: "re-root" } },
  {
    id: "chop-after-n3",
    type: "card",
    position: { x: chopX(CHOP_ARRIVALS[3]), y: 220 },
    data: { title: "n3", detail: "min_start_delay = arrival − t*" },
  },
  { id: "chop-after-n4", type: "card", position: { x: chopX(CHOP_ARRIVALS[4]), y: 220 }, data: { title: "n4", detail: "arrival 20s" } },
  { id: "chop-after-n5", type: "card", position: { x: chopX(CHOP_ARRIVALS[5]), y: 220 }, data: { title: "n5", detail: "arrival 26s" } },
];

const chopEdges: Edge[] = [
  { id: "e-chop-before-01", source: "chop-before-n0", target: "chop-before-n1", type: "flow" },
  { id: "e-chop-before-12", source: "chop-before-n1", target: "chop-before-n2", type: "flow" },
  { id: "e-chop-before-23", source: "chop-before-n2", target: "chop-before-n3", type: "flow" },
  { id: "e-chop-before-34", source: "chop-before-n3", target: "chop-before-n4", type: "flow" },
  { id: "e-chop-before-45", source: "chop-before-n4", target: "chop-before-n5", type: "flow" },
  {
    id: "e-chop-after-start-n3",
    source: "chop-after-start",
    target: "chop-after-n3",
    type: "flow",
    label: "re-rooted, t*-relative offset",
  },
  { id: "e-chop-after-n34", source: "chop-after-n3", target: "chop-after-n4", type: "flow" },
  { id: "e-chop-after-n45", source: "chop-after-n4", target: "chop-after-n5", type: "flow" },
];

/**
 * Weka timing transforms deck: two transforms that sit between recorded timestamps and
 * dispatch, ported from the hand-authored `weka-timing-transforms.canvas.tsx` Cursor Canvas.
 *
 * The <strong>idle-gap warp</strong> collapses dead air between request active intervals down
 * to a cap without ever cutting inside a request, and the <strong>t* snapshot chop</strong>
 * drops pre-t* warmup turns from a recorded trie and re-roots survivors from START at a
 * t*-relative offset so a trace can resume mid-stream.
 */
export function WekaTimingTransformsDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Timing Transforms" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Stack gap={26}>
            <Stack gap={10}>
              <Row align="center" gap={10} wrap>
                <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>Weka timing transforms</h1>
                <span className="rounded-none border border-stroke-primary bg-surface-elevated px-2 py-0.5 text-xs font-semibold text-ink-primary">
                  warped clock
                </span>
              </Row>
              <p className={`max-w-3xl text-sm ${inkClassName("secondary")}`}>
                Two transforms sit between recorded timestamps and dispatch: the{" "}
                <strong>idle-gap warp</strong> that removes dead air, and the{" "}
                <strong>t* snapshot chop</strong> that resumes a trace mid-stream.
              </p>
            </Stack>
            <Divider />

            <Stack gap={10}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Idle-gap warp — collapse dead air, preserve shape
              </h2>
              <p className={`max-w-3xl text-sm ${inkClassName("secondary")}`}>
                The warp works over the union of request <strong>active intervals</strong>{" "}
                [t, t+api_time], not their starts. An idle stretch where nothing runs longer than
                the cap is collapsed to the cap; everything after shifts left by the excess. No
                cut ever falls inside a request, so durations and overlaps stay exact.
              </p>
              <div className="rounded-none border border-stroke-secondary">
                <div className="flex items-center justify-between border-b border-stroke-secondary px-4 py-2">
                  <span className="text-sm font-semibold text-ink-primary">
                    Active intervals A–D · idle 26s -&gt; 5s
                  </span>
                  <span className="rounded-none border border-stroke-secondary px-2 py-0.5 text-xs font-medium text-ink-secondary">
                    cap = 5s
                  </span>
                </div>
                <div style={{ height: 340 }}>
                  <ReactFlow
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    nodes={idleWarpNodes}
                    edges={idleWarpEdges}
                    fitView
                    fitViewOptions={{ padding: 0.2 }}
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
                  </ReactFlow>
                </div>
              </div>
              <Grid columns={2} gap={16}>
                <Callout tone="info" title="Why active-interval, not start-to-start">
                  Capping start-to-start gaps eats into a long request&apos;s own api_time
                  (warping its end past the next start), manufacturing false overlaps.
                  Active-interval capping keeps{" "}
                  <strong>warped_end = warped_start + api_time</strong> always true.
                </Callout>
                <Callout tone="success" title="api_time is never warped">
                  A request&apos;s server-processing duration is added raw to the warped start, so
                  a request that genuinely finished before another still does so on the warped
                  clock.
                </Callout>
              </Grid>
            </Stack>

            <Stack gap={10}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                t* snapshot chop — resume from the live frontier
              </h2>
              <p className={`max-w-3xl text-sm ${inkClassName("secondary")}`}>
                <strong>chop_trie_at_tstar</strong> drops every node whose arrival_offset &lt; t*
                (those turns were warmed, not profiled). Survivors that lost all predecessors are
                re-rooted from START at a <strong>t*-relative</strong> offset; edges between two
                survivors are kept verbatim.
              </p>
              <div className="rounded-none border border-stroke-secondary">
                <div className="flex items-center justify-between border-b border-stroke-secondary px-4 py-2">
                  <span className="text-sm font-semibold text-ink-primary">
                    Before and after the chop at t*
                  </span>
                  <span className="rounded-none border border-stroke-secondary px-2 py-0.5 text-xs font-medium text-ink-secondary">
                    dashed = dropped / re-root
                  </span>
                </div>
                <div style={{ height: 380 }}>
                  <ReactFlow
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    nodes={chopNodes}
                    edges={chopEdges}
                    fitView
                    fitViewOptions={{ padding: 0.2 }}
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
                  </ReactFlow>
                </div>
              </div>
              <Callout tone="warning" title="Prompt path is kept whole">
                Surviving nodes keep their full <strong>prompt_segment_ids</strong> — no
                truncation. The dropped pre-t* turns were dispatched during warmup, so the server
                already holds their KV; the resume prompt must still name the exact full prefix.
                Input requirements on dropped predecessors&apos; channels are removed so
                await_inputs cannot deadlock.
              </Callout>
            </Stack>
          </Stack>
        </div>
      </div>
    </div>
  );
}
