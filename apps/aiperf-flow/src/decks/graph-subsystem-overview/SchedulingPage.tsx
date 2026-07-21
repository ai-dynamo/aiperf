/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scheduling page: how timing edges are reconstructed from the recorded clock and how a run
//! resumes mid-trace. Ports `TimelineVisual`, `IntervalFrontierVisual`, `IdleGapWarpVisual`, and
//! `TStarChopVisual` from `graph-subsystem-overview.canvas.tsx`. The hand-drawn SVG gantt/timeline
//! rows become data-driven percentage-width bars (a chart, not diagram boxes).

import { useState } from "react";
import clsx from "clsx";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Toggle } from "../../prose/Toggle.js";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  categoryBgClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

interface TReq {
  id: string;
  start: number;
  end: number;
}
const TIMELINE: TReq[] = [
  { id: "r1", start: 0, end: 3 },
  { id: "r2", start: 1, end: 4 },
  { id: "r3", start: 4, end: 7 },
  { id: "r4", start: 5, end: 9 },
];

function TimelineVisual(): React.JSX.Element {
  const [sel, setSel] = useState("r4");
  const selReq = TIMELINE.find((r) => r.id === sel)!;
  const preds = TIMELINE.filter((r) => r.id !== sel && r.end <= selReq.start);
  const maxT = 9;

  return (
    <Stack gap={12}>
      <Row gap={6} wrap align="center">
        <span className={clsx("text-xs", inkClassName("tertiary"))}>Inspect predecessors of</span>
        {TIMELINE.map((r) => (
          <button
            key={r.id}
            type="button"
            aria-pressed={r.id === sel}
            onClick={() => setSel(r.id)}
            className={clsx("rounded-none border px-2.5 py-0.5 text-xs font-medium", strokeClassName(r.id === sel ? "primary" : "secondary"), r.id === sel ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}
          >
            {r.id}
          </button>
        ))}
      </Row>
      <div className={clsx("rounded-none border px-3 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
        <Stack gap={6}>
          {TIMELINE.map((r) => {
            const isSel = r.id === sel;
            const isPred = preds.some((p) => p.id === r.id);
            const color: CategoryRole = isSel ? "blue" : isPred ? "green" : "gray";
            return (
              <Row key={r.id} gap={8} align="center">
                <div className={clsx("w-8 shrink-0 text-xs font-semibold", isSel ? inkClassName("primary") : inkClassName("tertiary"))}>{r.id}</div>
                <div className="relative h-6 flex-1">
                  <div
                    className={clsx("absolute top-0 flex h-6 items-center rounded-none border px-2 text-[10px] font-medium", strokeClassName("secondary"), categoryBgTintClassName(color), inkClassName("primary"))}
                    style={{ left: `${(r.start / maxT) * 100}%`, width: `${((r.end - r.start) / maxT) * 100}%` }}
                  >
                    {r.id} [{r.start},{r.end}]
                  </div>
                </div>
              </Row>
            );
          })}
          <Row gap={0} align="center">
            <div className="w-8 shrink-0" />
            <div className="relative h-4 flex-1">
              {[0, 3, 6, 9].map((t) => (
                <span key={t} className={clsx("absolute text-[10px]", inkClassName("quaternary"))} style={{ left: `${(t / maxT) * 100}%` }}>t={t}</span>
              ))}
            </div>
          </Row>
        </Stack>
      </div>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        Edge rule: <Code inline>A → B</Code> iff A finished before B started (<Code inline>end(A) ≤ start(B)</Code>).{" "}
        {sel}&apos;s finished-before predecessors are{" "}
        <strong>{preds.length ? preds.map((p) => p.id).join(", ") : "none — it roots at START"}</strong>. Anything still
        in flight when {sel} starts stays concurrent, so genuine racers run in parallel.
      </p>
    </Stack>
  );
}

const FRONTIER_ROWS = [
  { id: "n1", s: 0, e: 2, role: "dropped", color: "gray" as CategoryRole },
  { id: "n2", s: 1, e: 3, role: "frontier", color: "green" as CategoryRole },
  { id: "n3", s: 2, e: 4, role: "binding", color: "blue" as CategoryRole },
  { id: "n4", s: 5, e: 7, role: "target", color: "gray" as CategoryRole },
];

function IntervalFrontier(): React.JSX.Element {
  const maxT = 8;
  return (
    <Stack gap={12}>
      <div className={clsx("rounded-none border px-3 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
        <Stack gap={6}>
          {FRONTIER_ROWS.map((r) => (
            <Row key={r.id} gap={8} align="center">
              <div className={clsx("w-8 shrink-0 text-xs font-semibold", inkClassName("tertiary"))}>{r.id}</div>
              <div className="relative h-6 flex-1">
                <div
                  className={clsx("absolute top-0 flex h-6 items-center rounded-none border px-2 text-[10px] font-medium", strokeClassName("secondary"), categoryBgTintClassName(r.color), inkClassName("primary"), r.role === "dropped" && "opacity-60")}
                  style={{ left: `${(r.s / maxT) * 100}%`, width: `${((r.e - r.s) / maxT) * 100}%` }}
                >
                  {r.role}
                </div>
                <div className={clsx("absolute top-0 h-6 border-l-2", strokeClassName("primary"))} style={{ left: `${(5 / maxT) * 100}%` }} />
              </div>
            </Row>
          ))}
        </Stack>
        <p className={clsx("mt-1 text-[10px]", inkClassName("quaternary"))}>vertical rule = n4 starts (t=5)</p>
      </div>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        All three of <Code inline>n1, n2, n3</Code> finished before <Code inline>n4</Code> started, but{" "}
        <Code inline>n1</Code> is <strong>transitively covered</strong> (n1 → n3 → n4) so it is dropped by the frontier
        filter. The surviving frontier is <Code inline>{"{n2, n3}"}</Code>; the latest-ending predecessor{" "}
        <Code inline>n3</Code> is the binding cause and carries the warped <Code inline>end→start</Code> delay, while{" "}
        <Code inline>n2</Code> becomes a zero-delay AND-join wait.
      </p>
    </Stack>
  );
}

function IdleGapWarp(): React.JSX.Element {
  const [warp, setWarp] = useState(true);
  const raw = [
    { id: "r1", s: 0, e: 2 },
    { id: "r2", s: 2, e: 4 },
    { id: "r3", s: 22, e: 24 },
    { id: "r4", s: 24, e: 26 },
  ];
  const warped = [
    { id: "r1", s: 0, e: 2 },
    { id: "r2", s: 2, e: 4 },
    { id: "r3", s: 6, e: 8 },
    { id: "r4", s: 8, e: 10 },
  ];
  const data = warp ? warped : raw;
  const maxT = warp ? 10 : 26;
  return (
    <Stack gap={10}>
      <Row align="center" gap={10}>
        <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>{warp ? "Idle gaps capped" : "Raw recorded gaps"}</span>
        <div className="flex-1" />
        <span className={clsx("text-xs", inkClassName("tertiary"))}>Cap idle gaps</span>
        <Toggle checked={warp} onChange={setWarp} />
      </Row>
      <div className={clsx("relative h-10 w-full rounded-none border", strokeClassName("secondary"), surfaceClassName("elevated"))}>
        {data.map((r, i) => (
          <div
            key={r.id}
            className={clsx("absolute top-2 flex h-6 items-center rounded-none px-1.5 text-[10px] font-semibold", inkClassName("primary"), i >= 2 ? categoryBgTintClassName("orange") : categoryBgTintClassName("blue"))}
            style={{ left: `${(r.s / maxT) * 100}%`, width: `${((r.e - r.s) / maxT) * 100}%` }}
          >
            {r.id}
          </div>
        ))}
      </div>
      <Row justify="space-between">
        <span className={clsx("text-[10px]", inkClassName("quaternary"))}>t=0</span>
        <span className={clsx("text-[10px]", inkClassName("quaternary"))}>t={maxT}</span>
      </Row>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        The warp collapses only true inactive stretches longer than the cap; it never cuts inside a request&apos;s
        api_time or across overlapping subagent activity — so durations and overlap stay intact while multi-hour dead
        air can&apos;t park warmup forever.
      </p>
    </Stack>
  );
}

const TSTAR_NODES = [
  { id: "1", off: 1 },
  { id: "2", off: 2 },
  { id: "3", off: 4 },
  { id: "4", off: 6 },
  { id: "5", off: 7 },
  { id: "6", off: 9 },
];

function TStarChop(): React.JSX.Element {
  const [tstar, setTstar] = useState("4");
  const t = parseFloat(tstar);
  const maxT = 10;
  return (
    <Stack gap={10}>
      <Row gap={6} align="center" wrap>
        <span className={clsx("text-xs", inkClassName("tertiary"))}>t* =</span>
        {["2", "4", "6", "8"].map((v) => (
          <button
            key={v}
            type="button"
            aria-pressed={tstar === v}
            onClick={() => setTstar(v)}
            className={clsx("rounded-none border px-2.5 py-0.5 text-xs font-medium", strokeClassName(tstar === v ? "primary" : "secondary"), tstar === v ? clsx(categoryBgTintClassName("blue"), inkClassName("primary")) : clsx(surfaceClassName("elevated"), inkClassName("secondary")))}
          >
            {v}
          </button>
        ))}
      </Row>
      <div className={clsx("relative h-12 w-full rounded-none border", strokeClassName("secondary"), surfaceClassName("elevated"))}>
        <div className={clsx("absolute top-0 h-full", categoryBgTintClassName("gray"))} style={{ left: 0, width: `${(t / maxT) * 100}%` }} />
        <div className={clsx("absolute top-0 h-full border-l-2", strokeClassName("primary"))} style={{ left: `${(t / maxT) * 100}%` }} />
        {TSTAR_NODES.map((n) => {
          const warm = n.off < t;
          return (
            <div
              key={n.id}
              className={clsx("absolute top-3 flex h-6 w-6 -translate-x-1/2 items-center justify-center rounded-none border text-[10px] font-bold", strokeClassName("secondary"), warm ? clsx(categoryBgClassName("green"), "text-white") : clsx(categoryBgClassName("blue"), "text-white"))}
              style={{ left: `${(n.off / maxT) * 100}%` }}
            >
              {n.id}
            </div>
          );
        })}
      </div>
      <p className={clsx("text-sm", inkClassName("secondary"))}>
        Nodes before <Code inline>t*</Code> (green) are warmup — their KV is assumed resident, so survivors keep their
        full <Code inline>prompt_segment_ids</Code>. Nodes at/after <Code inline>t*</Code> (blue) are profiled; any whose
        predecessors were chopped are re-rooted from START at a t*-relative offset.
      </p>
    </Stack>
  );
}

export function SchedulingPage(): React.JSX.Element {
  return (
    <Stack gap={20}>
      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Walkthrough: how timing edges are derived</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          The build plane reconstructs dependencies from the recorded clock — no synthetic barriers. Pick a request to
          see its finished-before predecessors.
        </p>
        <TimelineVisual />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Timing edges: frontier &amp; transitive reduction</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Not every finished-before predecessor becomes an edge. The interval-order rule keeps only the maximal
          frontier and pins the binding delay to the latest-ending predecessor.
        </p>
        <IntervalFrontier />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>Idle-gap warp</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Recorded traces contain long idle stretches. The warp caps dead air without distorting real request durations
          or overlap.
        </p>
        <IdleGapWarp />
      </Stack>

      <Stack gap={10}>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>t* snapshot chop</h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          Replay can start mid-trace. Slide the split point to see which turns become warmup vs profiled work.
        </p>
        <TStarChop />
      </Stack>

      <Callout tone="info" title="Reconstructed, not invented">
        Edges come from a finished-before frontier over the recorded clock, with idle-gap warping to cap dead air.
        Concurrency is preserved, not invented — genuine racers in the original workload stay concurrent at replay.
      </Callout>
    </Stack>
  );
}
