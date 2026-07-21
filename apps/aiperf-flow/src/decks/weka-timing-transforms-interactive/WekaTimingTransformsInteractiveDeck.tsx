/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-view port of `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`: recorded
//! weka timestamps become dispatch timing through the pipeline `_weka_trie_build.py` runs —
//! idle-gap warp, interval-order edges, and the runtime t* snapshot chop. All state (scenario,
//! idle cap, warp toggle, t* sliders) is plain `useState`, matching the source canvas's
//! `useCanvasState` slots one-for-one.

import { useState } from "react";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Stat } from "../../prose/Stat.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { Select } from "../../prose/Select.js";
import { Toggle } from "../../prose/Toggle.js";
import { Code } from "../../prose/Code.js";
import { Button } from "../../prose/Button.js";
import { Legend } from "../../prose/Legend.js";
import { LineChart } from "../../prose/Chart.js";
import { inkClassName } from "../../theme/tokens.js";
import { StageExplorer } from "./StageExplorer.js";
import { Timeline } from "./Timeline.js";
import { TStarChop } from "./TStarChop.js";
import { CombinedTimeline } from "./CombinedTimeline.js";
import {
  LANE_KEYS,
  MINI_TRACES,
  SCENARIOS,
  buildCuts,
  computeEdges,
  derive,
  fmt,
  idleGaps,
  laneColorIndex,
  lanesOf,
} from "./logic.js";

const SOURCE_FILES: Array<{ label: string; path: string }> = [
  {
    label: "_weka_trie_build.py",
    path: "src/aiperf/dataset/loader/graph/adapters/_weka_trie_build.py",
  },
  { label: "graph_ir_replay.py", path: "src/aiperf/timing/strategies/graph_ir_replay.py" },
  { label: "step_emit_weka.py", path: "src/aiperf/dataset/loader/graph/step_emit_weka.py" },
];

const SCENARIO_OPTIONS = Object.entries(SCENARIOS).map(([value, s]) => ({ value, label: s.label }));

const EDGE_COLUMNS: TableColumn[] = [
  { key: "node", label: "Node" },
  { key: "lane", label: "Lane" },
  { key: "firesAfter", label: "Fires after" },
  { key: "delay", label: "delay_after_pred", align: "end" },
  { key: "and", label: "AND-fan-in waits" },
  { key: "arrival", label: "arrival_offset", align: "end" },
];

function ChopLegend(): React.JSX.Element {
  return (
    <Legend
      entries={[
        { color: "blue", label: "survivor (profiled)" },
        { color: "gray", label: "dropped (warmed)" },
        { color: "orange", label: "warmup turn nearest t* (KV retained)" },
      ]}
    />
  );
}

function LaneLegend({ lanes }: { lanes: string[] }): React.JSX.Element {
  return (
    <Legend entries={lanes.map((agent) => ({ color: LANE_KEYS[laneColorIndex(agent, lanes)]!, label: agent }))} />
  );
}

/** One mini-trace card: its own independent t* slider driving a `beforeOnly` chop diagram. */
function MiniTraceChop({
  traceKey,
  label,
  tStar,
  onChangeTStar,
}: {
  traceKey: string;
  label: string;
  tStar: number;
  onChangeTStar: (v: number) => void;
}): React.JSX.Element {
  const reqs = MINI_TRACES.find((t) => t.key === traceKey)!.reqs;
  const lanes = lanesOf(reqs);
  const nodes = derive(reqs, 60);
  const edges = computeEdges(nodes);
  const warpMax = Math.max(...nodes.map((n) => n.warpEnd));
  const clamped = Math.min(tStar, Math.ceil(warpMax));
  const warmed = nodes.filter((n) => n.warpStart < clamped).length;
  const profiled = nodes.length - warmed;

  return (
    <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
      <Row align="center" justify="space-between">
        <h4 className={`text-sm font-semibold ${inkClassName("primary")}`}>{label}</h4>
        <span className={`rounded-md border border-stroke-secondary px-2 py-0.5 text-xs shadow-sm ${inkClassName("secondary")}`}>
          t* = {fmt(clamped)}s
        </span>
      </Row>
      <Stack gap={10} className="mt-3">
        <Row gap={12} align="center" wrap>
          <span className={`min-w-[24px] text-sm font-semibold ${inkClassName("primary")}`}>t*</span>
          <input
            type="range"
            min={0}
            max={Math.max(1, Math.ceil(warpMax))}
            step={1}
            value={clamped}
            onChange={(e) => onChangeTStar(Number(e.target.value))}
            style={{ width: 220, accentColor: "var(--color-accent-primary)" }}
          />
          <span className={`text-sm ${inkClassName("secondary")}`}>{fmt(clamped)}s</span>
          <span className={`ml-auto text-sm ${inkClassName("tertiary")}`}>
            {warmed} warmed · {profiled} profiled
          </span>
        </Row>
        <TStarChop nodes={nodes} edges={edges} lanes={lanes} tStar={clamped} beforeOnly />
      </Stack>
    </div>
  );
}

/**
 * Weka timing transforms — interactive port of the Cursor Canvas of the same name. Six sections:
 * the nine-stage pipeline explorer, the idle-gap warp lab, interval-order edges, the t* snapshot
 * chop, three independently-driven mini traces, and their combined aligned timeline.
 */
export function WekaTimingTransformsInteractiveDeck(): React.JSX.Element {
  const [scenarioId, setScenarioId] = useState<string>("agent");
  const [cap, setCap] = useState<number>(60);
  const [warpOn, setWarpOn] = useState<boolean>(true);
  const [tStarRaw, setTStar] = useState<number>(0);
  const [miniTStars, setMiniTStars] = useState<Record<string, number>>({
    linear: 0,
    "one-sub": 0,
    "two-subs": 0,
  });

  const scenario = SCENARIOS[scenarioId] ?? SCENARIOS.agent!;
  const reqs = scenario.reqs;
  const lanes = lanesOf(reqs);
  const effectiveCap = warpOn ? cap : null;
  const nodes = derive(reqs, effectiveCap);
  const gaps = idleGaps(reqs, cap);
  const edges = computeEdges(nodes);

  const rawSpan = Math.max(...nodes.map((n) => n.rawEnd)) - Math.min(...nodes.map((n) => n.rawStart));
  const warpSpan = Math.max(...nodes.map((n) => n.warpEnd)) - Math.min(...nodes.map((n) => n.warpStart));
  const removed = Math.max(0, rawSpan - warpSpan);
  const cuts = warpOn
    ? buildCuts(
        reqs.map((r) => [r.t, r.t + r.api] as [number, number]),
        cap,
      )
    : [];
  const warpMax = Math.max(...nodes.map((n) => n.warpEnd));
  const tStar = Math.min(tStarRaw, Math.floor(warpMax));

  const edgeRows: TableRow[] = edges.map((e, i) => ({
    node: e.id,
    lane: nodes[i]!.agent,
    firesAfter: e.rootsAtStart ? "START (concurrent)" : e.firesAfter,
    delay: e.rootsAtStart ? "—" : `${fmt(e.delayMs)} ms`,
    and: e.andInputs.length ? e.andInputs.join(", ") : "—",
    arrival: `${fmt(nodes[i]!.warpStart)}s`,
    tone: e.rootsAtStart ? "success" : "neutral",
  }));

  const miniTStarArr: [number, number, number] = [
    miniTStars.linear ?? 0,
    miniTStars["one-sub"] ?? 0,
    miniTStars["two-subs"] ?? 0,
  ];

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Weka Timing Transforms" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-5xl bg-surface-page px-10 py-8">
          <Stack gap={26}>
            <Stack gap={10}>
              <Row align="center" gap={10} wrap>
                <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>Weka timing transforms</h1>
                <span className="rounded-md border border-accent-primary bg-accent-primary px-2 py-0.5 text-xs font-semibold text-white shadow-sm">
                  interactive
                </span>
                <span className="rounded-md border border-stroke-secondary px-2 py-0.5 text-xs font-medium text-ink-secondary shadow-sm">
                  warped clock
                </span>
              </Row>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Recorded weka timestamps become dispatch timing through a chain of transforms. Pick a scenario, drag
                the idle-gap cap and the t* snapshot, and watch the whole pipeline recompute with the exact{" "}
                <Code inline>_ActiveIdleWarp</Code> and interval-order edge logic from the loader.
              </p>
              <Row gap={8} wrap>
                {SOURCE_FILES.map((f) => (
                  <span key={f.path} title={f.path}>
                    <Code inline>{f.label}</Code>
                  </span>
                ))}
              </Row>
            </Stack>

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>The pipeline, stage by stage</h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Nine stages turn a raw trace into a schedulable trie. Five of them touch the clock; the rest only
                shape content and dependencies. Click a stage.
              </p>
              <StageExplorer />
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Idle-gap warp lab</h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Each subagent gets its own <strong>lane</strong>, but the warp runs over the union of{" "}
                <strong>all</strong> lanes&apos; active intervals [t, t+api_time]. A stretch where nothing is
                running in any lane longer than the cap is collapsed to the cap; every later timestamp shifts left
                by the excess. Long requests and overlapping subagents are never cut.
              </p>

              <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
                <Row align="center" justify="space-between">
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Controls</span>
                  <span className={`rounded-md border border-stroke-secondary px-2 py-0.5 text-xs shadow-sm ${inkClassName("secondary")}`}>
                    {warpOn ? `cap = ${cap}s` : "warp off"}
                  </span>
                </Row>
                <Stack gap={14} className="mt-3">
                  <Row gap={12} align="center" wrap>
                    <span className={`min-w-[70px] text-sm font-semibold ${inkClassName("primary")}`}>Scenario</span>
                    <div style={{ minWidth: 260 }}>
                      <Select value={scenarioId} onChange={setScenarioId} options={SCENARIO_OPTIONS} />
                    </div>
                  </Row>
                  <Row gap={12} align="center" wrap>
                    <span className={`min-w-[70px] text-sm font-semibold ${inkClassName("primary")}`}>Warp</span>
                    <Toggle checked={warpOn} onChange={setWarpOn} />
                    <span className={`text-sm ${inkClassName("tertiary")}`}>
                      {warpOn ? "active-interval idle capping on" : "no-cap passthrough (warped_start = raw t)"}
                    </span>
                  </Row>
                  <Row gap={12} align="center" wrap>
                    <span className={`min-w-[70px] text-sm font-semibold ${inkClassName("primary")}`}>Idle cap</span>
                    <input
                      type="range"
                      min={5}
                      max={120}
                      step={5}
                      value={cap}
                      disabled={!warpOn}
                      onChange={(e) => setCap(Number(e.target.value))}
                      style={{ width: 260, accentColor: "var(--color-accent-primary)" }}
                    />
                    <span className={`text-sm ${inkClassName("secondary")}`}>
                      {cap}s{cap === 60 ? " (default)" : ""}
                    </span>
                    <Button
                      variant="ghost"
                      className="ml-auto"
                      onClick={() => {
                        setCap(60);
                        setWarpOn(true);
                        setTStar(0);
                      }}
                    >
                      Reset
                    </Button>
                  </Row>
                </Stack>
              </div>

              <Grid columns={4} gap={12}>
                <Stat value={`${fmt(rawSpan)}s`} label="raw span" />
                <Stat value={`${fmt(warpSpan)}s`} label="warped span" />
                <Stat value={`${fmt(removed)}s`} label="dead air removed" tone="positive" />
                <Stat value={cuts.length} label="idle gaps cut" />
              </Grid>

              <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
                <Row align="center" justify="space-between">
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Raw vs warped timeline</span>
                  <span className={`rounded-md border border-stroke-secondary px-2 py-0.5 text-xs shadow-sm ${inkClassName("secondary")}`}>
                    one lane per subagent
                  </span>
                </Row>
                <Stack gap={10} className="mt-3">
                  <Timeline nodes={nodes} gaps={gaps} lanes={lanes} warpOn={warpOn} />
                  <LaneLegend lanes={lanes} />
                </Stack>
              </div>

              <Grid columns="1fr 1fr" gap={16}>
                <Callout tone="info" title="Why active-interval, not start-to-start">
                  Capping start-to-start gaps eats into a long request&apos;s own api_time, warping its end past
                  the next start and manufacturing false overlaps. Active-interval capping keeps{" "}
                  <strong>warped_end = warped_start + api_time</strong> exact.
                </Callout>
                <Callout tone="info" title="api_time is never warped">
                  A request&apos;s server-processing duration is added raw to its warped start, so
                  finished-before relationships on the raw clock survive onto the warped clock unchanged.
                </Callout>
              </Grid>

              <Stack gap={8}>
                <h3 className={`text-sm font-semibold ${inkClassName("primary")}`}>Warp mapping per node</h3>
                <Grid columns="1fr 1fr" gap={14}>
                  <div>
                    <p className={`mb-1 text-xs font-semibold ${inkClassName("tertiary")}`}>Raw start</p>
                    <LineChart
                      data={nodes.map((n) => ({ label: n.id, value: Number(n.rawStart.toFixed(2)) }))}
                      color="blue"
                    />
                  </div>
                  <div>
                    <p className={`mb-1 text-xs font-semibold ${inkClassName("tertiary")}`}>Warped start</p>
                    <LineChart
                      data={nodes.map((n) => ({ label: n.id, value: Number(n.warpStart.toFixed(2)) }))}
                      color="green"
                    />
                  </div>
                </Grid>
                <p className={`text-xs ${inkClassName("tertiary")}`}>
                  Source: idle-gap warp on the {scenario.label.toLowerCase()} scenario · x = node id, y = arrival
                  (s). The gap between the two lines is the cumulative dead air removed before that node.
                </p>
              </Stack>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Interval-order edges &amp; binding delay
              </h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Firing timing lives on edges, not segments. Each turn fires after its{" "}
                <strong>binding cause</strong> — the latest-ending cause that finished before it started — by the
                warped end-to-start gap. Turns whose causes were all still in flight root at START and fire
                concurrently at their own arrival offset.
              </p>
              <Table columns={EDGE_COLUMNS} rows={edgeRows} />
              <Callout tone="info" title="Delay is not clamped to the cap">
                A binding delay can exceed the cap: that only happens when the trace was genuinely{" "}
                <strong>busy</strong> across the interval. The warp cut only true idle gaps, so a large end-to-start
                gap on the warped clock is real waiting the client did.
              </Callout>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>t* snapshot chop</h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                Set t* to resume a trace mid-stream. Nodes arriving before t* were warmed (dashed) and drop out;
                survivors that lost every predecessor re-root to <Code inline>S*</Code> at a t*-relative offset.
                The orange box in each lane is that lane&apos;s warmup turn nearest t* — the last turn that ran
                before the snapshot, whose KV the survivor still resumes from. The scale below is the{" "}
                <strong>warped</strong> clock, so it moves with the cap above.
              </p>
              <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
                <Row gap={12} align="center" wrap>
                  <span className={`min-w-[70px] text-sm font-semibold ${inkClassName("primary")}`}>t*</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(1, Math.ceil(warpMax))}
                    step={5}
                    value={tStar}
                    onChange={(e) => setTStar(Number(e.target.value))}
                    style={{ width: 300, accentColor: "var(--color-accent-primary)" }}
                  />
                  <span className={`text-sm ${inkClassName("secondary")}`}>{fmt(tStar)}s</span>
                  <span className={`ml-auto text-sm ${inkClassName("tertiary")}`}>
                    {nodes.filter((n) => n.warpStart < tStar).length} warmed ·{" "}
                    {nodes.filter((n) => n.warpStart >= tStar).length} profiled
                  </span>
                </Row>
              </div>
              <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
                <Row align="center" justify="space-between">
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Before and after the chop at t*</span>
                  <span className={`rounded-md border border-stroke-secondary px-2 py-0.5 text-xs shadow-sm ${inkClassName("secondary")}`}>
                    dashed = dropped / re-root
                  </span>
                </Row>
                <Stack gap={10} className="mt-3">
                  <TStarChop nodes={nodes} edges={edges} lanes={lanes} tStar={tStar} />
                  <ChopLegend />
                </Stack>
              </div>
              <Callout tone="warning" title="Prompt path is kept whole">
                Surviving nodes keep their full <strong>prompt_segment_ids</strong> — no truncation. The dropped
                pre-t* turns ran during warmup, so the server already holds their KV; the resume prompt must still
                name the exact full prefix. Input requirements on dropped predecessors&apos; channels are removed
                so await_inputs cannot deadlock.
              </Callout>
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Independent t* across three traces
              </h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                In a real run every conversation carries its own snapshot frontier. Each trace below has its own t*
                slider — move one and the others hold. Orange marks each lane&apos;s warmup turn nearest that
                trace&apos;s t*.
              </p>
              <Stack gap={14}>
                {MINI_TRACES.map((tr) => (
                  <MiniTraceChop
                    key={tr.key}
                    traceKey={tr.key}
                    label={tr.label}
                    tStar={miniTStars[tr.key] ?? 0}
                    onChangeTStar={(v) => setMiniTStars((prev) => ({ ...prev, [tr.key]: v }))}
                  />
                ))}
              </Stack>
              <ChopLegend />
            </Stack>

            <Divider />

            <Stack gap={14}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                Combined timeline — all traces aligned at t*
              </h2>
              <p className={`text-sm ${inkClassName("secondary")}`}>
                The three traces collapsed into <strong>one combined swimlane stack</strong>, each shifted so its
                own t* lands on the <strong>single shared vertical line</strong>. Warmup turns fall to the left of
                the line, profiled turns to the right, so you can compare snapshots across traces regardless of
                their absolute timing. Drag any slider above and that trace&apos;s lanes slide to keep t* on the
                line.
              </p>
              <div className="rounded-lg border border-stroke-secondary bg-surface-elevated p-4 shadow-sm">
                <Row align="center" justify="space-between">
                  <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
                    All warmup + post-t* requests, aligned at t*
                  </span>
                  <span className={`rounded-md border border-stroke-secondary px-2 py-0.5 text-xs shadow-sm ${inkClassName("secondary")}`}>
                    aligned at t* · one line
                  </span>
                </Row>
                <Stack gap={10} className="mt-3">
                  <CombinedTimeline tStars={miniTStarArr} />
                  <ChopLegend />
                </Stack>
              </div>
              <p className={`text-xs ${inkClassName("tertiary")}`}>
                Source: three mini traces from the section above · x = seconds relative to t* (0 = the shared
                line), rows are trace·lane. Left of the line ran during warmup; right of the line is profiled.
              </p>
            </Stack>
          </Stack>
        </div>
      </div>
    </div>
  );
}
