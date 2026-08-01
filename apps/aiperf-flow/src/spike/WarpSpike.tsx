/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — two clocks racing the same recorded session.
//!
//! Both tracks are drawn at the *same* pixels-per-second, so the warped track being physically
//! shorter is the compression, and every bar keeping its exact width is the invariant: service
//! time is never squeezed, only the dead air between requests.

import { useEffect, useMemo, useRef, useState } from "react";
import { buildTrace, buildWarpMap, rawTimeFor } from "./warpTrace.js";
import { derive, idleGaps } from "../decks/weka-timing-transforms-interactive/logic.js";

const W = 1280;
const LEFT = 132;
const TOP = 92;
const LANE_H = 22;
const BAR_H = 13;
const BLOCK_GAP = 58;

const DEPTH_COLOR = [
  "var(--color-category-blue)",
  "var(--color-category-green)",
  "var(--color-category-purple)",
  "var(--color-category-orange)",
] as const;

const SPEEDS = [2, 1, 0.5, 0.25] as const;
/** Seconds of recorded session to freeze. Long enough to contain real dead air. */
const TRACE_MS = 75_000;

export function WarpSpike(): React.JSX.Element {
  const [seed, setSeed] = useState(3);
  const [cap, setCap] = useState(1.2);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [warpedNow, setWarpedNow] = useState(0);

  const trace = useMemo(() => buildTrace(seed, TRACE_MS), [seed]);
  const { reqs, lanes, depthOf, rawSpan } = trace;

  // The real weka transform, not a lookalike: cut table, warped nodes, and classified gaps.
  const nodes = useMemo(() => derive(reqs, cap), [reqs, cap]);
  const gaps = useMemo(() => idleGaps(reqs, cap), [reqs, cap]);
  const warpMap = useMemo(() => buildWarpMap(gaps, cap), [gaps, cap]);
  const warpSpan = nodes.reduce((m, n) => Math.max(m, n.warpEnd), 0);

  const runningRef = useRef(running);
  runningRef.current = running;
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const spanRef = useRef(warpSpan);
  spanRef.current = warpSpan;

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      if (runningRef.current) {
        setWarpedNow((p) => {
          const next = p + (dt / 1000) * speedRef.current;
          return next > spanRef.current + 0.8 ? 0 : next;
        });
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const rawNow = rawTimeFor(warpedNow, warpMap);
  const saved = Math.max(0, rawSpan - warpSpan);

  // One scale for both tracks. A shared pixels-per-second is what makes "the warped track is
  // shorter" mean something and keeps every bar's width identical across the two clocks.
  const px = Math.max(2, (W - LEFT - 30) / Math.max(rawSpan, 1));
  const x = (t: number) => LEFT + t * px;

  const rawTop = TOP;
  const warpTop = rawTop + lanes.length * LANE_H + BLOCK_GAP;
  const H = warpTop + lanes.length * LANE_H + 46;
  const laneY = (top: number, lane: string) => top + Math.max(0, lanes.indexOf(lane)) * LANE_H;
  const colorOf = (lane: string) =>
    DEPTH_COLOR[Math.min(depthOf.get(lane) ?? 0, DEPTH_COLOR.length - 1)]!;

  const block = (top: number, key: "raw" | "warp", head: number) => (
    <>
      {lanes.map((lane) => (
        <line key={`rule-${key}-${lane}`} x1={LEFT} y1={laneY(top, lane) + BAR_H / 2}
          x2={W - 26} y2={laneY(top, lane) + BAR_H / 2}
          stroke="var(--color-stroke-tertiary)" strokeWidth={1} opacity={0.25} />
      ))}
      {nodes.map((n) => {
        const s = key === "raw" ? n.rawStart : n.warpStart;
        const e = key === "raw" ? n.rawEnd : n.warpEnd;
        const c = colorOf(n.agent);
        const passed = head >= e;
        const inside = head >= s && head < e;
        return (
          <rect key={`${key}-${n.id}`} x={x(s)} y={laneY(top, n.agent)}
            width={Math.max(1.5, (e - s) * px)} height={BAR_H} rx={1.5}
            fill={c} fillOpacity={passed ? 0.4 : inside ? 0.7 : 0.09}
            stroke={c} strokeWidth={inside ? 1.4 : 0.7}
            strokeOpacity={head >= s ? 1 : 0.35} />
        );
      })}
      <line x1={x(head)} y1={top - 12} x2={x(head)} y2={top + lanes.length * LANE_H + 2}
        stroke="var(--color-category-red)" strokeWidth={1.6} />
    </>
  );

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-2xl font-extrabold">The warp — two clocks, one session</h1>
      </div>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        Both tracks are the same recorded session at the same scale. The top one replays it
        verbatim; the bottom one is what a runtime actually issues, with dead air capped. Watch the
        two playheads separate — and notice that every bar is exactly as wide on both. Service time
        is never compressed, only the gaps between requests.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => setRunning((r) => !r)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold">
              {running ? "Pause" : "Run"}
            </button>
            <button type="button" onClick={() => setWarpedNow(0)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Replay
            </button>
            <button type="button" onClick={() => { setSeed((s) => s + 1); setWarpedNow(0); }}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary tabular-nums">
              seed {seed}
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-sm text-ink-tertiary">speed</span>
            {SPEEDS.map((s) => (
              <button key={s} type="button" onClick={() => setSpeed(s)}
                className={`rounded border px-2.5 py-1 text-xs font-semibold tabular-nums ${
                  speed === s ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {s}×
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-6 text-sm tabular-nums">
            <span><span className="text-ink-tertiary">raw</span>{" "}
              <strong>{rawNow.toFixed(1)}s</strong>
              <span className="text-ink-quaternary"> / {rawSpan.toFixed(1)}</span></span>
            <span><span className="text-ink-tertiary">warped</span>{" "}
              <strong style={{ color: "var(--color-category-green)" }}>{warpedNow.toFixed(1)}s</strong>
              <span className="text-ink-quaternary"> / {warpSpan.toFixed(1)}</span></span>
            <span><span className="text-ink-tertiary">saved</span>{" "}
              <strong style={{ color: "var(--color-category-orange)" }}>
                {saved.toFixed(1)}s ({rawSpan > 0 ? Math.round((saved / rawSpan) * 100) : 0}%)
              </strong></span>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-8 gap-y-2 border-t border-white/10 pt-3">
          <label className="flex items-center gap-3 text-sm">
            <span className="w-40 text-ink-tertiary">idle cap <strong className="text-ink-secondary">{cap.toFixed(1)}s</strong></span>
            <input type="range" min={1} max={80} value={cap * 10}
              onChange={(e) => setCap(Number(e.target.value) / 10)} />
          </label>
          <span className="text-xs text-ink-quaternary">
            {gaps.filter((g) => g.capped).length} of {gaps.length} idle gaps exceed the cap and get collapsed
          </span>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        <text x={LEFT} y={rawTop - 24} fontSize={11} fontWeight={700}
          fill="var(--color-ink-secondary)" letterSpacing={1.1}>RAW CLOCK — AS RECORDED</text>

        {/* Only the capped gaps are drawn: those are the stretches the warp actually removes. */}
        {gaps.filter((g) => g.capped).map((g, i) => (
          <g key={`gap-${i}`}>
            <rect x={x(g.start)} y={rawTop - 8} width={Math.max(2, (g.end - g.start) * px)}
              height={lanes.length * LANE_H + 12} fill="var(--color-category-orange)" opacity={0.11} />
            <text x={(x(g.start) + x(g.end)) / 2} y={rawTop - 12} textAnchor="middle" fontSize={9}
              fill="var(--color-category-orange)">−{(g.idle - cap).toFixed(1)}s</text>
          </g>
        ))}

        {lanes.map((lane) => (
          <text key={`l-raw-${lane}`} x={LEFT - 10} y={laneY(rawTop, lane) + BAR_H - 2}
            textAnchor="end" fontSize={9.5} fill={colorOf(lane)}>{lane}</text>
        ))}
        {block(rawTop, "raw", rawNow)}

        <text x={LEFT} y={warpTop - 24} fontSize={11} fontWeight={700}
          fill="var(--color-category-green)" letterSpacing={1.1}>
          WARPED CLOCK — WHAT THE RUNTIME ISSUES
        </text>
        {/* The span the warp removed, left visible so the saving has a size. */}
        <rect x={x(warpSpan)} y={warpTop - 8} width={Math.max(0, (rawSpan - warpSpan) * px)}
          height={lanes.length * LANE_H + 12} fill="var(--color-category-green)" opacity={0.07} />
        <text x={x(warpSpan) + 8} y={warpTop + lanes.length * LANE_H + 18} fontSize={10}
          fill="var(--color-category-green)">{saved.toFixed(1)}s never replayed</text>

        {lanes.map((lane) => (
          <text key={`l-warp-${lane}`} x={LEFT - 10} y={laneY(warpTop, lane) + BAR_H - 2}
            textAnchor="end" fontSize={9.5} fill={colorOf(lane)}>{lane}</text>
        ))}
        {block(warpTop, "warp", warpedNow)}
      </svg>
    </div>
  );
}
