/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — two clocks racing the same recorded session, free-running.
//!
//! The chart lives in `WarpTracks`; this owns the timer and the knobs. The narrated variant at
//! `/spike-warp-narrated` draws the identical picture from the voice instead of a timer.

import { useEffect, useMemo, useRef, useState } from "react";
import { buildTrace, buildWarpMap, rawTimeFor } from "./warpTrace.js";
import { idleGaps } from "../decks/weka-timing-transforms-interactive/logic.js";
import { WarpTracks, warpSummary } from "./WarpTracks.js";
import { ControlBar, Legend, LegendItem, Readout, SourceNote, SpikeHeader, Toggle } from "./ui.js";

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
  const gaps = useMemo(() => idleGaps(trace.reqs, cap), [trace, cap]);
  const warpMap = useMemo(() => buildWarpMap(gaps, cap), [gaps, cap]);
  const { warpSpan, saved } = useMemo(() => warpSummary(trace, cap), [trace, cap]);

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

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <SpikeHeader title="The warp — two clocks, one session">
        <p>
          Both tracks are the same recorded session at the same scale. The top one replays it
          verbatim; the bottom one is what a runtime actually issues, with dead air capped.
        </p>
        <p>
          Watch the two playheads separate — and notice that <strong>every bar is exactly as wide
          on both</strong>. Service time is never compressed; only the gaps between requests are.
          That is what makes the warped replay a faithful one rather than merely a faster one.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <Toggle onClick={() => setRunning((r) => !r)} active>{running ? "Pause" : "Run"}</Toggle>
          <Toggle onClick={() => setWarpedNow(0)}>Replay</Toggle>
          <Toggle onClick={() => { setSeed((s) => s + 1); setWarpedNow(0); }}>seed {seed}</Toggle>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">speed</span>
          {SPEEDS.map((v) => (
            <Toggle key={v} active={speed === v} onClick={() => setSpeed(v)}>{v}×</Toggle>
          ))}
        </div>

        <label className="flex items-center gap-3 text-base">
          <span className="text-ink-tertiary">
            idle cap <strong className="text-ink-secondary tabular-nums">{cap.toFixed(1)}s</strong>
          </span>
          <input type="range" min={1} max={80} value={cap * 10}
            onChange={(e) => setCap(Number(e.target.value) / 10)} />
          <span className="text-[13px] text-ink-quaternary">
            {gaps.filter((g) => g.capped).length} of {gaps.length} gaps exceed it
          </span>
        </label>

        <div className="ml-auto flex items-center gap-6">
          <Readout label="raw" value={`${rawNow.toFixed(1)}s`} />
          <Readout label="warped" value={`${warpedNow.toFixed(1)}s`} color="var(--color-category-green)" />
          <span className="text-lg tabular-nums">
            <span className="text-ink-tertiary">saved</span>{" "}
            <strong style={{ color: "var(--color-category-orange)" }}>
              {saved.toFixed(1)}s ({trace.rawSpan > 0 ? Math.round((saved / trace.rawSpan) * 100) : 0}%)
            </strong>
          </span>
        </div>
      </ControlBar>

      <Legend>
        <LegendItem mark="▬" color="var(--color-category-blue)">main agent turn</LegendItem>
        <LegendItem mark="▬" color="var(--color-category-green)">subagent turn</LegendItem>
        <LegendItem mark="▧">idle gap over the cap — collapsed on the warped track</LegendItem>
        <LegendItem mark="▎" color="var(--color-category-orange)">playhead</LegendItem>
      </Legend>

      <WarpTracks trace={trace} cap={cap} rawNow={rawNow} warpedNow={warpedNow} />

      <SourceNote>
        The bars are turns; the shaded blocks between them are idle gaps that exceed the cap. Only
        those gaps shrink — a turn keeps the duration it was recorded with, so latency measured off
        the warped replay is the latency that was actually observed. The saving is entirely
        recovered dead air.
      </SourceNote>
    </div>
  );
}
