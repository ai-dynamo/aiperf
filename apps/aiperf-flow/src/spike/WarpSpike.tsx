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
              <span className="text-ink-quaternary"> / {trace.rawSpan.toFixed(1)}</span></span>
            <span><span className="text-ink-tertiary">warped</span>{" "}
              <strong style={{ color: "var(--color-category-green)" }}>{warpedNow.toFixed(1)}s</strong>
              <span className="text-ink-quaternary"> / {warpSpan.toFixed(1)}</span></span>
            <span><span className="text-ink-tertiary">saved</span>{" "}
              <strong style={{ color: "var(--color-category-orange)" }}>
                {saved.toFixed(1)}s ({trace.rawSpan > 0 ? Math.round((saved / trace.rawSpan) * 100) : 0}%)
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

      <WarpTracks trace={trace} cap={cap} rawNow={rawNow} warpedNow={warpedNow} />
    </div>
  );
}
