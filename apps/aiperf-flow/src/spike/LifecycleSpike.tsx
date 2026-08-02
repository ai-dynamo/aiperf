/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the request lifecycle as a living rig you watch, not a chart you read.
//!
//! Everything here is driven by one `requestAnimationFrame` loop over `sim.ts`. The concurrency
//! curve at the bottom is drawn from the same events that move the dots above it, in the same
//! frame, so the curve is visibly *generated* rather than replayed.

import { useEffect, useRef, useState } from "react";
import {
  createSim,
  step,
  inFlight,
  queued,
  HISTORY_MS,
  DEFAULT_CONFIG,
  type SimConfig,
  type SimState,
  type Request,
} from "./sim.js";

const W = 1180;
const RIG_TOP = 74;
const LANE_H = 190;
const CURVE_TOP = RIG_TOP + LANE_H + 42;
const CURVE_H = 150;
const H = CURVE_TOP + CURVE_H + 54;

const GATE_X = 250;
const CONNECT_X = 430;
const PREFILL_X = 620;
const DECODE_X = 830;
const DONE_X = 1105;
const TRACK_Y = RIG_TOP + 74;

const STATION_COLOR = {
  queued: "var(--color-category-orange)",
  connect: "var(--color-category-purple)",
  prefill: "var(--color-category-yellow)",
  decode: "var(--color-category-blue)",
  done: "var(--color-category-green)",
} as const;

type Point = { x: number; y: number };

/**
 * Where a request is heading right now.
 *
 * Travel is continuous in *time*, never in tokens: pinning x to `emitted / tokens` made the dot
 * teleport once per token — a 4-15px jump every two to four frames, which is what read as jitter.
 * The token rhythm still shows, as pulse rings, which is the better split anyway. The dot glides;
 * the rings flash.
 */
function targetOf(r: Request, now: number, waitIndex: number): Point {
  const frac = (span: number) => Math.min(1, (now - r.enteredAt) / Math.max(span, 1));
  if (r.stage === "queued") {
    return {
      x: GATE_X - 30 - (waitIndex % 14) * 15,
      y: TRACK_Y - 40 + Math.floor(waitIndex / 14) * 16,
    };
  }
  if (r.stage === "connect") {
    return { x: CONNECT_X + (PREFILL_X - CONNECT_X) * frac(r.connectMs) * 0.55, y: TRACK_Y };
  }
  if (r.stage === "prefill") {
    return { x: PREFILL_X + (DECODE_X - PREFILL_X) * frac(r.ttftMs) * 0.7, y: TRACK_Y };
  }
  if (r.stage === "decode") {
    return { x: DECODE_X + (DONE_X - DECODE_X) * frac(r.tokens * r.itlMs), y: TRACK_Y };
  }
  return { x: DONE_X, y: TRACK_Y };
}

/**
 * Ease rendered positions toward their targets.
 *
 * Some targets move discontinuously no matter how the math is written — admitting the front of the
 * queue shifts every dot behind it up one slot. Easing absorbs that as a glide. Frame-rate
 * independent, so a dropped frame does not produce a lurch.
 */
function ease(from: Point, to: Point, dtMs: number): Point {
  const k = 1 - Math.exp(-dtMs / 70);
  return { x: from.x + (to.x - from.x) * k, y: from.y + (to.y - from.y) * k };
}

/** Horizontal clearance a label needs before it collides with its neighbour. */
const LABEL_MIN_DX = 40;

/**
 * Assign each label a stacking row so overlapping dots do not smear their text together.
 *
 * Requests bunch up in DECODE by design — that bunching *is* the contention being shown — so the
 * dots are left overlapping and only the text is dodged. Greedy first-fit over x: a label takes
 * the topmost row whose last occupant is far enough to the left.
 */
function dodgeRows(items: readonly { id: number; x: number }[]): Map<number, number> {
  const rows: number[] = [];
  const assigned = new Map<number, number>();
  for (const item of [...items].sort((a, b) => a.x - b.x)) {
    let row = 0;
    while (rows[row] !== undefined && item.x - rows[row]! < LABEL_MIN_DX) row += 1;
    rows[row] = item.x;
    assigned.set(item.id, row);
  }
  return assigned;
}

function Station({ x, label, hint }: { x: number; label: string; hint: string }) {
  return (
    <g>
      <line x1={x} y1={RIG_TOP + 16} x2={x} y2={RIG_TOP + LANE_H - 34}
        stroke="var(--color-stroke-tertiary)" strokeWidth={1} strokeDasharray="3 5" />
      <text x={x} y={RIG_TOP + 6} textAnchor="middle" fontSize={13} fontWeight={700}
        fill="var(--color-ink-secondary)" letterSpacing={1.2}>{label}</text>
      <text x={x} y={RIG_TOP + LANE_H - 18} textAnchor="middle" fontSize={12}
        fill="var(--color-ink-quaternary)">{hint}</text>
    </g>
  );
}

export function LifecycleSpike(): React.JSX.Element {
  const [config, setConfig] = useState<SimConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(true);
  // Slow enough to follow one request end to end, tight enough that the gate still saturates.
  const [timeScale, setTimeScale] = useState(0.25);
  const [, forceRender] = useState(0);
  const timeScaleRef = useRef(timeScale);
  timeScaleRef.current = timeScale;
  /** Sim-time owed to a single-step request, drained by the next frame. */
  const stepDebtRef = useRef(0);

  const simRef = useRef<SimState>(createSim(0));
  const configRef = useRef(config);
  configRef.current = config;
  const runningRef = useRef(running);
  runningRef.current = running;
  /** Rendered positions, eased toward each request's target. Keyed by request id. */
  const posRef = useRef<Map<number, Point>>(new Map());

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const realDt = Math.min(64, t - last);
      last = t;

      // Time dilation applies to the world, not to the rendering. Easing below still uses the
      // real frame delta, so the picture stays responsive however slowly the world is running.
      const stepped = stepDebtRef.current;
      stepDebtRef.current = 0;
      const simDt = runningRef.current ? realDt * timeScaleRef.current : stepped;

      if (simDt > 0) {
        const sim = step(simRef.current, simDt, configRef.current);
        simRef.current = sim;

        const positions = posRef.current;
        let waitIndex = 0;
        const alive = new Set<number>();
        for (const r of sim.requests) {
          alive.add(r.id);
          const target = targetOf(r, sim.now, r.stage === "queued" ? waitIndex++ : 0);
          const current = positions.get(r.id);
          // A request appears where it belongs rather than flying in from the last one's slot.
          positions.set(r.id, current === undefined ? target : ease(current, target, realDt));
        }
        for (const id of positions.keys()) if (!alive.has(id)) positions.delete(id);

        forceRender((n) => n + 1);
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const sim = simRef.current;
  const now = sim.now;
  const waiting = sim.requests.filter((r) => r.stage === "queued");
  const live = inFlight(sim.requests);
  const nQueued = queued(sim.requests);

  // Curve scale: headroom above the larger of the limit and the deepest queue seen recently.
  const peak = Math.max(config.concurrency, ...sim.history.map((h) => h.inFlight + h.queued), 1);
  const cx = (t: number) => 60 + ((t - (now - HISTORY_MS)) / HISTORY_MS) * (W - 120);
  const cy = (v: number) => CURVE_TOP + CURVE_H - (v / peak) * CURVE_H;
  const line = (key: "inFlight" | "queued") =>
    sim.history.map((h, i) => `${i === 0 ? "M" : "L"} ${cx(h.t).toFixed(1)} ${cy(h[key]).toFixed(1)}`).join(" ");

  // The heartbeat: a system that is merely idle still breathes; a stopped one flatlines.
  const beat = running ? 1 + 0.4 * Math.sin(now / 320) : 0;

  const onTrack = sim.requests.filter((r) => r.stage !== "queued");
  const labelRow = dodgeRows(
    onTrack.flatMap((r) => {
      const p = posRef.current.get(r.id);
      return p === undefined ? [] : [{ id: r.id, x: p.x }];
    }),
  );

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-sm font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-3xl font-extrabold">Request lifecycle — live</h1>
      </div>
      <p className="mb-4 max-w-5xl text-base leading-relaxed text-ink-secondary">
        Nothing here is a recording. Requests are born, contend for the admission gate, stream
        tokens, and die. Drag <strong>concurrency</strong> down and watch the queue grow; drag{" "}
        <strong>rate</strong> up and watch it grow faster. The curve at the bottom is drawn from the
        same events moving the dots above it, in the same frame.
      </p>

      {/* Two deliberate rows: transport and readouts up top, the three knobs beneath. Left to
          wrap on its own, the readouts ended up orphaned on a second line beside a slider. */}
      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setRunning((r) => !r)}
            className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold"
          >
            {running ? "Pause" : "Run"}
          </button>
          <button
            type="button"
            onClick={() => {
              setRunning(false);
              // 80ms of world per click: about one token at default ITL, so a click is a beat.
              stepDebtRef.current += 80;
            }}
            className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary"
          >
            Step
          </button>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-sm text-ink-tertiary">speed</span>
          {([1, 0.5, 0.25, 0.1, 0.04] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setTimeScale(s)}
              className={`rounded border px-2.5 py-1 text-xs font-semibold tabular-nums ${
                timeScale === s
                  ? "border-transparent bg-accent-primary text-black"
                  : "border-white/15 bg-surface-panel text-ink-secondary"
              }`}
            >
              {s === 1 ? "1×" : `${s}×`}
            </button>
          ))}
        </div>

          <div className="ml-auto flex items-center gap-5 text-sm tabular-nums">
            <span className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: "var(--color-category-green)", transform: `scale(${beat})` }} />
              <span className="text-ink-tertiary">{running ? "running" : "stopped"}</span>
            </span>
            <span><span className="text-ink-tertiary">in flight</span> <strong>{live}</strong></span>
            <span><span className="text-ink-tertiary">queued</span>{" "}
              <strong style={{ color: nQueued > 0 ? "var(--color-category-orange)" : undefined }}>{nQueued}</strong>
            </span>
            <span><span className="text-ink-tertiary">done</span> <strong>{sim.completed}</strong></span>
            <span><span className="text-ink-tertiary">tokens</span> <strong>{sim.tokensOut}</strong></span>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-8 gap-y-2 border-t border-white/10 pt-3">
          <label className="flex items-center gap-3 text-sm">
            <span className="w-24 text-ink-tertiary">rate <strong className="text-ink-secondary">{config.rate}</strong>/s</span>
            <input type="range" min={1} max={40} value={config.rate}
              onChange={(e) => setConfig((c) => ({ ...c, rate: Number(e.target.value) }))} />
          </label>

          <label className="flex items-center gap-3 text-sm">
            <span className="w-36 text-ink-tertiary">concurrency <strong className="text-ink-secondary">{config.concurrency}</strong></span>
            <input type="range" min={1} max={40} value={config.concurrency}
              onChange={(e) => setConfig((c) => ({ ...c, concurrency: Number(e.target.value) }))} />
          </label>

          <label className="flex items-center gap-3 text-sm">
            <span className="w-32 text-ink-tertiary">service <strong className="text-ink-secondary">×{config.serviceScale.toFixed(1)}</strong></span>
            <input type="range" min={2} max={30} value={config.serviceScale * 10}
              onChange={(e) => setConfig((c) => ({ ...c, serviceScale: Number(e.target.value) / 10 }))} />
          </label>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        <Station x={GATE_X} label="ADMIT" hint="concurrency gate" />
        <Station x={CONNECT_X} label="CONNECT" hint="tcp + tls" />
        <Station x={PREFILL_X} label="PREFILL" hint="waiting out TTFT" />
        <Station x={DECODE_X} label="DECODE" hint="one pulse per token" />
        <Station x={DONE_X} label="DONE" hint="terminal" />

        <line x1={40} y1={TRACK_Y} x2={W - 30} y2={TRACK_Y}
          stroke="var(--color-stroke-tertiary)" strokeWidth={1} />

        {/* The gate itself. It reddens as it saturates — pressure you can see before you read it. */}
        <rect x={GATE_X - 7} y={TRACK_Y - 46} width={14} height={92} rx={3}
          fill={live >= config.concurrency ? "var(--color-category-orange)" : "var(--color-category-gray)"}
          opacity={live >= config.concurrency ? 0.9 : 0.35} />

        {/* Waiting requests stack backwards from the gate. This is backpressure, drawn. */}
        {waiting.map((r, i) => {
          const p = posRef.current.get(r.id);
          if (p === undefined) return null;
          return (
            <circle key={r.id} cx={p.x} cy={p.y} r={5}
              fill="var(--color-category-orange)" opacity={0.55 + 0.45 * Math.exp(-i / 8)} />
          );
        })}

        {onTrack.map((r) => {
          const p = posRef.current.get(r.id);
          if (p === undefined) return null;
          const x = p.x;
          const row = labelRow.get(r.id) ?? 0;
          const fading = r.stage === "done" ? Math.max(0, 1 - (now - r.doneAt) / 900) : 1;
          const sinceToken = now - r.lastTokenAt;
          const pulse = r.stage === "decode" && sinceToken < 130 ? 1 - sinceToken / 130 : 0;
          return (
            <g key={r.id} opacity={fading}>
              {/* Token pulse: one ring per token. A stalled stream visibly stops flashing. */}
              {pulse > 0 && (
                <circle cx={x} cy={TRACK_Y} r={7 + pulse * 13} fill="none"
                  stroke={STATION_COLOR.decode} strokeWidth={1.5} opacity={pulse * 0.8} />
              )}
              {r.stage === "decode" && (
                <rect x={DECODE_X} y={TRACK_Y + 15} width={Math.max(0, x - DECODE_X)} height={3}
                  fill={STATION_COLOR.decode} opacity={0.5} />
              )}
              <circle data-req={r.id} data-stage={r.stage} cx={x} cy={TRACK_Y}
                r={r.stage === "done" ? 7 : 6} fill={STATION_COLOR[r.stage]} />
              {/* Dodged upward, and tethered so a label pushed clear still reads as this dot's. */}
              {row > 0 && (
                <line x1={x} y1={TRACK_Y - 11} x2={x} y2={TRACK_Y - 11 - row * 11}
                  stroke="var(--color-ink-quaternary)" strokeWidth={0.75} opacity={0.5} />
              )}
              <text x={x} y={TRACK_Y - 13 - row * 11} textAnchor="middle" fontSize={13}
                fill="var(--color-ink-tertiary)">{r.id}</text>
              {r.stage === "decode" && (
                <text x={x} y={TRACK_Y + 30 + row * 11} textAnchor="middle" fontSize={13}
                  fill="var(--color-ink-quaternary)">{r.emitted}/{r.tokens}</text>
              )}
            </g>
          );
        })}

        <text x={40} y={CURVE_TOP - 14} fontSize={13} fontWeight={700}
          fill="var(--color-ink-secondary)">CONCURRENCY, LIVE</text>
        <text x={210} y={CURVE_TOP - 14} fontSize={12} fill="var(--color-ink-quaternary)">
          last {HISTORY_MS / 1000}s · blue = in flight · orange = queued
        </text>

        <line x1={60} y1={cy(config.concurrency)} x2={W - 60} y2={cy(config.concurrency)}
          stroke="var(--color-category-red)" strokeDasharray="4 4" strokeWidth={1} opacity={0.7} />
        <text x={W - 56} y={cy(config.concurrency) - 4} fontSize={13}
          fill="var(--color-category-red)">limit {config.concurrency}</text>

        <line x1={60} y1={CURVE_TOP + CURVE_H} x2={W - 60} y2={CURVE_TOP + CURVE_H}
          stroke="var(--color-stroke-secondary)" strokeWidth={1} />
        {sim.history.length > 1 && (
          <>
            <path d={line("inFlight")} fill="none" stroke="var(--color-category-blue)" strokeWidth={2} />
            <path d={line("queued")} fill="none" stroke="var(--color-category-orange)" strokeWidth={1.5} />
          </>
        )}
      </svg>
    </div>
  );
}
