/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — an agent session as it happens: lanes appear when spawned, bars grow while streaming.
//!
//! The time axis scrolls, so a bar's right edge is *now*. A running turn has no right edge yet —
//! it is still being drawn. That is the difference from a recorded swimlane, where every bar is
//! already closed before you see it.

import { useEffect, useRef, useState } from "react";
import {
  createAgentSim,
  stepAgents,
  laneOrder,
  idleFraction,
  WINDOW_MS,
  DEFAULT_AGENT_CONFIG,
  type AgentSimConfig,
  type AgentSimState,
} from "./agentSim.js";

const W = 1240;
const GUTTER = 168;
const TOP = 58;
/** Lane pitch shrinks as lanes multiply, so a busy session still fits without scrolling. */
const LANE_H_MAX = 34;
const LANE_H_MIN = 19;
const LANES_BUDGET_PX = 470;
const AXIS_H = 34;

const DEPTH_COLOR = [
  "var(--color-category-blue)",
  "var(--color-category-green)",
  "var(--color-category-purple)",
  "var(--color-category-orange)",
] as const;

function color(depth: number): string {
  return DEPTH_COLOR[Math.min(depth, DEPTH_COLOR.length - 1)]!;
}

/** Vertical pitch that keeps `count` lanes inside the height budget. */
function lanePitch(count: number): number {
  if (count <= 0) return LANE_H_MAX;
  return Math.max(LANE_H_MIN, Math.min(LANE_H_MAX, LANES_BUDGET_PX / count));
}

/** Indent per nesting level, in the label gutter. */
const INDENT = 14;
/** Left edge of the label column. Labels read left-to-right; the tree rail sits to their left. */
const LABEL_X = 16;

const SPEEDS = [1, 0.5, 0.25, 0.1] as const;

export function AgentSwimlaneSpike(): React.JSX.Element {
  const [config, setConfig] = useState<AgentSimConfig>(DEFAULT_AGENT_CONFIG);
  const [running, setRunning] = useState(true);
  const [timeScale, setTimeScale] = useState(0.5);
  const [seed, setSeed] = useState(1);
  const [, forceRender] = useState(0);

  const simRef = useRef<AgentSimState>(createAgentSim(1));
  const configRef = useRef(config);
  configRef.current = config;
  const runningRef = useRef(running);
  runningRef.current = running;
  const scaleRef = useRef(timeScale);
  scaleRef.current = timeScale;
  const stepDebtRef = useRef(0);
  /** Eased lane y, so a lane born mid-list slides its neighbours down instead of snapping. */
  const laneYRef = useRef<Map<number, number>>(new Map());

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const realDt = Math.min(64, t - last);
      last = t;
      const owed = stepDebtRef.current;
      stepDebtRef.current = 0;
      const simDt = runningRef.current ? realDt * scaleRef.current : owed;

      if (simDt > 0) {
        const sim = stepAgents(simRef.current, simDt, configRef.current);
        simRef.current = sim;

        const ys = laneYRef.current;
        const order = laneOrder(sim.agents);
        const alive = new Set(order.map((a) => a.id));
        const k = 1 - Math.exp(-realDt / 90);
        const pitch = lanePitch(order.length);
        order.forEach((a, i) => {
          const target = TOP + i * pitch;
          const current = ys.get(a.id);
          // A new lane fades in at its final position; existing lanes glide to make room.
          ys.set(a.id, current === undefined ? target : current + (target - current) * k);
        });
        for (const id of ys.keys()) if (!alive.has(id)) ys.delete(id);

        forceRender((n) => n + 1);
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const sim = simRef.current;
  const now = sim.now;
  const lanes = laneOrder(sim.agents);
  const pitch = lanePitch(lanes.length);
  const BAR_H = Math.min(17, pitch - 8);
  const H = TOP + Math.max(lanes.length, 5) * pitch + AXIS_H;
  const idle = idleFraction(sim, WINDOW_MS);

  const tMin = now - WINDOW_MS;
  const x = (t: number) => GUTTER + ((t - tMin) / WINDOW_MS) * (W - GUTTER - 24);
  const yOf = (id: number) => laneYRef.current.get(id) ?? TOP;

  const active = sim.agents.filter((a) => a.retiredAt === null).length;
  const streaming = sim.turns.filter((t) => t.endAt === null && t.firstTokenAt !== null).length;

  const ticks: number[] = [];
  const tickStep = 5000;
  for (let t = Math.ceil(tMin / tickStep) * tickStep; t <= now; t += tickStep) ticks.push(t);

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-2xl font-extrabold">Agent session — lanes appear as they spawn</h1>
      </div>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        A lane does not exist until something spawns it. Watch <strong>main</strong> take a turn,
        fan out subagents on completion, then sit idle while they work. Bars with no right edge are
        still streaming — the axis scrolls, so the right edge is <em>now</em>. The dashed stretches
        are dead air, accumulating in front of you rather than summarised afterwards.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => setRunning((r) => !r)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold">
              {running ? "Pause" : "Run"}
            </button>
            <button type="button"
              onClick={() => { setRunning(false); stepDebtRef.current += 120; }}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Step
            </button>
            <button type="button"
              onClick={() => { simRef.current = createAgentSim(seed); laneYRef.current.clear(); }}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Restart
            </button>
            <button type="button"
              onClick={() => { const n = seed + 1; setSeed(n);
                simRef.current = createAgentSim(n); laneYRef.current.clear(); }}
              title="A different session, equally reproducible"
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary tabular-nums">
              seed {seed}
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-sm text-ink-tertiary">speed</span>
            {SPEEDS.map((s) => (
              <button key={s} type="button" onClick={() => setTimeScale(s)}
                className={`rounded border px-2.5 py-1 text-xs font-semibold tabular-nums ${
                  timeScale === s ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {s === 1 ? "1×" : `${s}×`}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-5 text-sm tabular-nums">
            <span><span className="text-ink-tertiary">lanes</span> <strong>{lanes.length}</strong></span>
            <span><span className="text-ink-tertiary">active</span> <strong>{active}</strong></span>
            <span><span className="text-ink-tertiary">streaming</span>{" "}
              <strong style={{ color: streaming > 0 ? "var(--color-category-blue)" : undefined }}>{streaming}</strong>
            </span>
            <span><span className="text-ink-tertiary">spawned</span> <strong>{sim.spawnedTotal}</strong></span>
            {/* The compressible dead air, measured as a union of active intervals. */}
            <span title="fraction of the window with no lane working">
              <span className="text-ink-tertiary">idle</span>{" "}
              <strong style={{ color: idle > 0.35 ? "var(--color-category-orange)" : undefined }}>
                {Math.round(idle * 100)}%
              </strong>
            </span>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-8 gap-y-2 border-t border-white/10 pt-3">
          <label className="flex items-center gap-3 text-sm">
            <span className="w-40 text-ink-tertiary">spawn chance <strong className="text-ink-secondary">{Math.round(config.spawnChance * 100)}%</strong></span>
            <input type="range" min={0} max={100} value={config.spawnChance * 100}
              onChange={(e) => setConfig((c) => ({ ...c, spawnChance: Number(e.target.value) / 100 }))} />
          </label>
          <label className="flex items-center gap-3 text-sm">
            <span className="w-40 text-ink-tertiary">think time <strong className="text-ink-secondary">{(config.thinkMs / 1000).toFixed(1)}s</strong></span>
            <input type="range" min={100} max={4000} step={100} value={config.thinkMs}
              onChange={(e) => setConfig((c) => ({ ...c, thinkMs: Number(e.target.value) }))} />
          </label>
          <label className="flex items-center gap-3 text-sm">
            <span className="w-32 text-ink-tertiary">max depth <strong className="text-ink-secondary">{config.maxDepth}</strong></span>
            <input type="range" min={0} max={3} value={config.maxDepth}
              onChange={(e) => setConfig((c) => ({ ...c, maxDepth: Number(e.target.value) }))} />
          </label>
          <label className="flex items-center gap-3 text-sm">
            <span className="w-36 text-ink-tertiary">max active <strong className="text-ink-secondary">{config.maxActive}</strong></span>
            <input type="range" min={1} max={16} value={config.maxActive}
              onChange={(e) => setConfig((c) => ({ ...c, maxActive: Number(e.target.value) }))} />
          </label>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} y1={TOP - 18} x2={x(t)} y2={H - AXIS_H} stroke="var(--color-stroke-tertiary)"
              strokeWidth={1} opacity={0.45} />
            <text x={x(t)} y={H - AXIS_H + 16} textAnchor="middle" fontSize={10}
              fill="var(--color-ink-quaternary)">
              {t >= now - 250 ? "now" : `${Math.round((t - now) / 1000)}s`}
            </text>
          </g>
        ))}
        <text x={W - 24} y={TOP - 26} textAnchor="end" fontSize={10} fontWeight={700}
          fill="var(--color-category-red)">now</text>
        <line x1={x(now)} y1={TOP - 20} x2={x(now)} y2={H - AXIS_H}
          stroke="var(--color-category-red)" strokeWidth={1} opacity={0.55} />

        {/* Spawn links: parent's completing turn down to the child lane's first moment. */}
        {sim.spawns.map((s) => {
          const parent = sim.turns.find((t) => t.id === s.parentTurnId);
          const child = sim.agents.find((a) => a.id === s.childAgentId);
          if (parent === undefined || child === undefined) return null;
          const x0 = x(s.at);
          const y0 = yOf(parent.agentId) + BAR_H / 2;
          const y1 = yOf(child.id) + BAR_H / 2;
          return (
            <path key={`${s.parentTurnId}-${s.childAgentId}`}
              d={`M ${x0} ${y0} C ${x0 + 22} ${y0}, ${x0 - 10} ${y1}, ${x0 + 14} ${y1}`}
              fill="none" stroke={color(child.depth)} strokeWidth={1.2} opacity={0.55} />
          );
        })}

        {lanes.map((a) => {
          const y = yOf(a.id);
          const parentY = a.parentId === null ? undefined : laneYRef.current.get(a.parentId);
          const c = color(a.depth);
          const retired = a.retiredAt !== null;
          // A lane fades in over its first moment so a spawn reads as an arrival, not a pop.
          const age = Math.min(1, (now - a.bornAt) / 400);
          const turns = sim.turns.filter((t) => t.agentId === a.id).sort((p, q) => p.startAt - q.startAt);
          return (
            <g key={a.id} opacity={age * (retired ? 0.4 : 1)}>
              {/* Tree rail: drops from the parent lane and turns into this one. Sits in the
                  indent channel to the LEFT of the label, so it never crosses the text. */}
              {a.depth > 0 && parentY !== undefined && (
                // Starts *below* the parent's text baseline, not at its centre: the parent is
                // indented one level less, so its label runs straight across this rail's channel.
                <path
                  d={`M ${LABEL_X + (a.depth - 1) * INDENT + 4} ${parentY + BAR_H + 2}
                      V ${y + BAR_H / 2} h ${INDENT - 7}`}
                  fill="none" stroke={c} strokeWidth={1} opacity={0.5} />
              )}
              <text x={LABEL_X + a.depth * INDENT} y={y + BAR_H - 4} textAnchor="start"
                fontSize={pitch < 24 ? 9.5 : 11}
                fontWeight={a.depth === 0 ? 700 : 600} fill={c}>
                {a.label}
              </text>
              <line x1={GUTTER} y1={y + BAR_H / 2} x2={W - 24} y2={y + BAR_H / 2}
                stroke="var(--color-stroke-tertiary)" strokeWidth={1} opacity={0.3} />

              {/* Dead air between this lane's turns — the gap the idle warp later collapses. */}
              {turns.map((t, i) => {
                const prev = turns[i - 1];
                if (prev?.endAt == null) return null;
                return (
                  <line key={`gap-${t.id}`} x1={x(prev.endAt)} y1={y + BAR_H / 2}
                    x2={x(t.startAt)} y2={y + BAR_H / 2} stroke={c} strokeWidth={1}
                    strokeDasharray="3 4" opacity={0.5} />
                );
              })}

              {turns.map((t) => {
                const x0 = x(t.startAt);
                const x1 = x(t.endAt ?? now);
                const live = t.endAt === null;
                const prefill = t.firstTokenAt === null;
                return (
                  <g key={t.id}>
                    <rect x={x0} y={y} width={Math.max(2, x1 - x0)} height={BAR_H} rx={2}
                      fill={c} fillOpacity={prefill ? 0.16 : 0.42}
                      stroke={c} strokeWidth={live ? 1.6 : 0.8}
                      strokeDasharray={prefill ? "3 3" : undefined} />
                    {/* No cap on the right edge while streaming: the bar is still being written. */}
                    {live && !prefill && (
                      <circle cx={x1} cy={y + BAR_H / 2} r={3.2} fill={c}>
                        <animate attributeName="opacity" values="1;0.25;1" dur="0.9s" repeatCount="indefinite" />
                      </circle>
                    )}
                    {/* A turn that fanned out carries a notch at the moment it did. */}
                    {t.spawnCount > 0 && t.endAt !== null && (
                      <polygon
                        points={`${x1},${y + BAR_H} ${x1 - 5},${y + BAR_H + 6} ${x1 + 5},${y + BAR_H + 6}`}
                        fill={c} opacity={0.9} />
                    )}
                    {x1 - x0 > 30 && BAR_H >= 13 && (
                      <text x={x0 + 5} y={y + BAR_H - 5} fontSize={9}
                        fill="var(--color-ink-secondary)">{t.emitted}/{t.tokens}</text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
