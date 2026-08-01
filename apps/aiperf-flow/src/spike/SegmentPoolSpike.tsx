/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — a Dynamo trace lowering into a segment pool, live.
//!
//! Three columns, left to right in the order the data moves: the recorded sessions, the intern
//! decision for the message currently being lowered, and the dense arena filling up. The arena is
//! append-only, so a hit adds nothing at all — which is the saving, drawn.

import { useEffect, useMemo, useRef, useState } from "react";
import { TraceRecordPanel } from "./TraceRecordPanel.js";
import {
  createSegmentSim,
  stepSegments,
  prefixChain,
  colorForId,
  messageWire,
  type SegmentSimState,
} from "./segmentSim.js";

const SPEEDS = [1, 0.5, 0.25, 0.1] as const;
const CELL = 46;
const CELL_GAP = 5;

export function SegmentPoolSpike(): React.JSX.Element {
  const [seed, setSeed] = useState(1);
  const [running, setRunning] = useState(true);
  // One message per 260ms tick at 1x is quicker than the intern panel can be read.
  const [speed, setSpeed] = useState(0.25);
  const [, force] = useState(0);

  const simRef = useRef<SegmentSimState>(createSegmentSim(seed, 9));
  const runningRef = useRef(running);
  runningRef.current = running;
  const speedRef = useRef(speed);
  speedRef.current = speed;

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      if (runningRef.current && !simRef.current.done) {
        simRef.current = stepSegments(simRef.current, dt * speedRef.current);
        force((n) => n + 1);
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const sim = simRef.current;
  const restart = (s: number) => {
    simRef.current = createSegmentSim(s, 9);
    force((n) => n + 1);
  };

  const latest = sim.events[sim.events.length - 1];
  const chain = useMemo(
    () => prefixChain(sim.arena, latest?.handle ?? null).slice(0, 6),
    [sim.arena, latest?.handle],
  );
  // Interning shows the message about to be hashed; materializing shows the stored wire a worker
  // is copying out right now — the same bytes, read back.
  const readingHandle = sim.workers.find((w) => w.reading !== null)?.reading ?? null;
  const currentWire =
    sim.phase === "materialize"
      ? readingHandle === null
        ? undefined
        : messageWire(sim.arena[readingHandle]!.role, sim.arena[readingHandle]!.text)
      : sim.sessions[sim.cursor.session]?.messages[sim.cursor.message]?.wire;
  const dedup = sim.interned > 0 ? sim.hits / sim.interned : 0;
  const saved = sim.bytesNaive > 0 ? 1 - sim.bytesStored / sim.bytesNaive : 0;

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-2xl font-extrabold">A Dynamo trace becoming a segment pool</h1>
      </div>
      <p className="mb-4 max-w-4xl text-sm text-ink-secondary">
        Every message is hashed together with <em>its prefix parent's id</em>, then looked up. A miss
        appends one entry to a dense arena and the handle is its index; a hit returns the existing
        handle and appends nothing. That is why a session which continues an earlier one costs
        almost nothing to store — and why the same text under a different parent is deliberately a
        different segment.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => setRunning((r) => !r)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold">
              {running ? "Pause" : "Run"}
            </button>
            <button type="button"
              onClick={() => { simRef.current = stepSegments(simRef.current, 260); setRunning(false); force((n) => n + 1); }}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Step
            </button>
            <button type="button" onClick={() => restart(seed)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Restart
            </button>
            <button type="button" onClick={() => { const n = seed + 1; setSeed(n); restart(n); }}
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

          <div className="ml-auto flex items-center gap-5 text-sm tabular-nums">
            <span><span className="text-ink-tertiary">interned</span> <strong>{sim.interned}</strong></span>
            <span><span className="text-ink-tertiary">segments</span> <strong>{sim.arena.length}</strong></span>
            <span><span className="text-ink-tertiary">dedup</span>{" "}
              <strong style={{ color: "var(--color-category-green)" }}>{Math.round(dedup * 100)}%</strong></span>
            <span><span className="text-ink-tertiary">wire saved</span>{" "}
              <strong style={{ color: "var(--color-category-orange)" }}>{Math.round(saved * 100)}%</strong></span>
            <span className="rounded px-2 py-0.5 text-[11px] font-bold"
              style={{ background: sim.phase === "intern"
                ? "var(--color-category-orange)" : "var(--color-category-cyan)", color: "#000" }}>
              {sim.phase === "intern" ? "INTERNING" : sim.done ? "MATERIALIZED" : "MATERIALIZING"}
            </span>
          </div>
        </div>
      </div>

      <div className="mb-3">
        <TraceRecordPanel turn={sim.turn} parentSessionId={null} />
      </div>

      <div className="grid grid-cols-[300px_320px_1fr] gap-4">
        {/* 1 — the recorded trace */}
        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[11px] font-bold tracking-widest text-ink-secondary">
            DYNAMO TRACE
          </h2>
          <div className="flex flex-col gap-1.5">
            {sim.sessions.map((s, si) => {
              const state = si < sim.cursor.session ? "done" : si === sim.cursor.session ? "live" : "todo";
              return (
                <div key={s.id}
                  className="flex items-center gap-2 rounded px-1.5 py-1"
                  style={{ background: state === "live" ? "rgba(255,255,255,0.05)" : undefined,
                           opacity: state === "todo" ? 0.35 : 1 }}>
                  <span className="w-7 text-xs font-semibold text-ink-tertiary">{s.id}</span>
                  <div className="flex flex-wrap gap-[3px]">
                    {s.messages.map((m, mi) => {
                      const handle = sim.resolved.get(`${si}:${mi}`);
                      const now = state === "live" && mi === sim.cursor.message;
                      // Coloured by the segment it became, so two cells matching in different
                      // sessions *is* the dedup — visible without reading a number.
                      const bg = handle === undefined
                        ? "var(--color-stroke-secondary)"
                        : colorForId(sim.arena[handle]!.id);
                      return (
                        <span key={mi} title={`${m.role}: ${m.text}`}
                          style={{
                            width: 10, height: 14, borderRadius: 2, background: bg,
                            opacity: handle === undefined ? 0.35 : 1,
                            outline: now ? "2px solid var(--color-category-red)" : undefined,
                          }} />
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] leading-snug text-ink-quaternary">
            Each cell is one message. Sessions that continue an earlier one replay its turns first.
          </p>
        </section>

        {/* 2 — the intern decision */}
        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[11px] font-bold tracking-widest text-ink-secondary">
            {sim.phase === "intern" ? "INTERN — hash, then look up" : "MATERIALIZE — workers read it back"}
          </h2>
          {sim.phase === "materialize" && (
            <div className="flex flex-col gap-2 text-xs">
              <p className="text-[11px] leading-snug text-ink-quaternary">
                The pool is frozen. Each worker walks a chain of handles, clones the stored wire for
                each, and appends a pre-serialized override tail. Nothing is decoded and nothing is
                re-validated — the wires are well-formed by construction.
              </p>
              {sim.workers.map((w) => (
                <div key={w.id} className="rounded border border-white/10 bg-surface-panel p-2">
                  <div className="mb-1 flex items-baseline justify-between">
                    <span className="font-semibold text-ink-primary">worker {w.id}</span>
                    <span className="text-[10px] tabular-nums text-ink-tertiary">
                      {w.done ? "idle" : `${w.sessionId} · ${w.cursor}/${w.chain.length}`}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-[3px]">
                    {w.chain.map((h, i) => (
                      <span key={i} style={{
                        width: 9, height: 12, borderRadius: 2,
                        background: colorForId(sim.arena[h]?.id ?? ""),
                        opacity: i < w.cursor ? 1 : 0.22,
                        outline: i === w.cursor && !w.done ? "2px solid var(--color-category-cyan)" : undefined,
                      }} />
                    ))}
                  </div>
                  <div className="mt-1 text-[10px] tabular-nums text-ink-quaternary">
                    {w.bytes} bytes concatenated{w.cursor >= w.chain.length && !w.done ? " · body complete" : ""}
                  </div>
                </div>
              ))}
              <div className="flex items-baseline justify-between text-[11px]">
                <span className="text-ink-tertiary">bodies built</span>
                <strong className="text-ink-primary tabular-nums">{sim.bodiesBuilt}/{sim.sessions.length}</strong>
              </div>
              <div className="flex items-baseline justify-between text-[11px]">
                <span className="text-ink-tertiary">bytes copied out</span>
                <strong className="text-ink-primary tabular-nums">{sim.bytesMaterialized}</strong>
              </div>
            </div>
          )}
          {sim.phase === "intern" && (<>
          {latest === undefined ? (
            <p className="text-sm text-ink-quaternary">Waiting for the first message…</p>
          ) : (
            <div className="flex flex-col gap-2 text-xs">
              <Row label="parent id" value={latest.parent === null ? "none (root)" : sim.arena[latest.parent]?.id ?? "?"} />
              <Row label="role" value={sim.arena[latest.handle]?.role ?? "?"} />
              <Row label="tokens" value={String(sim.arena[latest.handle]?.tokens ?? 0)} />
              <div className="my-1 text-center text-ink-quaternary">↓ blake3(parent ‖ role ‖ tokens ‖ wire)</div>
              <Row label="segment id" value={latest.id} mono />
              <div className={`mt-1 rounded px-2 py-2 text-center text-sm font-bold ${
                latest.hit ? "text-black" : "text-black"}`}
                style={{ background: latest.hit ? "var(--color-category-green)" : "var(--color-category-orange)" }}>
                {latest.hit ? `HIT → reuse handle ${latest.handle}` : `MISS → append handle ${latest.handle}`}
              </div>
              <p className="text-[11px] leading-snug text-ink-quaternary">
                {latest.hit
                  ? "The id was already registered, so nothing is appended and no wire bytes are stored again."
                  : "A fresh id: one entry is pushed onto the arena and its index becomes the handle."}
              </p>
              {chain.length > 1 && (
                <div className="mt-1">
                  <div className="mb-1 text-[10px] font-semibold tracking-widest text-ink-tertiary">
                    PREFIX CHAIN
                  </div>
                  <div className="flex flex-wrap items-center gap-1">
                    {chain.map((h, i) => (
                      <span key={h} className="flex items-center gap-1">
                        {i > 0 && <span className="text-ink-quaternary">←</span>}
                        <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                          style={{ background: colorForId(sim.arena[h]!.id), color: "#000" }}>
                          {h}
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}</>)}
        </section>

        {/* 3 — the arena */}
        <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
          <h2 className="mb-2 text-[11px] font-bold tracking-widest text-ink-secondary">
            SEGMENT POOL — dense arena, append only
          </h2>
          <div className="flex flex-wrap" style={{ gap: CELL_GAP }}>
            {sim.arena.map((seg) => {
              const isCurrent = latest?.handle === seg.handle;
              const inChain = chain.includes(seg.handle);
              return (
                <div key={seg.handle}
                  title={`handle ${seg.handle} · ${seg.role} · ${seg.tokens} tok · ${seg.refs} refs\n${seg.text}`}
                  className="flex flex-col items-center justify-center rounded"
                  style={{
                    width: CELL, height: CELL,
                    background: colorForId(seg.id),
                    opacity: isCurrent ? 1 : inChain ? 0.82 : 0.34,
                    outline: isCurrent
                      ? `3px solid ${
                          sim.phase === "materialize"
                            ? "var(--color-category-cyan)"
                            : latest?.hit
                              ? "var(--color-category-green)"
                              : "var(--color-category-orange)"
                        }`
                      : inChain ? "1px solid rgba(255,255,255,0.35)" : undefined,
                  }}>
                  <span className="text-[11px] font-bold tabular-nums text-black">{seg.handle}</span>
                  <span className="text-[8px] tabular-nums text-black/70">{seg.id}</span>
                  {seg.refs > 1 && (
                    <span className="text-[8px] font-bold text-black/80">×{seg.refs}</span>
                  )}
                </div>
              );
            })}
            {sim.arena.length === 0 && (
              <p className="text-sm text-ink-quaternary">Empty. The first message will land at handle 0.</p>
            )}
          </div>
          <p className="mt-3 text-[11px] leading-snug text-ink-quaternary">
            Handle = arena index, assigned on append and never reused. A cell marked ×n was
            resolved n times — every one after the first cost nothing. Highlighted cells are the
            current message's prefix chain back to its root.
          </p>
        </section>
      </div>

      {/* Below the pool, because this is what a segment *stores*: the exact bytes that are hashed and stored. `intern_message` serializes the dialect
          message and folds these very bytes into the identity, so this is the unit of work. */}
      <div className="mb-3 rounded-lg border border-white/10 bg-surface-panel px-4 py-2.5">
        <div className="mb-1 text-[10px] font-bold tracking-widest text-ink-tertiary">
          WIRE — the serialized message being interned
        </div>
        <code className="block overflow-x-auto whitespace-pre font-mono text-[12px] text-ink-primary">
          {currentWire ?? "\u2014"}
        </code>
        <div className="mt-1 text-[10px] text-ink-quaternary">
          {currentWire === undefined
            ? "Waiting for the first message."
            : `${currentWire.length} bytes \u00b7 hashed together with the parent id, role and token ids`}
        </div>
      </div>

      <p className="mt-3 text-[11px] text-ink-quaternary">
        Modelled on <code>rust/runtime/src/dataset/segment.rs</code>: <code>SegmentPool</code>'s
        arena plus id map, and <code>payload_id</code> folding the parent id into the hash. The
        digest here is a short stand-in; the structure — prefix-dependence, dense handles,
        append-or-reuse — is the real one. Sessions are generated, not a captured trace.
      </p>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-ink-tertiary">{label}</span>
      <span className={mono === true ? "font-mono text-ink-primary" : "text-ink-primary"}>{value}</span>
    </div>
  );
}
