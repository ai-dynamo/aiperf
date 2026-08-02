/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the segment pool, narrated. The voice decides how many messages have been interned.
//!
//! Progress is a message count rather than elapsed time, so the arena is exactly as full as the
//! narration says it is. Nothing races ahead of the explanation.

import { useMemo, useRef } from "react";
import { PresentationShell } from "../shell/PresentationShell.js";
import { useNarratedDeck } from "../audio/index.js";
import type { SlideDefinition } from "../deck/types.js";
import { useBeatClock, type BeatAnchor } from "./useBeatClock.js";
import { TraceRecordPanel } from "./TraceRecordPanel.js";
import {
  createSegmentSim,
  advanceToTick,
  prefixChain,
  colorForId,
  messageWire,
  totalTicks,
  type SegmentSimState,
} from "./segmentSim.js";

const SEED = 1;
const SESSIONS = 9;

type Beat = BeatAnchor & { title: string; lede: string; caption: string };

const BEATS: Beat[] = [
  {
    endAt: 0.12,
    title: "A trace record, building as the turn grows",
    lede: "One dynamo.request.trace.v1 line per request_end, with its KV block hashes.",
    caption: "Schema: rust/runtime/src/graph/recorded/dynamo/schema.rs.",
    narration:
      "This is what a Dynamo trace line actually looks like. One record per completed request, carrying the agent context, the token counts, and — the part that matters here — a replay section with a list of block hashes. Each hash covers thirty-two tokens of the prompt. Watch the list at the bottom of the record grow as the conversation accumulates: every time the input crosses another block boundary, one more hash is appended.",
  },
  {
    endAt: 0.3,
    title: "Orange is new, green was already cached",
    lede: "A block hash the server has seen before is a prefix hit; its tokens never recompute.",
    caption: "cached_tokens counts exactly the tokens covered by already-known blocks.",
    narration:
      "The colours on those blocks are the whole reason the trace ships them. An orange block is new — the server has never seen that prefix, so those tokens have to be processed. A green block is one it has seen before, so the key-value cache already holds it and the work is skipped entirely. Look at the cached-tokens field in the record: it is exactly the green blocks, multiplied by the block size. As later turns in a session replay the same opening, more and more of their blocks come back green.",
  },
  {
    endAt: 0.5,
    title: "Each message is interned under its parent",
    lede: "The hash folds in the previous message's id, so identity is prefix-dependent.",
    caption: "payload_id in rust/runtime/src/dataset/segment.rs.",
    narration:
      "Underneath the trace, the messages themselves are being lowered into a pool. Each one is hashed together with the identity of the message before it — its prefix parent — plus the role, the token identifiers, and the exact wire bytes shown at the bottom of the screen. Because the parent is folded in, the same sentence continuing a different conversation is deliberately a different segment. A fresh identity appends one entry to a dense arena and takes its index as the handle. A familiar one returns the handle already there, and appends nothing at all.",
  },
  {
    endAt: 0.66,
    title: "A continued session collapses onto what is there",
    lede: "One stored copy per distinct prefix, not one per message.",
    caption: "Matching colours in the trace column are the same segment, reused.",
    narration:
      "Several of these sessions continue an earlier conversation, replaying its turns before adding their own. Every replayed turn hashes to exactly what the arena already holds, and collapses onto it. That is what the matching colours on the left mean: two cells the same colour in different sessions are not two similar messages, they are one segment being reused. By the end the pool holds one entry per distinct prefix rather than one per message.",
  },
  {
    endAt: 0.86,
    title: "Now the workers read it back",
    lede: "Each body is a walk down a chain of handles, cloning stored slices.",
    caption: "build_body_from_handles in rust/runtime/src/dataset/materialize.rs.",
    narration:
      "With the trace fully interned, the pool freezes and the workers start. Building a request body is now just a walk down a chain of handles: for each one, clone the wire already stored in the arena, and concatenate. Watch the arena light up as they read. Several workers pull the very same entries at the same time, because their conversations share a prefix — one stored copy, many readers, no coordination between them.",
  },
  {
    endAt: 1,
    title: "Nothing is decoded, nothing is rebuilt",
    lede: "Concatenate stored slices, append a pre-serialized override tail, send.",
    caption: "The override tail is serialized once per distinct value, not per request.",
    narration:
      "And notice what does not happen. No message is parsed, no JSON is re-serialized, and nothing is validated on the way out — the stored wires are well-formed by construction, so re-scanning them per request would be pure overhead. The only fresh bytes are the per-dispatch overrides, and even those are serialized once per distinct value and reused. Parse once, store once, and send the same bytes for as long as the run lasts.",
  },
];

const SLIDES: readonly SlideDefinition[] = BEATS.map((b, i) => ({
  id: `seg-${i}`,
  eyebrow: `${String(i + 1).padStart(2, "0")} · SEGMENT POOL`,
  title: b.title,
  lede: b.lede,
  narration: b.narration,
  caption: b.caption,
  nodes: [],
  edges: [],
}));

export function SegmentPoolNarratedSpike(): React.JSX.Element {
  const base = useMemo(() => createSegmentSim(SEED, SESSIONS), []);
  // Narration spans interning *and* materialization, so it drives ticks rather than messages.
  const total = useMemo(() => totalTicks(SEED, SESSIONS), []);

  const narrated = useNarratedDeck({
    narrations: BEATS.map((b) => b.narration),
    storagePrefix: "spike:segments-narrated",
  });

  const { position } = useBeatClock(BEATS, narrated.index, narrated.activeWordIndex, total);

  // The pool is a pure function of how many messages have been interned, so it is recomputed from
  // the narration position rather than advanced by a timer. Cached because narration re-renders on
  // every spoken word and replaying the whole trace each time would be wasteful.
  const cache = useRef<{ count: number; state: SegmentSimState }>({ count: 0, state: base });
  const want = Math.max(0, Math.min(total, Math.round(position)));
  if (want < cache.current.count) cache.current = { count: 0, state: base };
  if (want > cache.current.count) {
    cache.current = { count: want, state: advanceToTick(cache.current.state, want) };
  }
  const sim = cache.current.state;

  const latest = sim.events[sim.events.length - 1];
  const chain = prefixChain(sim.arena, latest?.handle ?? null).slice(0, 6);
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
    <PresentationShell
      slides={SLIDES}
      slideIndex={narrated.index}
      onSlideIndexChange={narrated.goTo}
      narrated={narrated}
      title="The segment pool, narrated"
    >
      <div className="flex h-full flex-col px-6 pt-3">
        <div className="mb-2 flex items-center gap-5 text-sm tabular-nums">
          <span><span className="text-ink-tertiary">interned</span> <strong>{sim.interned}</strong>
            <span className="text-ink-quaternary"> / {total}</span></span>
          <span><span className="text-ink-tertiary">segments</span> <strong>{sim.arena.length}</strong></span>
          <span><span className="text-ink-tertiary">dedup</span>{" "}
            <strong style={{ color: "var(--color-category-green)" }}>{Math.round(dedup * 100)}%</strong></span>
          <span><span className="text-ink-tertiary">wire saved</span>{" "}
            <strong style={{ color: "var(--color-category-orange)" }}>{Math.round(saved * 100)}%</strong></span>
        </div>

        <div className="mb-2">
          <TraceRecordPanel turn={sim.turn} parentSessionId={null} />
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-[240px_290px_1fr] gap-3">
          <section className="overflow-hidden rounded-lg border border-white/10 bg-surface-elevated p-2.5">
            <h2 className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">DYNAMO TRACE</h2>
            <div className="flex flex-col gap-1">
              {sim.sessions.map((s, si) => (
                <div key={s.id} className="flex items-center gap-1.5"
                  style={{ opacity: si > sim.cursor.session ? 0.3 : 1 }}>
                  <span className="w-6 text-[13px] font-semibold text-ink-tertiary">{s.id}</span>
                  <div className="flex flex-wrap gap-[2px]">
                    {s.messages.map((m, mi) => {
                      const handle = sim.resolved.get(`${si}:${mi}`);
                      const now = si === sim.cursor.session && mi === sim.cursor.message;
                      // Coloured by the segment it became: matching cells across sessions are
                      // the dedup, visible without reading a number.
                      const bg = handle === undefined
                        ? "var(--color-stroke-secondary)"
                        : colorForId(sim.arena[handle]!.id);
                      return (
                        <span key={mi} title={`${m.role}: ${m.text}`} style={{
                          width: 8, height: 12, borderRadius: 2, background: bg,
                          opacity: handle === undefined ? 0.35 : 1,
                          outline: now ? "2px solid var(--color-category-red)" : undefined,
                        }} />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-white/10 bg-surface-elevated p-2.5 text-xs">
            <h2 className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">
              {sim.phase === "intern" ? "INTERN — hash, then look up" : "MATERIALIZE — workers read it back"}
            </h2>
            {sim.phase === "materialize" ? (
              <div className="flex flex-col gap-1.5">
                {sim.workers.map((w) => (
                  <div key={w.id} className="rounded border border-white/10 bg-surface-panel p-1.5">
                    <div className="flex items-baseline justify-between">
                      <span className="text-[13px] font-semibold text-ink-primary">worker {w.id}</span>
                      <span className="text-[12px] tabular-nums text-ink-tertiary">
                        {w.done ? "idle" : `${w.sessionId} · ${w.cursor}/${w.chain.length}`}
                      </span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-[2px]">
                      {w.chain.map((h, i) => (
                        <span key={i} style={{
                          width: 8, height: 11, borderRadius: 2,
                          background: colorForId(sim.arena[h]?.id ?? ""),
                          opacity: i < w.cursor ? 1 : 0.22,
                          outline: i === w.cursor && !w.done ? "2px solid var(--color-category-cyan)" : undefined,
                        }} />
                      ))}
                    </div>
                    <div className="mt-0.5 text-[12px] tabular-nums text-ink-quaternary">
                      {w.bytes} bytes concatenated
                    </div>
                  </div>
                ))}
                <div className="flex items-baseline justify-between text-[13px]">
                  <span className="text-ink-tertiary">bodies built</span>
                  <strong className="tabular-nums text-ink-primary">{sim.bodiesBuilt}/{sim.sessions.length}</strong>
                </div>
                <p className="text-[13px] leading-snug text-ink-quaternary">
                  Clone each handle's stored wire, concatenate, append a pre-serialized override
                  tail. Nothing decoded, nothing re-validated.
                </p>
              </div>
            ) : latest === undefined ? (
              <p className="text-ink-quaternary">Waiting for the first message…</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                <Row label="parent id" value={latest.parent === null ? "none (root)" : sim.arena[latest.parent]?.id ?? "?"} />
                <Row label="role" value={sim.arena[latest.handle]?.role ?? "?"} />
                <Row label="tokens" value={String(sim.arena[latest.handle]?.tokens ?? 0)} />
                <div className="text-center text-[13px] text-ink-quaternary">
                  ↓ blake3(parent ‖ role ‖ tokens ‖ wire)
                </div>
                <Row label="segment id" value={latest.id} mono />
                <div className="rounded px-2 py-1.5 text-center text-xs font-bold text-black"
                  style={{ background: latest.hit ? "var(--color-category-green)" : "var(--color-category-orange)" }}>
                  {latest.hit ? `HIT → reuse ${latest.handle}` : `MISS → append ${latest.handle}`}
                </div>
                {chain.length > 1 && (
                  <div className="mt-0.5">
                    <div className="mb-1 text-[12px] font-semibold tracking-widest text-ink-tertiary">PREFIX CHAIN</div>
                    <div className="flex flex-wrap items-center gap-1">
                      {chain.map((h, i) => (
                        <span key={h} className="flex items-center gap-1">
                          {i > 0 && <span className="text-ink-quaternary">←</span>}
                          <span className="rounded px-1 py-0.5 text-[12px] font-semibold tabular-nums"
                            style={{ background: colorForId(sim.arena[h]!.id), color: "#000" }}>{h}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="overflow-hidden rounded-lg border border-white/10 bg-surface-elevated p-2.5">
            <h2 className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">
              SEGMENT POOL — dense arena, append only
            </h2>
            <div className="flex flex-wrap gap-1">
              {sim.arena.map((seg) => {
                const reader = sim.workers.find((w) => w.reading === seg.handle);
                const isCurrent = sim.phase === "intern" ? latest?.handle === seg.handle : reader !== undefined;
                const inChain = sim.phase === "intern" ? chain.includes(seg.handle) : false;
                return (
                  <div key={seg.handle} title={`handle ${seg.handle} · ${seg.role} · ${seg.refs} refs`}
                    className="flex flex-col items-center justify-center rounded"
                    style={{
                      width: 40, height: 40, background: colorForId(seg.id),
                      opacity: isCurrent ? 1 : inChain ? 0.85 : 0.32,
                      outline: isCurrent
                        ? `3px solid ${sim.phase === "materialize"
                            ? "var(--color-category-cyan)"
                            : latest?.hit ? "var(--color-category-green)" : "var(--color-category-orange)"}`
                        : inChain ? "1px solid rgba(255,255,255,0.35)" : undefined,
                    }}>
                    <span className="text-[13px] font-bold tabular-nums text-black">{seg.handle}</span>
                    {seg.refs > 1 && <span className="text-[8px] font-bold text-black/80">×{seg.refs}</span>}
                  </div>
                );
              })}
              {sim.arena.length === 0 && (
                <p className="text-xs text-ink-quaternary">Empty. The first message lands at handle 0.</p>
              )}
            </div>
          </section>
        </div>

        <div className="mt-2 mb-2 rounded border border-white/10 bg-surface-panel px-3 py-2">
          <div className="mb-0.5 text-[12px] font-bold tracking-widest text-ink-tertiary">
            WIRE — the serialized message being interned, and the bytes a segment stores
          </div>
          <code className="block overflow-x-auto whitespace-pre font-mono text-[13px] text-ink-primary">
            {currentWire ?? "\u2014"}
          </code>
        </div>
      </div>
    </PresentationShell>
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
