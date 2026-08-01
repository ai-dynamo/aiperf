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
import {
  createSegmentSim,
  internUpTo,
  prefixChain,
  colorForId,
  totalMessages,
  type SegmentSimState,
} from "./segmentSim.js";

const SEED = 1;
const SESSIONS = 9;

type Beat = BeatAnchor & { title: string; lede: string; caption: string };

const BEATS: Beat[] = [
  {
    endAt: 0.1,
    title: "One message at a time",
    lede: "A recorded trace arrives as sessions of messages. Each is lowered on its own.",
    caption: "SegmentPool::intern in rust/runtime/src/dataset/segment.rs.",
    narration:
      "A recorded trace arrives as a set of sessions, each a chain of messages. They are lowered one at a time, in order, and the first thing that happens to a message is that it gets hashed. Watch the arena on the right stay empty until that hash comes back — nothing is stored before its identity is known.",
  },
  {
    endAt: 0.28,
    title: "The parent is part of the identity",
    lede: "The hash folds in the previous message's id, not just this message's content.",
    caption: "payload_id hashes parent id, role, token IDs, then the wire bytes.",
    narration:
      "Here is the part that matters. The hash is not taken over the message alone. It folds in the identity of the message before it — its prefix parent — along with the role, the token identifiers, and the exact wire bytes. So a segment's identity depends on everything that came before it in the conversation. The same sentence, continuing a different conversation, is deliberately a different segment.",
  },
  {
    endAt: 0.55,
    title: "Miss appends, hit costs nothing",
    lede: "A fresh id pushes one entry and takes its index as the handle. A known id returns it.",
    caption: "arena: Vec<Segment> plus ids: HashMap<SegmentId, Handle>; handles are never reused.",
    narration:
      "That identity is then looked up in a map. A miss appends one entry to a dense arena, and the handle is simply its index — assigned on append and never reused. A hit returns the handle that is already there and appends nothing at all. No entry, no wire bytes, no second copy. Watch the counter: the number of messages interned climbs steadily, but the number of segments does not.",
  },
  {
    endAt: 0.8,
    title: "A continued session is nearly free",
    lede: "Sessions that replay an earlier conversation collapse onto the segments already there.",
    caption: "Every session shares a system prompt; several continue an earlier conversation.",
    narration:
      "Now the payoff. Several of these sessions continue an earlier conversation, replaying its turns before adding their own. Because identity is prefix-dependent, every one of those replayed turns hashes to exactly what is already in the arena, and collapses onto it. Look at the first cell — the shared system prompt — and the count beside it. One segment, serving every session in the trace.",
  },
  {
    endAt: 1,
    title: "What the pool is for",
    lede: "One stored copy per distinct prefix, and a body built by concatenating slices.",
    caption: "build_body concatenates stored wire slices and serializes only the override tail.",
    narration:
      "By the end, the arena holds one entry per distinct prefix rather than one per message, and the dedup rate is the fraction of the trace that cost nothing to store. Building a request body is then just concatenating the stored slices for a chain of handles, with only the per-dispatch overrides serialized fresh. That is the whole point: parse once, store once, and send the same bytes without ever rebuilding them.",
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
  const total = useMemo(() => totalMessages(base), [base]);

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
    cache.current = { count: want, state: internUpTo(cache.current.state, want) };
  }
  const sim = cache.current.state;

  const latest = sim.events[sim.events.length - 1];
  const chain = prefixChain(sim.arena, latest?.handle ?? null).slice(0, 6);
  const currentWire = sim.sessions[sim.cursor.session]?.messages[sim.cursor.message]?.wire;
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

        <div className="mb-2 rounded border border-white/10 bg-surface-panel px-3 py-2">
          <div className="mb-0.5 text-[9px] font-bold tracking-widest text-ink-tertiary">
            WIRE — the serialized message being interned
          </div>
          <code className="block overflow-x-auto whitespace-pre font-mono text-[11px] text-ink-primary">
            {currentWire ?? "\u2014"}
          </code>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-[240px_290px_1fr] gap-3">
          <section className="overflow-hidden rounded-lg border border-white/10 bg-surface-elevated p-2.5">
            <h2 className="mb-1.5 text-[10px] font-bold tracking-widest text-ink-secondary">DYNAMO TRACE</h2>
            <div className="flex flex-col gap-1">
              {sim.sessions.map((s, si) => (
                <div key={s.id} className="flex items-center gap-1.5"
                  style={{ opacity: si > sim.cursor.session ? 0.3 : 1 }}>
                  <span className="w-6 text-[10px] font-semibold text-ink-tertiary">{s.id}</span>
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
            <h2 className="mb-1.5 text-[10px] font-bold tracking-widest text-ink-secondary">
              INTERN — hash, then look up
            </h2>
            {latest === undefined ? (
              <p className="text-ink-quaternary">Waiting for the first message…</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                <Row label="parent id" value={latest.parent === null ? "none (root)" : sim.arena[latest.parent]?.id ?? "?"} />
                <Row label="role" value={sim.arena[latest.handle]?.role ?? "?"} />
                <Row label="tokens" value={String(sim.arena[latest.handle]?.tokens ?? 0)} />
                <div className="text-center text-[10px] text-ink-quaternary">
                  ↓ blake3(parent ‖ role ‖ tokens ‖ wire)
                </div>
                <Row label="segment id" value={latest.id} mono />
                <div className="rounded px-2 py-1.5 text-center text-xs font-bold text-black"
                  style={{ background: latest.hit ? "var(--color-category-green)" : "var(--color-category-orange)" }}>
                  {latest.hit ? `HIT → reuse ${latest.handle}` : `MISS → append ${latest.handle}`}
                </div>
                {chain.length > 1 && (
                  <div className="mt-0.5">
                    <div className="mb-1 text-[9px] font-semibold tracking-widest text-ink-tertiary">PREFIX CHAIN</div>
                    <div className="flex flex-wrap items-center gap-1">
                      {chain.map((h, i) => (
                        <span key={h} className="flex items-center gap-1">
                          {i > 0 && <span className="text-ink-quaternary">←</span>}
                          <span className="rounded px-1 py-0.5 text-[9px] font-semibold tabular-nums"
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
            <h2 className="mb-1.5 text-[10px] font-bold tracking-widest text-ink-secondary">
              SEGMENT POOL — dense arena, append only
            </h2>
            <div className="flex flex-wrap gap-1">
              {sim.arena.map((seg) => {
                const isCurrent = latest?.handle === seg.handle;
                const inChain = chain.includes(seg.handle);
                return (
                  <div key={seg.handle} title={`handle ${seg.handle} · ${seg.role} · ${seg.refs} refs`}
                    className="flex flex-col items-center justify-center rounded"
                    style={{
                      width: 40, height: 40, background: colorForId(seg.id),
                      opacity: isCurrent ? 1 : inChain ? 0.85 : 0.32,
                      outline: isCurrent
                        ? `2px solid ${latest?.hit ? "var(--color-category-green)" : "var(--color-category-orange)"}`
                        : inChain ? "1px solid rgba(255,255,255,0.35)" : undefined,
                    }}>
                    <span className="text-[10px] font-bold tabular-nums text-black">{seg.handle}</span>
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
