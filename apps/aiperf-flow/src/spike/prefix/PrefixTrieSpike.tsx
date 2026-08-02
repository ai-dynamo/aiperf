/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — why the same bytes are two different segments.
//!
//! The segment-pool spike shows the pool as a flat arena. The structure it actually has is a trie,
//! because a segment's identity folds its parent's identity in: content that arrives by a
//! different path is a different segment, and once two conversations diverge they can never share
//! a slot again no matter how identical their later turns are.
//!
//! Everything drawn here is derived from the same functions the tests pin, which mirror
//! `text_payload_id` and `push_interned` in `rust/runtime/src/dataset/segment.rs`.

import { useEffect, useMemo, useState } from "react";
import {
  childrenOf,
  createPool,
  defaultConversations,
  hashInput,
  internConversation,
  poolStats,
  segmentId,
  tokenize,
  type HashField,
  type Pool,
  type Role,
  type Segment,
} from "./prefixSim.js";
import { ControlBar, Legend, LegendItem, Panel, Readout, SourceNote, SpikeHeader, Toggle } from "../ui.js";

const CONVERSATIONS = defaultConversations();
const CONV_COLOR: Record<string, string> = {
  A: "var(--color-category-blue)",
  B: "var(--color-category-green)",
  C: "var(--color-category-orange)",
};
const ROLE_COLOR: Record<Role, string> = {
  system: "var(--color-category-purple)",
  user: "var(--color-category-cyan)",
  assistant: "var(--color-category-gray)",
};
const DEDUP = "var(--color-category-green)";
const FRESH = "var(--color-category-yellow)";

/** Every turn across every conversation, in the order they are interned. */
const STEPS = CONVERSATIONS.flatMap((c) => c.turns.map((_, i) => ({ conversation: c.id, turnIndex: i })));

/** Replay the pool from empty up to `step` turns. Cheap enough to redo per render. */
function poolAt(step: number): Pool {
  const pool = createPool();
  let remaining = step;
  for (const conversation of CONVERSATIONS) {
    if (remaining <= 0) break;
    const take = Math.min(remaining, conversation.turns.length);
    internConversation(pool, { ...conversation, turns: conversation.turns.slice(0, take) });
    remaining -= take;
  }
  return pool;
}

export function PrefixTrieSpike(): React.JSX.Element {
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);

  const pool = useMemo(() => poolAt(step), [step]);
  const stats = poolStats(pool);
  const last = pool.log.at(-1);

  useEffect(() => {
    if (!running) return undefined;
    const handle = window.setInterval(() => {
      setStep((s) => {
        if (s >= STEPS.length) {
          setRunning(false);
          return s;
        }
        return s + 1;
      });
    }, 1400);
    return () => window.clearInterval(handle);
  }, [running]);

  // The turn about to be interned, so the hash input can be shown before it lands.
  const pending = STEPS[step];
  const pendingTurn =
    pending === undefined
      ? undefined
      : CONVERSATIONS.find((c) => c.id === pending.conversation)?.turns[pending.turnIndex];

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <SpikeHeader title="Same bytes, different segment">
        <p>
          A segment&apos;s identity is a hash of its content <em>and its parent&apos;s identity</em>.
          The parent id is not stored alongside the hash; it is fed into the hash, before the role
          and the tokens. So two turns with identical text, identical roles and identical token IDs
          are the <strong>same</strong> segment when they sit at the same point in a conversation,
          and <strong>different</strong> segments when they do not.
        </p>
        <p>
          Step through the three conversations below. A and B open identically, so B&apos;s first
          three turns cost nothing — the pool hands back the handles it already had. C says the
          same words as A but without the system prompt, and shares nothing at all: its first turn
          inherits a different parent, so every turn after it does too. Prefixes converge; once
          they fork, they never rejoin.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <Toggle active onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Play"}</Toggle>
          <Toggle onClick={() => { setRunning(false); setStep((s) => Math.min(STEPS.length, s + 1)); }}>
            Intern next turn
          </Toggle>
          <Toggle onClick={() => { setRunning(false); setStep(0); }}>Reset</Toggle>
        </div>

        <div className="flex items-center gap-4">
          {CONVERSATIONS.map((c) => (
            <span key={c.id} className="flex items-center gap-1.5 text-[15px]">
              <span style={{ color: CONV_COLOR[c.id] }}>●</span>
              <span className="text-ink-tertiary">{c.label}</span>
            </span>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-6">
          <Readout label="turns interned" value={stats.interned} />
          <Readout label="segments stored" value={stats.stored} color={FRESH} />
          <Readout label="dedup hits" value={stats.deduped} color={DEDUP} />
        </div>
      </ControlBar>

      <div className="grid grid-cols-[1fr_560px] gap-4">
        <Panel label="THE POOL, AS IT REALLY IS" hint="a trie — an edge is “my parent's id went into my hash”">
          <Legend>
            <LegendItem mark="▬" color={ROLE_COLOR.system}>system</LegendItem>
            <LegendItem mark="▬" color={ROLE_COLOR.user}>user</LegendItem>
            <LegendItem mark="▬" color={ROLE_COLOR.assistant}>assistant</LegendItem>
            <LegendItem mark="◆" color={DEDUP}>reused by a later conversation</LegendItem>
          </Legend>
          <Trie pool={pool} lastHandle={last?.handle ?? null} lastDeduped={last?.deduped ?? false} />
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel label="WHAT GETS HASHED" hint="text_payload_id, field by field">
            {pendingTurn === undefined ? (
              <p className="py-6 text-center text-[15px] text-ink-quaternary">
                Every turn interned. Reset to watch it again.
              </p>
            ) : (
              <HashInputView
                parentId={parentIdFor(pool, pending!.conversation, pending!.turnIndex)}
                role={pendingTurn.role}
                text={pendingTurn.text}
                conversation={pending!.conversation}
              />
            )}
          </Panel>

          <Panel label="THE ARENA" hint="dense, append-only — a dedup hit allocates nothing">
            <Arena pool={pool} lastHandle={last?.handle ?? null} lastDeduped={last?.deduped ?? false} />
          </Panel>
        </div>
      </div>

      <Collision />

      <SourceNote>
        Modelled on <code>rust/runtime/src/dataset/segment.rs</code>:{" "}
        <code>text_payload_id</code> at :620, <code>hash_parent</code> at :632, and{" "}
        <code>push_interned</code> at :339, whose dedup index is what makes identity have to be
        exact. The property this page is about is pinned by that file&apos;s own test,{" "}
        <code>handles_are_dense_and_dedup_by_prefix_dependent_id</code>. The digest is FNV-1a
        rather than BLAKE3 — what is faithful is the field order fed into it, not the mixing
        function.
      </SourceNote>
    </div>
  );
}

/** The parent id a turn will hash against, given how far the replay has got. */
function parentIdFor(pool: Pool, conversation: string, turnIndex: number): string | null {
  if (turnIndex === 0) return null;
  const previous = pool.log.filter((e) => e.conversation === conversation).at(-1);
  if (previous === undefined) return null;
  return pool.arena[previous.handle]?.id ?? null;
}

/**
 * The hash input as labelled byte runs.
 *
 * The parent run is the point: it is inside the input, not beside it. When it is empty the segment
 * is a root, and every descendant of that root is keyed differently from the same content reached
 * through a parent.
 */
function HashInputView({
  parentId,
  role,
  text,
  conversation,
}: {
  parentId: string | null;
  role: Role;
  text: string;
  conversation: string;
}): React.JSX.Element {
  const tokens = tokenize(text);
  const fields = hashInput(parentId, role, tokens);
  const id = segmentId(parentId, role, tokens);

  return (
    <div>
      <div className="mb-2 flex items-baseline gap-2">
        <span className="rounded px-2 py-0.5 text-[13px] font-bold text-black"
          style={{ background: CONV_COLOR[conversation] }}>
          {conversation}
        </span>
        <span className="font-mono text-[15px]" style={{ color: ROLE_COLOR[role] }}>{role}</span>
        <span className="text-[15px] text-ink-secondary">“{text}”</span>
      </div>

      <div className="flex flex-col gap-1.5">
        {fields.map((field, i) => (
          <FieldRow key={i} field={field} />
        ))}
      </div>

      <div className="mt-3 flex items-baseline gap-2 border-t border-white/10 pt-2.5">
        <span className="text-[13px] font-bold tracking-widest text-ink-secondary">ID</span>
        <span className="font-mono text-[17px] font-bold" style={{ color: FRESH }}>{id}</span>
      </div>
    </div>
  );
}

function FieldRow({ field }: { field: HashField }): React.JSX.Element {
  const isParent = field.kind === "parent";
  const empty = field.bytes.length === 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-36 shrink-0 text-right font-mono text-[13px]"
        style={{ color: isParent ? FRESH : "var(--color-ink-quaternary)" }}>
        {field.label}
      </span>
      <span className="flex flex-wrap items-center gap-[2px]">
        {empty && (
          <span className="text-[13px] italic" style={{ color: FRESH }}>
            nothing — this is a root, and that alone makes it different
          </span>
        )}
        {field.bytes.slice(0, 32).map((b, i) => (
          <span key={i} className="h-4 w-[7px] rounded-[1px]"
            style={{
              background: isParent ? FRESH : "rgba(255,255,255,0.22)",
              opacity: isParent ? 0.55 + (b / 255) * 0.45 : 0.35 + (b / 255) * 0.5,
            }} />
        ))}
        {field.bytes.length > 32 && (
          <span className="ml-1 text-[12px] text-ink-quaternary">+{field.bytes.length - 32}</span>
        )}
      </span>
    </div>
  );
}

const NODE_W = 210;
const NODE_H = 46;
const COL = 250;
const ROW = 62;

/** The trie, laid out by depth. Depth is chain position, which is what parentage means here. */
function Trie({
  pool,
  lastHandle,
  lastDeduped,
}: {
  pool: Pool;
  lastHandle: number | null;
  lastDeduped: boolean;
}): React.JSX.Element {
  const depth = new Map<number, number>();
  for (const segment of pool.arena) {
    depth.set(segment.handle, segment.parent === null ? 0 : (depth.get(segment.parent) ?? 0) + 1);
  }
  const rows = new Map<number, number>();
  const perDepth = new Map<number, number>();
  for (const segment of pool.arena) {
    const d = depth.get(segment.handle) ?? 0;
    const r = perDepth.get(d) ?? 0;
    rows.set(segment.handle, r);
    perDepth.set(d, r + 1);
  }

  const maxDepth = Math.max(0, ...[...depth.values()]);
  const maxRow = Math.max(0, ...[...perDepth.values()]) ;
  const width = (maxDepth + 1) * COL + 20;
  const height = Math.max(3, maxRow) * ROW + 24;
  const x = (h: number) => (depth.get(h) ?? 0) * COL + 8;
  const y = (h: number) => (rows.get(h) ?? 0) * ROW + 10;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
      role="img" aria-label="segment trie, one node per stored segment">
      {pool.arena.map((segment) => {
        if (segment.parent === null) return null;
        const px = x(segment.parent) + NODE_W;
        const py = y(segment.parent) + NODE_H / 2;
        const cx = x(segment.handle);
        const cy = y(segment.handle) + NODE_H / 2;
        const mid = (px + cx) / 2;
        return (
          <path key={`e${segment.handle}`}
            d={`M ${px} ${py} C ${mid} ${py}, ${mid} ${cy}, ${cx} ${cy}`}
            fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={1.5} />
        );
      })}

      {pool.arena.map((segment) => (
        <TrieNode key={segment.handle} segment={segment} x={x(segment.handle)} y={y(segment.handle)}
          isLast={segment.handle === lastHandle} lastDeduped={lastDeduped}
          reused={childrenOf(pool, segment.handle).length > 1} />
      ))}
    </svg>
  );
}

function TrieNode({
  segment,
  x,
  y,
  isLast,
  lastDeduped,
  reused,
}: {
  segment: Segment;
  x: number;
  y: number;
  isLast: boolean;
  lastDeduped: boolean;
  reused: boolean;
}): React.JSX.Element {
  const stroke = isLast ? (lastDeduped ? DEDUP : FRESH) : "rgba(255,255,255,0.14)";
  return (
    <g>
      <rect x={x} y={y} width={NODE_W} height={NODE_H} rx={5}
        fill="rgba(255,255,255,0.035)" stroke={stroke} strokeWidth={isLast ? 2 : 1} />
      <rect x={x} y={y} width={4} height={NODE_H} rx={2} fill={ROLE_COLOR[segment.role]} />
      <text x={x + 12} y={y + 18} fontSize={13} fill="var(--color-ink-primary)">
        {segment.text.length > 26 ? `${segment.text.slice(0, 25)}…` : segment.text}
      </text>
      <text x={x + 12} y={y + 35} fontSize={12} fontFamily="var(--font-mono, monospace)"
        fill="var(--color-ink-quaternary)">
        #{segment.handle} · {segment.id.slice(0, 10)}
      </text>
      {reused && <text x={x + NODE_W - 12} y={y + 35} fontSize={13} textAnchor="end" fill={DEDUP}>◆</text>}
    </g>
  );
}

/** The dense arena. A dedup hit adds nothing here, which is the saving. */
function Arena({
  pool,
  lastHandle,
  lastDeduped,
}: {
  pool: Pool;
  lastHandle: number | null;
  lastDeduped: boolean;
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-1">
      {pool.arena.length === 0 && (
        <span className="text-[15px] text-ink-quaternary">Empty. Intern a turn.</span>
      )}
      {pool.arena.map((segment) => {
        const hit = segment.handle === lastHandle;
        return (
          <div key={segment.handle}
            className="flex items-center gap-3 rounded border px-2.5 py-1 font-mono text-[13px]"
            style={{
              borderColor: hit ? (lastDeduped ? DEDUP : FRESH) : "rgba(255,255,255,0.08)",
              background: hit ? "rgba(255,255,255,0.04)" : undefined,
            }}>
            <span className="w-8 text-ink-tertiary">#{segment.handle}</span>
            <span style={{ color: ROLE_COLOR[segment.role] }}>{segment.role}</span>
            <span className="text-ink-quaternary">{segment.id.slice(0, 12)}</span>
            <span className="ml-auto text-ink-quaternary">
              {segment.parent === null ? "root" : `child of #${segment.parent}`}
            </span>
            {hit && (
              <span className="rounded px-1.5 text-[12px] font-bold text-black"
                style={{ background: lastDeduped ? DEDUP : FRESH }}>
                {lastDeduped ? "DEDUP" : "NEW"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * The two identical turns, side by side, with their ids.
 *
 * Stated rather than animated, because it is the conclusion: everything above exists to make this
 * pair unsurprising.
 */
function Collision(): React.JSX.Element {
  const text = "Summarise the quarterly report";
  const tokens = tokenize(text);
  const systemId = segmentId(null, "system", tokenize("You are a careful assistant"));
  const withParent = segmentId(systemId, "user", tokens);
  const withoutParent = segmentId(null, "user", tokens);

  return (
    <div className="mt-4 rounded-lg border px-5 py-4"
      style={{ borderColor: FRESH, background: "rgba(255,255,0,0.03)" }}>
      <div className="mb-2 text-[12px] font-bold tracking-widest" style={{ color: FRESH }}>
        THE SAME TURN, TWICE
      </div>
      <p className="mb-3 max-w-5xl text-base leading-relaxed text-ink-secondary">
        Identical role. Identical text. Identical token IDs — all {tokens.length} of them. The only
        difference is what came before, and it is enough to make these two separate segments that
        will never share a slot, a cache entry, or a prefix.
      </p>
      <div className="grid grid-cols-2 gap-4">
        {[
          { label: "in conversation A — after the system prompt", id: withParent, parent: true },
          { label: "in conversation C — first turn, no parent", id: withoutParent, parent: false },
        ].map((row) => (
          <div key={row.label} className="rounded border border-white/10 px-3 py-2.5">
            <div className="mb-1 text-[14px] text-ink-tertiary">{row.label}</div>
            <div className="font-mono text-[15px] text-ink-secondary">user · “{text}”</div>
            <div className="mt-1.5 font-mono text-[17px] font-bold"
              style={{ color: row.parent ? CONV_COLOR.A : CONV_COLOR.C }}>
              {row.id}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
