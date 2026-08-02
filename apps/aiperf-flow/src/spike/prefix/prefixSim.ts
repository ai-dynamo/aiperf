/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — prefix-dependent segment identity.
//!
//! Modelled on `rust/runtime/src/dataset/segment.rs`:
//!
//! - `text_payload_id` (:620) hashes, in order: `HASH_VERSION`, the domain tag `text-only\0`, the
//!   parent's identity, the role, a NUL, then every authoritative token ID as little-endian u32.
//! - `hash_parent` (:632) writes the parent's 32 bytes when there is one and always writes a
//!   trailing NUL, so a root and a child are distinguishable even before content is folded in.
//! - `push_interned` (:339) looks the finished id up in `ids` and returns the existing handle on a
//!   hit; otherwise it appends a fresh dense arena slot.
//!
//! The consequence, which the pool's own test `handles_are_dense_and_dedup_by_prefix_dependent_id`
//! (:640) pins: the same role and the same tokens under a *different* parent produce a different
//! id and a second arena slot. Identity is not a property of the bytes. It is a property of the
//! bytes *and the path taken to reach them*.
//!
//! The digest here is FNV-1a rather than BLAKE3 — a 64-bit non-cryptographic hash is enough to
//! demonstrate keying and keeps the page dependency-free. What is faithful is the byte stream fed
//! into it and the order of the fields, which is the thing the page is about.

const HASH_VERSION = "aiperf-dataset-segment-v1\0";
const DOMAIN_TEXT = "text-only\0";

export type Role = "system" | "user" | "assistant";

/** One turn of a conversation, before it becomes a segment. */
export type Turn = { role: Role; text: string };

export type Conversation = { id: string; label: string; turns: Turn[] };

/** A run of bytes in the hash input, labelled with the field it came from. */
export type HashField = {
  label: string;
  bytes: number[];
  kind: "version" | "domain" | "parent" | "role" | "sep" | "tokens";
};

/** A segment in the pool's dense arena. */
export type Segment = {
  handle: number;
  id: string;
  parent: number | null;
  role: Role;
  text: string;
  tokens: number[];
};

/** What happened when a turn was interned. */
export type InternEvent = {
  conversation: string;
  turnIndex: number;
  role: Role;
  text: string;
  id: string;
  handle: number;
  /** True when the id was already present and no arena slot was allocated. */
  deduped: boolean;
};

export type Pool = {
  arena: Segment[];
  /** id → handle. The dedup index, and the reason identity has to be exact. */
  ids: Map<string, number>;
  log: InternEvent[];
};

/** FNV-1a over 64 bits, rendered as 16 hex characters. */
function fnv1a(bytes: readonly number[]): string {
  let hash = 0xcbf2_9ce4_8422_2325n;
  const prime = 0x1000_0000_01b3n;
  const mask = 0xffff_ffff_ffff_ffffn;
  for (const b of bytes) {
    hash = (hash ^ BigInt(b & 0xff)) * prime & mask;
  }
  return hash.toString(16).padStart(16, "0");
}

function ascii(text: string): number[] {
  return [...text].map((c) => c.codePointAt(0) ?? 0);
}

/** Hex string back to bytes, for folding a parent id into a child's input. */
function hexBytes(hex: string): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < hex.length; i += 2) out.push(Number.parseInt(hex.slice(i, i + 2), 16));
  return out;
}

/**
 * Deterministic stand-in for the tokenizer.
 *
 * Identical text must produce identical token IDs, because that is what makes two turns with the
 * same content collide in the first place. Nothing else about it needs to be realistic.
 */
export function tokenize(text: string): number[] {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .map((word) => {
      let h = 2166136261;
      for (const ch of word) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
      return h % 50_000;
    });
}

/**
 * The exact field sequence `text_payload_id` hashes, in order.
 *
 * Returned as labelled runs so the page can show that the parent's identity is *inside the input*
 * rather than merely associated with the result. That placement is the whole mechanism.
 */
export function hashInput(parentId: string | null, role: Role, tokens: readonly number[]): HashField[] {
  const tokenBytes: number[] = [];
  for (const t of tokens) {
    // Little-endian u32, matching `token.to_le_bytes()`.
    tokenBytes.push(t & 0xff, (t >>> 8) & 0xff, (t >>> 16) & 0xff, (t >>> 24) & 0xff);
  }
  return [
    { label: "HASH_VERSION", bytes: ascii(HASH_VERSION), kind: "version" },
    { label: '"text-only\\0"', bytes: ascii(DOMAIN_TEXT), kind: "domain" },
    {
      label: parentId === null ? "no parent" : "parent id",
      bytes: parentId === null ? [] : hexBytes(parentId),
      kind: "parent",
    },
    { label: "\\0", bytes: [0], kind: "sep" },
    { label: `"${role}"`, bytes: ascii(role), kind: "role" },
    { label: "\\0", bytes: [0], kind: "sep" },
    { label: `${tokens.length} tokens, LE u32`, bytes: tokenBytes, kind: "tokens" },
  ];
}

/** The content identity of a text segment under an optional prefix parent. */
export function segmentId(parentId: string | null, role: Role, tokens: readonly number[]): string {
  return fnv1a(hashInput(parentId, role, tokens).flatMap((f) => f.bytes));
}

export function createPool(): Pool {
  return { arena: [], ids: new Map(), log: [] };
}

/**
 * Intern one turn under an optional parent handle.
 *
 * Mirrors `intern_text` into `push_interned`: compute the identity, return the existing handle on
 * a hit, otherwise append a dense slot. The pool is mutated in place; callers clone when they need
 * a snapshot for rendering.
 */
export function intern(
  pool: Pool,
  parent: number | null,
  turn: Turn,
  provenance: { conversation: string; turnIndex: number },
): number {
  const parentId = parent === null ? null : (pool.arena[parent]?.id ?? null);
  const tokens = tokenize(turn.text);
  const id = segmentId(parentId, turn.role, tokens);

  const existing = pool.ids.get(id);
  if (existing !== undefined) {
    pool.log.push({
      conversation: provenance.conversation,
      turnIndex: provenance.turnIndex,
      role: turn.role,
      text: turn.text,
      id,
      handle: existing,
      deduped: true,
    });
    return existing;
  }

  const handle = pool.arena.length;
  pool.arena.push({ handle, id, parent, role: turn.role, text: turn.text, tokens });
  pool.ids.set(id, handle);
  pool.log.push({
    conversation: provenance.conversation,
    turnIndex: provenance.turnIndex,
    role: turn.role,
    text: turn.text,
    id,
    handle,
    deduped: false,
  });
  return handle;
}

/** Intern a whole conversation as a chain, each turn parented on the one before it. */
export function internConversation(pool: Pool, conversation: Conversation): number[] {
  let parent: number | null = null;
  const handles: number[] = [];
  conversation.turns.forEach((turn, turnIndex) => {
    parent = intern(pool, parent, turn, { conversation: conversation.id, turnIndex });
    handles.push(parent);
  });
  return handles;
}

/**
 * Three conversations that share a system prompt and then diverge.
 *
 * B repeats A's first user turn verbatim, so its first two segments are dedup hits. C opens with
 * the same user turn as A but *without* the shared system prompt, which is the case the page is
 * built around: identical role, identical tokens, different parent, different identity.
 */
export function defaultConversations(): Conversation[] {
  const system: Turn = { role: "system", text: "You are a careful assistant" };
  return [
    {
      id: "A",
      label: "conversation A",
      turns: [
        system,
        { role: "user", text: "Summarise the quarterly report" },
        { role: "assistant", text: "Revenue rose and costs held flat" },
        { role: "user", text: "Now compare it with last year" },
      ],
    },
    {
      id: "B",
      label: "conversation B",
      turns: [
        system,
        { role: "user", text: "Summarise the quarterly report" },
        { role: "assistant", text: "Revenue rose and costs held flat" },
        { role: "user", text: "Break that down by region" },
      ],
    },
    {
      id: "C",
      label: "conversation C — no system prompt",
      turns: [
        { role: "user", text: "Summarise the quarterly report" },
        { role: "assistant", text: "Revenue rose and costs held flat" },
      ],
    },
  ];
}

/** Children of a segment, in arena order — the trie's edges, derived rather than stored. */
export function childrenOf(pool: Pool, handle: number | null): Segment[] {
  return pool.arena.filter((s) => s.parent === handle);
}

/** How many turns were interned versus how many arena slots that required. */
export function poolStats(pool: Pool): { interned: number; stored: number; deduped: number } {
  const deduped = pool.log.filter((e) => e.deduped).length;
  return { interned: pool.log.length, stored: pool.arena.length, deduped };
}
