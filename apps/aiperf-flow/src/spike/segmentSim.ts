/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — a Dynamo trace being lowered into a segment pool, one message at a time.
//!
//! Models the real write side in `rust/runtime/src/dataset/segment.rs`:
//!
//! - `SegmentPool` is a dense append-only `arena: Vec<Segment>` plus `ids: HashMap<SegmentId,
//!   Handle>`, and a `Handle` is just the arena index.
//! - `intern` computes `payload_id(parent_id, payload)` and either returns the handle already
//!   registered for that id or appends a fresh segment.
//! - `payload_id` folds the *parent's* id into the hash alongside role, token IDs and the wire
//!   bytes, so identity is prefix-dependent: the same message under a different parent is a
//!   different segment. That is the whole point, and it is what makes a shared prefix collapse.
//!
//! The one deliberate departure: the real id is BLAKE3 over a version tag and a domain tag. Here
//! it is a short string hash, because the property being shown is prefix-dependence and dedup, not
//! the choice of digest. Everything structural is faithful.

export type Role = "system" | "user" | "assistant";

/** One message of a recorded conversation, as the trace supplies it. */
export type TraceMessage = {
  role: Role;
  text: string;
  /** Token count; the real identity folds in the token IDs themselves. */
  tokens: number;
  /** The serialized message. `intern_message` hashes and stores exactly these bytes. */
  wire: string;
};

/** The JSON one message serializes to, matching the dialect shape the real path interns. */
export function messageWire(role: Role, text: string): string {
  return JSON.stringify({ role, content: text });
}

/** One recorded session. Sessions that share an opening are what make the pool dedup. */
export type TraceSession = {
  id: string;
  messages: TraceMessage[];
};

/** An arena entry. `parent` is the prefix it was interned under. */
export type Segment = {
  handle: number;
  id: string;
  parent: number | null;
  role: Role;
  text: string;
  tokens: number;
  /** Wire bytes this segment stores. */
  bytes: number;
  /** How many interning attempts resolved to this handle, including the first. */
  refs: number;
};

/** One entry of `input_sequence_hashes`, and whether it was already known. */
export type BlockHash = {
  /** Rendered wide: real values routinely exceed u64::MAX. */
  value: string;
  index: number;
  /** True when this exact block was seen before, i.e. a KV prefix hit. */
  reused: boolean;
};

/** One `request_end` record, built up as its session's messages are interned. */
export type TraceTurn = {
  sessionId: string;
  requestId: string;
  eventTimeMs: number;
  inputLength: number;
  outputTokens: number;
  cachedTokens: number;
  ttftMs: number;
  totalTimeMs: number;
  hashes: BlockHash[];
};

/**
 * One worker materializing a request body from the frozen pool.
 *
 * Mirrors `build_body_from_handles` in `rust/runtime/src/dataset/materialize.rs`: walk a chain of
 * handles, clone each stored wire, then append a *pre-serialized* override tail. Nothing is
 * decoded and nothing is validated — the wires are well-formed by construction, so re-scanning
 * them per dispatch would be pure overhead.
 */
export type Worker = {
  id: number;
  sessionId: string;
  /** Handles to concatenate, in order. */
  chain: number[];
  /** How many have been pulled so far. */
  cursor: number;
  /** Bytes copied out of the arena so far. */
  bytes: number;
  /** Handle being read this instant, for lighting the arena cell. */
  reading: number | null;
  done: boolean;
};

export type InternEvent = {
  sessionId: string;
  messageIndex: number;
  handle: number;
  hit: boolean;
  id: string;
  parent: number | null;
};

export type SegmentSimState = {
  seed: number;
  /** Ticks elapsed. Narration drives by this, since it spans both phases. */
  ticks: number;
  pending: number;
  now: number;
  /** The dense arena. Index is the handle. */
  arena: Segment[];
  /** Content id to handle, the dedup map. */
  ids: Map<string, number>;
  sessions: TraceSession[];
  /** Cursor into the flattened (session, message) work list. */
  cursor: { session: number; message: number };
  /** Handle of the previous message in the session being walked; the prefix parent. */
  parent: number | null;
  events: InternEvent[];
  interned: number;
  hits: number;
  /** Wire bytes that would have been stored without dedup. */
  bytesNaive: number;
  bytesStored: number;
  /** `"session:message"` to the handle it resolved to, so the trace can be coloured by outcome. */
  resolved: Map<string, number>;
  /** The `request_end` record being built right now, growing block by block. */
  turn: TraceTurn | null;
  /** Completed records, newest last. */
  turns: TraceTurn[];
  /** Every block hash ever emitted, to classify a reuse. */
  blocksSeen: Set<string>;
  /** Running token total for the session being walked; drives `input_length`. */
  sessionTokens: number;
  /** `intern` fills the pool; `materialize` reads it back out. */
  phase: "intern" | "materialize";
  workers: Worker[];
  /** Sessions still waiting for a free worker. */
  queue: number[];
  bodiesBuilt: number;
  bytesMaterialized: number;
  done: boolean;
};

/** One message is interned per tick, so the arena fills at a watchable pace. */
export const TICK_MS = 260;

/**
 * Tokens per KV block: the trace's `request.replay.trace_block_size`.
 *
 * A `dynamo.request.trace.v1` record carries `input_sequence_hashes`, one hash per block of this
 * many tokens covering `input_length`. Consecutive turns in a session share a prefix of that list,
 * and that shared prefix is what `cached_tokens` counts.
 */
export const BLOCK_SIZE = 32;

function rand(seed: number, a: number, b: number): number {
  const x = Math.sin(a * 127.1 + b * 311.7 + seed * 51.17) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Short prefix-dependent content id.
 *
 * Mirrors `payload_id`'s inputs — parent id, role, tokens, wire — so that changing the parent
 * changes the id even when the message is byte-identical.
 */
export function segmentId(
  parent: string | null,
  role: Role,
  text: string,
  tokens: number,
): string {
  let h = 0x811c9dc5;
  const feed = (s: string) => {
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    h ^= 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  };
  feed(parent ?? "\0root");
  feed(role);
  feed(String(tokens));
  feed(text);
  return h.toString(16).padStart(8, "0").slice(0, 6);
}

/**
 * A stable colour per content id.
 *
 * Colouring by role gave three hues over dozens of cells, so two cells matching meant nothing.
 * Keyed on the id instead, matching colours mean matching segments — dedup becomes visible
 * directly, without reading a single number.
 */
export function colorForId(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (Math.imul(h, 31) + id.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 62% 58%)`;
}

const TOPICS = [
  "summarise the incident report",
  "find the failing test",
  "explain this stack trace",
  "draft the migration plan",
  "review the config diff",
  "trace the memory growth",
];

const SYSTEM = "You are a helpful engineering assistant with repository access.";

/**
 * A small recorded trace with the structure that makes prefix reuse worth having: every session
 * opens with the same system prompt, and several sessions continue an earlier conversation before
 * diverging.
 */
export function buildSessions(seed: number, count: number): TraceSession[] {
  const sessions: TraceSession[] = [];
  for (let s = 0; s < count; s++) {
    const messages: TraceMessage[] = [
      { role: "system", text: SYSTEM, tokens: 46, wire: messageWire("system", SYSTEM) },
    ];
    // Continuing an earlier session replays its turns verbatim before adding new ones — which is
    // exactly the case a prefix-keyed pool collapses to nothing.
    const continues = s > 1 && rand(seed, s, 1) < 0.55 ? Math.floor(rand(seed, s, 2) * s) : -1;
    if (continues >= 0) {
      const base = sessions[continues]!;
      const keep = 1 + Math.floor(rand(seed, s, 3) * Math.max(1, base.messages.length - 1));
      for (const m of base.messages.slice(1, keep)) messages.push({ ...m });
    }
    const turns = 1 + Math.floor(rand(seed, s, 4) * 3);
    for (let t = 0; t < turns; t++) {
      const topic = TOPICS[Math.floor(rand(seed, s * 7 + t, 5) * TOPICS.length)]!;
      messages.push({
        role: "user",
        text: topic,
        tokens: 40 + Math.floor(rand(seed, s * 7 + t, 6) * 90),
        wire: messageWire("user", topic),
      });
      const reply = `re: ${topic}`;
      messages.push({
        role: "assistant",
        text: reply,
        tokens: 90 + Math.floor(rand(seed, s * 7 + t, 7) * 240),
        wire: messageWire("assistant", reply),
      });
    }
    sessions.push({ id: `s${s}`, messages });
  }
  return sessions;
}

export function createSegmentSim(seed = 1, sessionCount = 8): SegmentSimState {
  return {
    seed,
    ticks: 0,
    pending: 0,
    now: 0,
    arena: [],
    ids: new Map(),
    sessions: buildSessions(seed, sessionCount),
    cursor: { session: 0, message: 0 },
    parent: null,
    events: [],
    interned: 0,
    hits: 0,
    bytesNaive: 0,
    bytesStored: 0,
    resolved: new Map(),
    turn: null,
    turns: [],
    blocksSeen: new Set(),
    sessionTokens: 0,
    phase: "intern",
    workers: [],
    queue: [],
    bodiesBuilt: 0,
    bytesMaterialized: 0,
    done: false,
  };
}

/**
 * Block hash covering the first `(index + 1) * BLOCK_SIZE` tokens of a prefix.
 *
 * Keyed on the segment id at that point, so two sessions whose conversations agree up to a block
 * boundary emit byte-identical hashes there. That is exactly the property a KV cache exploits, and
 * the reason the trace ships these at all.
 */
export function blockHash(prefixId: string, index: number): string {
  let h = 0x9e3779b9;
  const feed = (str: string) => {
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x85ebca6b) >>> 0;
    }
  };
  feed(prefixId);
  feed(`#${index}`);
  return `${h}${(h * 2654435761) % 100000}`.slice(0, 15);
}

/** Wire size a message occupies: the serialized bytes themselves. */
function wireBytes(m: TraceMessage): number {
  return m.wire.length;
}

/** Intern exactly one message: the arena-append-or-dedup decision, drawn one step at a time. */
export function tick(input: SegmentSimState): SegmentSimState {
  const state = { ...input, ticks: input.ticks + 1 };
  if (state.phase === "materialize") return tickMaterialize(state);
  if (state.done) return state;
  const session = state.sessions[state.cursor.session];
  if (session === undefined) return beginMaterialize(state);

  const message = session.messages[state.cursor.message];
  if (message === undefined) {
    // Session finished: the next one starts a fresh chain with no prefix parent, and its
    // `input_length` restarts from zero.
    const nextSession = state.cursor.session + 1;
    const next: SegmentSimState = {
      ...state,
      cursor: { session: nextSession, message: 0 },
      parent: null,
      turn: null,
      turns: state.turn === null ? state.turns : [...state.turns, state.turn],
      sessionTokens: 0,
    };
    return next.cursor.session >= state.sessions.length ? beginMaterialize(next) : next;
  }

  const parentId = state.parent === null ? null : state.arena[state.parent]!.id;
  const id = segmentId(parentId, message.role, message.wire, message.tokens);
  const bytes = wireBytes(message);

  const existing = state.ids.get(id);
  const arena = state.arena.slice();
  const ids = new Map(state.ids);
  let handle: number;
  let hit: boolean;

  if (existing !== undefined) {
    // Dedup: the id is already registered, so the handle is reused and nothing is appended.
    handle = existing;
    hit = true;
    arena[handle] = { ...arena[handle]!, refs: arena[handle]!.refs + 1 };
  } else {
    handle = arena.length;
    hit = false;
    arena.push({
      handle,
      id,
      parent: state.parent,
      role: message.role,
      text: message.text,
      tokens: message.tokens,
      bytes,
      refs: 1,
    });
    ids.set(id, handle);
  }

  const events = [
    ...state.events,
    { sessionId: session.id, messageIndex: state.cursor.message, handle, hit, id, parent: state.parent },
  ].slice(-40);

  const resolved = new Map(state.resolved);
  resolved.set(`${state.cursor.session}:${state.cursor.message}`, handle);

  // The trace record grows with the conversation: `input_length` accumulates, and a new entry
  // joins `input_sequence_hashes` each time the running token count crosses a block boundary.
  const sessionTokens = state.sessionTokens + message.tokens;
  const existingHashes = state.turn?.hashes ?? [];
  const wanted = Math.floor(sessionTokens / BLOCK_SIZE);
  const blocksSeen = new Set(state.blocksSeen);
  const hashes = existingHashes.slice();
  for (let b = hashes.length; b < wanted; b++) {
    const value = blockHash(id, b);
    // A block already emitted anywhere is a cache hit; the first sighting is a miss.
    const reused = blocksSeen.has(value);
    if (!reused) blocksSeen.add(value);
    hashes.push({ value, index: b, reused });
  }

  const cachedTokens = hashes.filter((h) => h.reused).length * BLOCK_SIZE;
  const turn: TraceTurn = {
    sessionId: session.id,
    requestId: `${session.id}-r${state.cursor.message}`,
    eventTimeMs: 1735689600000 + state.now,
    inputLength: sessionTokens,
    outputTokens: message.role === "assistant" ? message.tokens : 0,
    cachedTokens,
    ttftMs: 180 + (hashes.length - cachedTokens / BLOCK_SIZE) * 11,
    totalTimeMs: 400 + message.tokens * 9,
    hashes,
  };

  return {
    ...state,
    turn,
    sessionTokens,
    blocksSeen,
    arena,
    ids,
    resolved,
    events,
    parent: handle,
    cursor: { ...state.cursor, message: state.cursor.message + 1 },
    interned: state.interned + 1,
    hits: state.hits + (hit ? 1 : 0),
    bytesNaive: state.bytesNaive + bytes,
    bytesStored: state.bytesStored + (hit ? 0 : bytes),
  };
}

/** Advance by `dtMs` in whole ticks, carrying the remainder so frame cadence cannot change it. */
export function stepSegments(state: SegmentSimState, dtMs: number): SegmentSimState {
  let acc = state.pending + Math.max(0, dtMs);
  let out = state;
  let budget = 500;
  while (acc >= TICK_MS && budget-- > 0) {
    out = tick({ ...out, now: out.now + TICK_MS, pending: 0 });
    acc -= TICK_MS;
  }
  return { ...out, pending: acc };
}

/** How many handles a worker pulls per tick. One, so the concatenation is followable. */
const PULLS_PER_TICK = 1;
/** Workers reading the pool at once. */
const WORKER_COUNT = 3;

/**
 * Freeze the pool and hand it to workers.
 *
 * Every session becomes one body to build: its chain of handles, in message order. Note that the
 * chains overlap heavily — that is the point of having interned them — so several workers read the
 * very same arena entries without any of them owning a copy.
 */
function beginMaterialize(state: SegmentSimState): SegmentSimState {
  const queue = state.sessions.map((_, i) => i);
  const workers: Worker[] = [];
  for (let i = 0; i < WORKER_COUNT && queue.length > 0; i++) {
    workers.push(spawnWorker(state, i, queue.shift()!));
  }
  // Keep the last record on screen. The session-rollover path already cleared `turn`, and
  // blanking the panel mid-explanation loses the context the materialization narration refers to.
  const turn = state.turn ?? state.turns[state.turns.length - 1] ?? null;
  return { ...state, phase: "materialize", workers, queue, turn };
}

function chainFor(state: SegmentSimState, sessionIndex: number): number[] {
  const session = state.sessions[sessionIndex];
  if (session === undefined) return [];
  const chain: number[] = [];
  for (let m = 0; m < session.messages.length; m++) {
    const handle = state.resolved.get(`${sessionIndex}:${m}`);
    if (handle !== undefined) chain.push(handle);
  }
  return chain;
}

function spawnWorker(state: SegmentSimState, id: number, sessionIndex: number): Worker {
  return {
    id,
    sessionId: state.sessions[sessionIndex]?.id ?? `s${sessionIndex}`,
    chain: chainFor(state, sessionIndex),
    cursor: 0,
    bytes: 0,
    reading: null,
    done: false,
  };
}

/** One step of the read side: each worker pulls its next handle's wire out of the arena. */
function tickMaterialize(state: SegmentSimState): SegmentSimState {
  const queue = [...state.queue];
  let bodiesBuilt = state.bodiesBuilt;
  let bytesMaterialized = state.bytesMaterialized;

  const workers = state.workers.map((w) => {
    if (w.done) return w;
    if (w.cursor >= w.chain.length) {
      // Body complete. The override tail is appended pre-serialized, never rebuilt per request.
      bodiesBuilt += 1;
      if (queue.length > 0) return spawnWorker(state, w.id, queue.shift()!);
      return { ...w, reading: null, done: true };
    }
    const handle = w.chain[w.cursor]!;
    const bytes = state.arena[handle]?.bytes ?? 0;
    bytesMaterialized += bytes;
    return {
      ...w,
      cursor: w.cursor + PULLS_PER_TICK,
      bytes: w.bytes + bytes,
      reading: handle,
    };
  });

  return {
    ...state,
    workers,
    queue,
    bodiesBuilt,
    bytesMaterialized,
    done: workers.every((w) => w.done),
  };
}

/** Total messages the trace will intern, for mapping a narration fraction onto progress. */
export function totalMessages(state: SegmentSimState): number {
  return state.sessions.reduce((n, s) => n + s.messages.length, 0);
}

/**
 * Intern forward until `target` messages have been consumed.
 *
 * Lets narration drive the pool by message count rather than elapsed time, so the picture is
 * exactly where the voice says it is. Session boundaries consume a tick without interning, which
 * is why this counts `interned` rather than ticks.
 */
export function internUpTo(state: SegmentSimState, target: number): SegmentSimState {
  let out = state;
  let guard = 4000;
  while (out.interned < target && !out.done && guard-- > 0) {
    out = tick({ ...out, now: out.now + TICK_MS });
  }
  return out;
}

/** Advance to an exact tick, so narration can drive interning and materialization alike. */
export function advanceToTick(state: SegmentSimState, target: number): SegmentSimState {
  let out = state;
  let guard = 6000;
  while (out.ticks < target && !out.done && guard-- > 0) {
    out = tick({ ...out, now: out.now + TICK_MS });
  }
  return out;
}

/** Ticks a whole run takes, by running it headlessly. Deterministic, so it is a fixed number. */
export function totalTicks(seed: number, sessionCount: number): number {
  let s = createSegmentSim(seed, sessionCount);
  let guard = 6000;
  while (!s.done && guard-- > 0) s = tick({ ...s, now: s.now + TICK_MS });
  return s.ticks;
}

/** Chain of handles from a segment back to its root, nearest first. */
export function prefixChain(arena: readonly Segment[], handle: number | null): number[] {
  const chain: number[] = [];
  let at = handle;
  let guard = 64;
  while (at !== null && guard-- > 0) {
    chain.push(at);
    at = arena[at]?.parent ?? null;
  }
  return chain;
}
