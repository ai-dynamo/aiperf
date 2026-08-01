/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  createSegmentSim,
  stepSegments,
  segmentId,
  prefixChain,
  TICK_MS,
  type SegmentSimState,
} from "./segmentSim.js";

function runToEnd(seed: number, chunk = TICK_MS): SegmentSimState {
  let s = createSegmentSim(seed, 9);
  for (let i = 0; i < 4000 && !s.done; i++) s = stepSegments(s, chunk);
  return s;
}

describe("segmentId", () => {
  it("gives the same id for the same content under the same parent", () => {
    expect(segmentId("aaa", "user", "hello", 3)).toBe(segmentId("aaa", "user", "hello", 3));
  });

  it("gives a different id for identical content under a different parent", () => {
    // The property the whole design turns on: identity is prefix-dependent, so a message that
    // continues one conversation is not the same segment as the same message continuing another.
    expect(segmentId("aaa", "user", "hello", 3)).not.toBe(segmentId("bbb", "user", "hello", 3));
    expect(segmentId(null, "user", "hello", 3)).not.toBe(segmentId("aaa", "user", "hello", 3));
  });

  it("separates role and token count from text", () => {
    expect(segmentId(null, "user", "hi", 3)).not.toBe(segmentId(null, "assistant", "hi", 3));
    expect(segmentId(null, "user", "hi", 3)).not.toBe(segmentId(null, "user", "hi", 4));
  });
});

describe("the pool's invariants", () => {
  it("assigns handles densely, in append order, never reusing one", () => {
    const s = runToEnd(1);
    expect(s.arena.length).toBeGreaterThan(0);
    s.arena.forEach((seg, i) => expect(seg.handle).toBe(i));
  });

  it("keeps exactly one arena entry per distinct id", () => {
    const s = runToEnd(1);
    expect(s.ids.size).toBe(s.arena.length);
    expect(new Set(s.arena.map((x) => x.id)).size).toBe(s.arena.length);
  });

  it("appends on a miss and appends nothing on a hit", () => {
    // interned = hits + appends, which is the arena length.
    const s = runToEnd(2);
    expect(s.interned - s.hits).toBe(s.arena.length);
  });

  it("stores wire bytes only for the segments it actually appended", () => {
    const s = runToEnd(2);
    expect(s.bytesStored).toBe(s.arena.reduce((n, x) => n + x.bytes, 0));
    expect(s.bytesStored).toBeLessThan(s.bytesNaive);
  });

  it("dedups a shared prefix, so a continued session is mostly free", () => {
    // Every session opens with the same system prompt and several continue an earlier one, so a
    // meaningful fraction of interning must resolve to existing handles.
    const s = runToEnd(1);
    expect(s.hits / s.interned).toBeGreaterThan(0.15);
  });

  it("points every segment's parent at an earlier handle", () => {
    // Prefix parents are interned before their children, so the arena is topologically ordered.
    const s = runToEnd(3);
    for (const seg of s.arena) {
      if (seg.parent === null) continue;
      expect(seg.parent).toBeLessThan(seg.handle);
    }
  });
});

describe("determinism", () => {
  it("is independent of how elapsed time was chopped into frames", () => {
    const a = runToEnd(5, TICK_MS);
    const b = runToEnd(5, TICK_MS * 7);
    expect(a.arena.map((x) => `${x.handle}:${x.id}:${x.refs}`))
      .toEqual(b.arena.map((x) => `${x.handle}:${x.id}:${x.refs}`));
  });

  it("gives a different trace for a different seed", () => {
    expect(runToEnd(1).arena.map((x) => x.id)).not.toEqual(runToEnd(2).arena.map((x) => x.id));
  });
});

describe("prefixChain", () => {
  it("walks back to the root, nearest first", () => {
    const s = runToEnd(1);
    const leaf = s.arena[s.arena.length - 1]!;
    const chain = prefixChain(s.arena, leaf.handle);
    expect(chain[0]).toBe(leaf.handle);
    expect(s.arena[chain[chain.length - 1]!]!.parent).toBeNull();
  });

  it("returns nothing for no handle", () => {
    expect(prefixChain([], null)).toEqual([]);
  });
});

describe("materialize", () => {
  it("hands over to workers once every message is interned", () => {
    let s = createSegmentSim(1, 9);
    for (let i = 0; i < 4000 && s.phase === "intern"; i++) s = stepSegments(s, TICK_MS);
    expect(s.phase).toBe("materialize");
    // The pool is frozen at handover: reading it back must not add segments.
    const frozen = s.arena.length;
    const end = runToEnd(1);
    expect(end.arena.length).toBe(frozen);
  });

  it("builds one body per session and finishes", () => {
    const s = runToEnd(1);
    expect(s.done).toBe(true);
    expect(s.bodiesBuilt).toBe(s.sessions.length);
  });

  it("copies out more bytes than it stores, because chains overlap", () => {
    // Several sessions share prefix handles, so the same arena entry is read by more than one
    // worker. That is the point of interning: one stored copy, many readers.
    const s = runToEnd(1);
    expect(s.bytesMaterialized).toBeGreaterThan(s.bytesStored);
  });

  it("gives every worker a chain of real handles in message order", () => {
    let s = createSegmentSim(2, 9);
    for (let i = 0; i < 4000 && s.phase === "intern"; i++) s = stepSegments(s, TICK_MS);
    for (const w of s.workers) {
      expect(w.chain.length).toBeGreaterThan(0);
      for (const h of w.chain) expect(s.arena[h]).toBeDefined();
    }
  });
});
