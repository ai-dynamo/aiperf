/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  childrenOf,
  createPool,
  defaultConversations,
  hashInput,
  intern,
  internConversation,
  poolStats,
  segmentId,
  tokenize,
  type Turn,
} from "./prefixSim.js";

const user = (text: string): Turn => ({ role: "user", text });
const at = (conversation: string, turnIndex: number) => ({ conversation, turnIndex });

describe("the property the pool is built on", () => {
  it("gives the same content a different identity under a different parent", () => {
    // This is `handles_are_dense_and_dedup_by_prefix_dependent_id` in segment.rs, restated: the
    // same role and the same tokens are a *different segment* when the prefix differs.
    const pool = createPool();
    const root = intern(pool, null, { role: "system", text: "system" }, at("A", 0));
    const child = intern(pool, root, user("hello"), at("A", 1));
    const orphan = intern(pool, null, user("hello"), at("B", 0));

    expect(pool.arena[child]!.id).not.toBe(pool.arena[orphan]!.id);
    expect(orphan).not.toBe(child);
    expect(pool.arena).toHaveLength(3);
  });

  it("returns the existing handle when content and parent both match", () => {
    const pool = createPool();
    const root = intern(pool, null, { role: "system", text: "system" }, at("A", 0));
    const first = intern(pool, root, user("hello"), at("A", 1));
    const again = intern(pool, root, user("hello"), at("B", 1));

    expect(again).toBe(first);
    expect(pool.arena).toHaveLength(2);
    expect(pool.log.at(-1)?.deduped).toBe(true);
  });

  it("folds the parent's identity into the child's hash input", () => {
    // Not merely associated with the result — inside the bytes being hashed. Everything else in
    // the input is held fixed here, so the parent field is the only thing that can differ.
    const rooted = hashInput("00ff00ff00ff00ff", "user", [1, 2]);
    const orphaned = hashInput(null, "user", [1, 2]);
    const parentField = (fields: typeof rooted) => fields.find((f) => f.kind === "parent")!;

    expect(parentField(rooted).bytes).toHaveLength(8);
    expect(parentField(orphaned).bytes).toHaveLength(0);
    expect(rooted.filter((f) => f.kind !== "parent")).toEqual(
      orphaned.filter((f) => f.kind !== "parent"),
    );
  });

  it("keys on the version and domain tag before anything else", () => {
    // The version prefix is what lets the identity scheme change without silently colliding with
    // ids minted by an earlier one; the domain tag keeps text, raw, and media disjoint.
    const [version, domain] = hashInput(null, "user", [1]);
    expect(version?.kind).toBe("version");
    expect(domain?.kind).toBe("domain");
  });

  it("distinguishes roles that carry identical text", () => {
    expect(segmentId(null, "user", tokenize("hello"))).not.toBe(
      segmentId(null, "assistant", tokenize("hello")),
    );
  });

  it("hashes token ids little-endian, four bytes each", () => {
    const tokens = hashInput(null, "user", [1, 0x0102_0304]);
    const field = tokens.find((f) => f.kind === "tokens")!;
    expect(field.bytes).toEqual([1, 0, 0, 0, 0x04, 0x03, 0x02, 0x01]);
  });
});

describe("interning conversations", () => {
  it("shares a prefix instead of copying it", () => {
    const pool = createPool();
    const [a, b] = defaultConversations();
    const handlesA = internConversation(pool, a!);
    const handlesB = internConversation(pool, b!);

    // A and B open with the same system prompt and the same first exchange, so those three
    // segments are one set of slots, and only the diverging turn is new.
    expect(handlesB.slice(0, 3)).toEqual(handlesA.slice(0, 3));
    expect(handlesB[3]).not.toBe(handlesA[3]);
  });

  it("stores fewer segments than it interns", () => {
    const pool = createPool();
    for (const c of defaultConversations()) internConversation(pool, c);
    const { interned, stored, deduped } = poolStats(pool);
    expect(stored).toBeLessThan(interned);
    expect(stored + deduped).toBe(interned);
  });

  it("forks the trie where the conversations diverge", () => {
    const pool = createPool();
    const [a, b] = defaultConversations();
    internConversation(pool, a!);
    internConversation(pool, b!);
    // The shared assistant turn is the fork point: one node, two children.
    const forks = pool.arena.filter((s) => childrenOf(pool, s.handle).length > 1);
    expect(forks).toHaveLength(1);
    expect(forks[0]?.role).toBe("assistant");
  });

  it("never re-converges once the prefix differs", () => {
    // C repeats A's first two turns verbatim but without the system prompt. Every one of its
    // segments is fresh, because each inherits an already-different parent.
    const pool = createPool();
    const [a, , c] = defaultConversations();
    internConversation(pool, a!);
    const before = pool.arena.length;
    const handles = internConversation(pool, c!);
    expect(pool.arena.length).toBe(before + c!.turns.length);
    expect(handles.every((h) => h >= before)).toBe(true);
  });
});
