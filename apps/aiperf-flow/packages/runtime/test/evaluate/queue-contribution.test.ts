// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  contributeQueue,
  type QueueContributionInput,
} from "../../src/evaluate/contributions/queue.js";

const input = {
  id: "scheduler",
  arrivals: [
    { id: "A", arriveAt: 0, serviceMs: 10 },
    { id: "B", arriveAt: 1, serviceMs: 5 },
  ],
  policy: "fifo",
  bounds: { x: 0, y: 0, width: 120, height: 24 },
  chipWidth: 20,
  padding: 4,
  gap: 4,
  order: 2,
} satisfies Omit<QueueContributionInput, "atMs">;

function evaluate(atMs: number) {
  return contributeQueue({ ...input, atMs });
}

describe("contributeQueue", () => {
  test.each([
    {
      atMs: 0,
      occupancy: {
        waiting: [],
        serving: "A",
        departed: [],
        rejected: [],
      },
    },
    {
      atMs: 5,
      occupancy: {
        waiting: ["B"],
        serving: "A",
        departed: [],
        rejected: [],
      },
    },
    {
      atMs: 10,
      occupancy: {
        waiting: [],
        serving: "B",
        departed: ["A"],
        rejected: [],
      },
    },
    {
      atMs: 15,
      occupancy: {
        waiting: [],
        serving: null,
        departed: ["A", "B"],
        rejected: [],
      },
    },
  ])("derives occupancy at the authored $atMs ms beat", ({ atMs, occupancy }) => {
    expect(evaluate(atMs).occupancy).toEqual(occupancy);
  });

  test("emits backend-neutral commands, semantics, and hit regions for visible requests", () => {
    const contribution = evaluate(5);

    expect(contribution.commands).toEqual([
      {
        kind: "path",
        id: "scheduler:lane",
        order: 2,
        paintBounds: input.bounds,
        damageBounds: input.bounds,
        path: "M 0 0 H 120 V 24 H 0 Z",
        fill: "#111827",
      },
      {
        kind: "path",
        id: "scheduler:request:B",
        order: 3,
        paintBounds: { x: 4, y: 4, width: 20, height: 16 },
        damageBounds: { x: 4, y: 4, width: 20, height: 16 },
        path: "M 4 4 H 24 V 20 H 4 Z",
        fill: "#64748b",
      },
      {
        kind: "path",
        id: "scheduler:request:A",
        order: 4,
        paintBounds: { x: 96, y: 4, width: 20, height: 16 },
        damageBounds: { x: 96, y: 4, width: 20, height: 16 },
        path: "M 96 4 H 116 V 20 H 96 Z",
        fill: "#22c55e",
      },
    ]);
    expect(contribution.semanticEntities).toEqual([
      {
        id: "scheduler:request:A",
        label: "A",
        role: "listitem",
        kind: "serving",
        description: "Queue request A is serving",
      },
      {
        id: "scheduler:request:B",
        label: "B",
        role: "listitem",
        kind: "waiting",
        description: "Queue request B is waiting",
      },
    ]);
    expect(contribution.hitRegions).toEqual([
      {
        id: "hit:scheduler:request:B",
        semanticId: "scheduler:request:B",
        order: 3,
        bounds: { x: 4, y: 4, width: 20, height: 16 },
      },
      {
        id: "hit:scheduler:request:A",
        semanticId: "scheduler:request:A",
        order: 4,
        bounds: { x: 96, y: 4, width: 20, height: 16 },
      },
    ]);
  });

  test("direct seek equals continuous evaluation and never reads wall time", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });
    let continuous = evaluate(0);
    for (let atMs = 1; atMs <= 10; atMs += 1) {
      continuous = evaluate(atMs);
    }

    expect(evaluate(10)).toEqual(continuous);
    expect(dateNow).not.toHaveBeenCalled();
  });

  test("returns deeply immutable products", () => {
    const contribution = evaluate(5);

    expect(Object.isFrozen(contribution)).toBe(true);
    expect(Object.isFrozen(contribution.occupancy)).toBe(true);
    expect(Object.isFrozen(contribution.occupancy.waiting)).toBe(true);
    expect(Object.isFrozen(contribution.commands)).toBe(true);
    expect(Object.isFrozen(contribution.commands[0])).toBe(true);
    expect(Object.isFrozen(contribution.commands[0]?.paintBounds)).toBe(true);
    expect(Object.isFrozen(contribution.semanticEntities)).toBe(true);
    expect(Object.isFrozen(contribution.hitRegions)).toBe(true);
  });

  test("does not freeze or mutate authored input", () => {
    const authoredBounds = { x: 0, y: 0, width: 120, height: 24 };
    const authored = { ...input, bounds: authoredBounds, atMs: 5 };

    contributeQueue(authored);

    expect(Object.isFrozen(authored)).toBe(false);
    expect(Object.isFrozen(authoredBounds)).toBe(false);
    expect(authoredBounds).toEqual({ x: 0, y: 0, width: 120, height: 24 });
  });

  test("rejects non-integer authored time", () => {
    expect(() => evaluate(1.5)).toThrow(
      "Queue evaluation time must be a non-negative safe integer",
    );
  });
});
