// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { simulateQueue } from "../../src/leaves/queue-policy.js";

describe("leaf.viz.queue.policy", () => {
  test("FIFO serves arrivals in order on a single server", () => {
    const simulation = simulateQueue(
      [
        { id: "A", arriveAt: 0, serviceMs: 10 },
        { id: "B", arriveAt: 1, serviceMs: 5 },
      ],
      "fifo",
    );

    expect(
      simulation.events.map(({ requestId, kind, at }) => ({ requestId, kind, at })),
    ).toEqual([
      { requestId: "A", kind: "enqueue", at: 0 },
      { requestId: "A", kind: "start-service", at: 0 },
      { requestId: "A", kind: "depart", at: 10 },
      { requestId: "B", kind: "enqueue", at: 1 },
      { requestId: "B", kind: "start-service", at: 10 },
      { requestId: "B", kind: "depart", at: 15 },
    ]);
  });
});
