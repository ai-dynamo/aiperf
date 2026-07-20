/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { simulateQueue } from "./queue-policy.js";

describe("simulateQueue", () => {
  it("starts queued service when the server frees, not at a later arrival", () => {
    // A occupies 0–100; B waits from 10; C arrives at 200 after B should already be done.
    const { events } = simulateQueue(
      [
        { id: "A", arriveAt: 0, serviceMs: 100 },
        { id: "B", arriveAt: 10, serviceMs: 10 },
        { id: "C", arriveAt: 200, serviceMs: 10 },
      ],
      "fifo",
    );

    const startById = Object.fromEntries(
      events.filter((e) => e.kind === "start-service").map((e) => [e.requestId, e.at]),
    );
    const departById = Object.fromEntries(
      events.filter((e) => e.kind === "depart").map((e) => [e.requestId, e.at]),
    );

    expect(startById.A).toBe(0);
    expect(departById.A).toBe(100);
    // B must start when A frees (100), not when C arrives (200).
    expect(startById.B).toBe(100);
    expect(departById.B).toBe(110);
    expect(startById.C).toBe(200);
    expect(departById.C).toBe(210);
  });

  it("pulls the next waiter when a capacity reject coincides with server free", () => {
    // capacity=1: A serves 0–100; B waits; C rejected at t=100 (server free) must
    // still start B immediately. A later D arrival must not steal B's start time.
    const { events } = simulateQueue(
      [
        { id: "A", arriveAt: 0, serviceMs: 100 },
        { id: "B", arriveAt: 10, serviceMs: 50 },
        { id: "C", arriveAt: 100, serviceMs: 10 },
        { id: "D", arriveAt: 120, serviceMs: 10 },
      ],
      "fifo",
      1,
    );

    const kinds = events.map((e) => `${e.requestId}:${e.kind}@${e.at}`);
    expect(kinds).toContain("C:reject@100");

    const startById = Object.fromEntries(
      events.filter((e) => e.kind === "start-service").map((e) => [e.requestId, e.at]),
    );
    const departById = Object.fromEntries(
      events.filter((e) => e.kind === "depart").map((e) => [e.requestId, e.at]),
    );

    // Without in-loop pull on reject, D's arrival would start B at 120 instead of 100.
    expect(startById.B).toBe(100);
    expect(departById.B).toBe(150);
    expect(startById.D).toBe(150);
    expect(departById.D).toBe(160);
  });
});
