/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  abortAll,
  beginCheck,
  cancel,
  createParkState,
  evaluate,
  isQuiescent,
  publish,
  runChecks,
  setThreading,
} from "./parkSim.js";

const reader = (s: ReturnType<typeof createParkState>, id: string) =>
  s.readers.find((r) => r.id === id)!;

describe("the three synchronous exits", () => {
  it("proceeds once arrivals reach the target", () => {
    let s = runChecks(createParkState(3, [2]));
    expect(reader(s, "r0").state).toBe("parked");
    s = publish(s, "p0");
    expect(reader(s, "r0").state).toBe("parked");
    s = publish(s, "p1");
    expect(reader(s, "r0").state).toBe("satisfied");
  });

  it("orphans only the reader whose count became unreachable", () => {
    // Two readers on one channel: three producers, targets 3 and 2. Cancel one producer and the
    // three-target reader can never be met, while the two-target reader is still fine. The channel
    // is emphatically not poisoned.
    let s = runChecks(createParkState(3, [3, 2]));
    s = cancel(s, "p0");
    expect(reader(s, "r0").state).toBe("orphaned_unreachable");
    expect(reader(s, "r1").state).toBe("parked");
    expect(s.poisoned).toBe(false);

    s = publish(s, "p1");
    s = publish(s, "p2");
    expect(reader(s, "r1").state).toBe("satisfied");
  });

  it("poisons the channel only when nothing can ever arrive", () => {
    let s = runChecks(createParkState(2, [1]));
    s = cancel(s, "p0");
    expect(s.poisoned).toBe(false);
    s = cancel(s, "p1");
    // No producer left, nothing written, no init seed.
    expect(s.poisoned).toBe(true);
    expect(reader(s, "r0").state).toBe("orphaned_poisoned");
  });

  it("does not poison when an arrival already landed", () => {
    let s = runChecks(createParkState(2, [2]));
    s = publish(s, "p0");
    s = cancel(s, "p1");
    expect(s.poisoned).toBe(false);
    // The reader still orphans itself: 1 + 0 < 2.
    expect(reader(s, "r0").state).toBe("orphaned_unreachable");
  });

  it("checks satisfied before poisoned, so a met reader is never orphaned", () => {
    const s = { ...createParkState(1, [1]), arrival: 1, remaining: 0, poisoned: true };
    expect(evaluate(s, s.readers[0]!)).toBe("satisfied");
  });
});

describe("the race the runtime forecloses", () => {
  it("cannot be opened on a single-threaded runtime", () => {
    // beginCheck is a no-op under `single`: nothing can interleave between the synchronous check
    // and the await, so the window has no duration.
    const s = beginCheck(runChecks(createParkState(3, [2])));
    expect(s.midCheck).toEqual([]);
  });

  it("loses the wake when a publish lands mid-check under multi-threading", () => {
    let s = setThreading(runChecks(createParkState(3, [2])), "multi");
    s = beginCheck(s);
    expect(s.midCheck).toContain("r0");

    s = publish(s, "p0");

    // notify_waiters has no queue: a reader that was not yet parked never hears it.
    expect(reader(s, "r0").state).toBe("lost_wakeup");
    expect(s.log.some((l) => l.includes("WAKE LOST"))).toBe(true);
  });

  it("leaves a lost reader stuck even once its target is met", () => {
    let s = setThreading(runChecks(createParkState(3, [2])), "multi");
    s = beginCheck(s);
    s = publish(s, "p0");
    s = publish(s, "p1");
    s = publish(s, "p2");
    // Arrivals are past the target, and it still never runs again. That is the deadlock.
    expect(s.arrival).toBe(3);
    expect(reader(s, "r0").state).toBe("lost_wakeup");
  });

  it("survives the same sequence on the real single-threaded model", () => {
    let s = runChecks(createParkState(3, [2]));
    s = beginCheck(s);
    s = publish(s, "p0");
    s = publish(s, "p1");
    expect(reader(s, "r0").state).toBe("satisfied");
  });
});

describe("wakeups", () => {
  it("re-checks every parked reader on each write", () => {
    let s = runChecks(createParkState(4, [4, 3]));
    s = publish(s, "p0");
    expect(reader(s, "r0").rechecks).toBe(1);
    expect(reader(s, "r1").rechecks).toBe(1);
  });

  it("stops re-checking a reader once it has settled", () => {
    let s = runChecks(createParkState(3, [1]));
    s = publish(s, "p0");
    expect(reader(s, "r0").state).toBe("satisfied");
    const before = reader(s, "r0").rechecks;
    s = publish(s, "p1");
    expect(reader(s, "r0").rechecks).toBe(before);
  });
});

describe("abort_all", () => {
  it("poisons every channel and settles all readers", () => {
    let s = runChecks(createParkState(3, [3, 2]));
    s = abortAll(s);
    expect(isQuiescent(s)).toBe(true);
    for (const r of s.readers) expect(r.state).toBe("orphaned_poisoned");
  });
});
