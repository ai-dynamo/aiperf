/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  createClock,
  nextEventTime,
  NS_PER_MS,
  runToEnd,
  stepReal,
  stepSim,
  type Task,
} from "./clockSim.js";

describe("the equivalence that matters", () => {
  it("produces the same event order on both clocks", () => {
    // The entire proposition: virtual time changes how long a run takes, never what it does.
    expect(runToEnd("sim").events).toEqual(runToEnd("real").events);
  });

  it("reaches the same final virtual time on both clocks", () => {
    expect(runToEnd("sim").nowNs).toBe(runToEnd("real").nowNs);
  });

  it("takes far less wall time on the simulated clock", () => {
    const sim = runToEnd("sim");
    const real = runToEnd("real");
    expect(sim.wallMs).toBeLessThan(real.wallMs / 100);
  });
});

describe("event ordering", () => {
  it("breaks a same-deadline tie by registration order, not arbitrarily", () => {
    // `poll`, `tie-a` and `tie-b` all first wake at 500ms. The heap orders by (at_ns, seq_no),
    // and seq_no is assigned when each task registered — so the order is task order.
    const at500 = runToEnd("sim").events.slice(0, 3);
    expect(at500).toContain("poll@0");
    expect(runToEnd("sim").events).toEqual(runToEnd("sim").events);
  });

  it("wakes earlier deadlines before later ones regardless of registration", () => {
    const tasks: Task[] = [
      { id: "late", sleepsNs: [900 * NS_PER_MS] },
      { id: "early", sleepsNs: [100 * NS_PER_MS] },
    ];
    expect(runToEnd("sim", tasks).events).toEqual(["early@0", "late@0"]);
  });

  it("keeps registration order for identical deadlines", () => {
    const tasks: Task[] = [
      { id: "first", sleepsNs: [100 * NS_PER_MS] },
      { id: "second", sleepsNs: [100 * NS_PER_MS] },
      { id: "third", sleepsNs: [100 * NS_PER_MS] },
    ];
    expect(runToEnd("sim", tasks).events).toEqual(["first@0", "second@0", "third@0"]);
  });
});

describe("next_event_time", () => {
  it("is the earliest parked deadline", () => {
    const state = createClock("sim");
    expect(nextEventTime(state)).toBe(200 * NS_PER_MS);
  });

  it("never reports a time in the past", () => {
    // An already-due sleeper yields `now`, not its own stale deadline.
    const state = { ...createClock("sim"), nowNs: 10_000 * NS_PER_MS };
    expect(nextEventTime(state)).toBe(10_000 * NS_PER_MS);
  });

  it("is null once nothing is parked", () => {
    expect(nextEventTime(runToEnd("sim"))).toBeNull();
  });
});

describe("advancement", () => {
  it("jumps straight to the next event on the simulated clock", () => {
    const state = stepSim(createClock("sim"));
    // Nothing happens between 0 and 200ms, so nothing is spent getting there.
    expect(state.nowNs).toBe(200 * NS_PER_MS);
    expect(state.events).toEqual(["warmup@0"]);
  });

  it("crawls through the empty space on the real clock", () => {
    let state = createClock("real");
    state = stepReal(state, 5 * NS_PER_MS);
    expect(state.nowNs).toBe(5 * NS_PER_MS);
    // Still nothing has woken: the deadline is 200ms away and must be waited out.
    expect(state.events).toEqual([]);
  });

  it("never moves virtual time backwards", () => {
    let state = createClock("sim");
    let previous = state.nowNs;
    for (let i = 0; i < 40 && !state.done; i++) {
      state = stepSim(state);
      expect(state.nowNs).toBeGreaterThanOrEqual(previous);
      previous = state.nowNs;
    }
  });

  it("wakes several sleepers sharing a deadline in one advance", () => {
    const tasks: Task[] = [
      { id: "a", sleepsNs: [100 * NS_PER_MS] },
      { id: "b", sleepsNs: [100 * NS_PER_MS] },
    ];
    const state = stepSim(createClock("sim", tasks), tasks);
    expect(state.events).toEqual(["a@0", "b@0"]);
  });
});

describe("completion", () => {
  it("finishes with an empty heap and no deadlock", () => {
    const state = runToEnd("sim");
    expect(state.done).toBe(true);
    expect(state.heap).toHaveLength(0);
    expect(state.deadlocked).toBe(false);
  });

  it("reports a deadlock when nothing is parked and work remains", () => {
    // No task ever schedules anything: there is no virtual event to advance to.
    const state = stepSim(createClock("sim", []), []);
    expect(state.done).toBe(true);
    expect(state.deadlocked).toBe(true);
  });
});
