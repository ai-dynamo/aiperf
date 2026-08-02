/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  createClock,
  defaultRequests,
  nextEventTime,
  requestsToTasks,
  summarize,
  type Request,
  NS_PER_MS,
  runToEnd,
  stepReal,
  stepSim,
  type Task,
} from "./clockSim.js";

const order = (s: ReturnType<typeof runToEnd>) => s.events.map((e) => `${e.taskId} ${e.label}`);

describe("the equivalence that matters", () => {
  it("produces exactly the same events, and each request in its own order", () => {
    // What virtual time preserves is *what happened to each request*. The global interleaving is
    // not preserved and cannot be: a real tick drains several due sleepers as one batch, and a
    // wake that re-schedules during that drain lands in the next batch rather than immediately.
    // Simulation visits every deadline separately, so it sees the order the timings imply.
    const sim = runToEnd("sim");
    const real = runToEnd("real");
    expect([...order(real)].sort()).toEqual([...order(sim)].sort());
    for (const id of new Set(sim.events.map((e) => e.taskId))) {
      const per = (s: typeof sim) => s.events.filter((e) => e.taskId === id).map((e) => e.label);
      expect(per(real)).toEqual(per(sim));
    }
  });

  it("gives the simulated run the interleaving the timings actually imply", () => {
    // Sorting the sim's own events by timestamp reproduces its order exactly; the real run's
    // does not, because its timestamps were rounded up to tick boundaries.
    const sim = runToEnd("sim");
    const byTime = [...sim.events].sort((a, b) => a.atNs - b.atNs).map((e) => `${e.taskId} ${e.label}`);
    expect(byTime).toEqual(order(sim));
  });

  it("times events exactly on the simulated clock", () => {
    // advance_to jumps to the deadline itself, so a sleep of 380ms lands at 380ms.
    const one = requestsToTasks([
      { id: "r", arrivalNs: 100 * NS_PER_MS, ttftNs: 380 * NS_PER_MS, itlNs: 20 * NS_PER_MS, tokens: 2 },
    ]);
    const sim = runToEnd("sim", one);
    expect(sim.events[0]!.atNs).toBe(100 * NS_PER_MS);
    expect(sim.events[1]!.atNs).toBe(480 * NS_PER_MS);
  });

  it("times events late on the real clock, never early", () => {
    // A real timer fires when the runtime next gets to it, which is at or after the deadline.
    // That is scheduler jitter, and it is exactly what simulation removes.
    const one = requestsToTasks([
      { id: "r", arrivalNs: 100 * NS_PER_MS, ttftNs: 380 * NS_PER_MS, itlNs: 20 * NS_PER_MS, tokens: 2 },
    ]);
    const sim = runToEnd("sim", one);
    const real = runToEnd("real", one);
    for (let i = 0; i < sim.events.length; i++) {
      expect(real.events[i]!.atNs).toBeGreaterThanOrEqual(sim.events[i]!.atNs);
    }
    expect(real.nowNs).toBeGreaterThanOrEqual(sim.nowNs);
  });

  it("takes far less wall time on the simulated clock", () => {
    const sim = runToEnd("sim");
    const real = runToEnd("real");
    expect(sim.wallMs).toBeLessThan(real.wallMs / 100);
  });
});

describe("event ordering", () => {
  it("is reproducible run to run", () => {
    expect(order(runToEnd("sim"))).toEqual(order(runToEnd("sim")));
  });

  it("wakes earlier deadlines before later ones regardless of registration", () => {
    const tasks: Task[] = [
      { id: "late", sleepsNs: [900 * NS_PER_MS] },
      { id: "early", sleepsNs: [100 * NS_PER_MS] },
    ];
    expect(order(runToEnd("sim", tasks))).toEqual(["early step 0", "late step 0"]);
  });

  it("keeps registration order for identical deadlines", () => {
    const tasks: Task[] = [
      { id: "first", sleepsNs: [100 * NS_PER_MS] },
      { id: "second", sleepsNs: [100 * NS_PER_MS] },
      { id: "third", sleepsNs: [100 * NS_PER_MS] },
    ];
    expect(order(runToEnd("sim", tasks))).toEqual(["first step 0", "second step 0", "third step 0"]);
  });
});

describe("next_event_time", () => {
  it("is the earliest parked deadline", () => {
    expect(nextEventTime(createClock("sim"))).toBe(120 * NS_PER_MS);
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
    // The first arrival is at 120ms; nothing happens before it, so nothing is spent getting there.
    expect(state.nowNs).toBe(120 * NS_PER_MS);
    expect(state.events).toHaveLength(1);
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
    expect(state.events.map((e) => e.taskId)).toEqual(["a", "b"]);
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

describe("what an end user compares", () => {
  it("agrees on what happened, within scheduler jitter", () => {
    // The mechanism is beside the point; this is the claim. Same run, same conclusions — and the
    // simulated numbers are the exact ones, because nothing had to wait to produce them.
    const sim = summarize(runToEnd("sim"));
    const real = summarize(runToEnd("real"));
    expect(real.completed).toBe(sim.completed);
    expect(real.tokens).toBe(sim.tokens);
    // Real timings are late by at most one scheduler tick per event, never early.
    expect(real.meanTtftMs).toBeGreaterThanOrEqual(sim.meanTtftMs);
    expect(real.meanTtftMs - sim.meanTtftMs).toBeLessThan(6);
  });

  it("reports every request as completed", () => {
    expect(summarize(runToEnd("sim")).completed).toBe(defaultRequests().length);
  });

  it("measures TTFT from sent to first token", () => {
    const one: Request[] = [{ id: "r", arrivalNs: 100 * NS_PER_MS, ttftNs: 250 * NS_PER_MS, itlNs: 10 * NS_PER_MS, tokens: 3 }];
    const result = summarize(runToEnd("sim", requestsToTasks(one)));
    expect(result.meanTtftMs).toBeCloseTo(250, 6);
  });

  it("counts every generated token", () => {
    const one: Request[] = [{ id: "r", arrivalNs: 0, ttftNs: 100 * NS_PER_MS, itlNs: 10 * NS_PER_MS, tokens: 5 }];
    expect(summarize(runToEnd("sim", requestsToTasks(one))).tokens).toBe(5);
  });
});
