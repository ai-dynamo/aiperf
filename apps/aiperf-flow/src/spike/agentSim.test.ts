/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  createAgentSim,
  stepAgents,
  laneOrder,
  idleFraction,
  DEFAULT_AGENT_CONFIG,
  TICK_MS,
  type AgentSimState,
} from "./agentSim.js";

/** Advance `ms` in chunks of `chunk`, mimicking a particular frame cadence. */
function run(state: AgentSimState, ms: number, chunk: number): AgentSimState {
  let out = state;
  for (let elapsed = 0; elapsed < ms; elapsed += chunk) {
    out = stepAgents(out, chunk, DEFAULT_AGENT_CONFIG);
  }
  return out;
}

/** The parts of a session a viewer would notice differing. */
function shape(s: AgentSimState) {
  return {
    now: s.now,
    lanes: laneOrder(s.agents).map((a) => `${a.label}@${a.bornAt}`),
    spawned: s.spawnedTotal,
    turns: s.turns.map((t) => `${t.id}:${t.agentId}:${t.startAt}:${t.endAt ?? "live"}`),
  };
}

describe("determinism", () => {
  it("is independent of how elapsed time was chopped into frames", () => {
    // The bug this pins: a variable timestep let a slow frame shift a completion, which shifted
    // the live population, which changed a spawn decision. Same config gave 6 lanes on one run
    // and 4 on another at the same elapsed time.
    const steady = run(createAgentSim(7), 8000, 16);
    const chunky = run(createAgentSim(7), 8000, 100);
    const single = stepAgents(createAgentSim(7), 8000, DEFAULT_AGENT_CONFIG);

    expect(shape(chunky)).toEqual(shape(steady));
    expect(shape(single)).toEqual(shape(steady));
  });

  it("replays identically from the same seed", () => {
    expect(shape(run(createAgentSim(3), 6000, 20))).toEqual(
      shape(run(createAgentSim(3), 6000, 20)),
    );
  });

  it("gives a different session for a different seed", () => {
    const a = run(createAgentSim(1), 6000, 20);
    const b = run(createAgentSim(2), 6000, 20);
    expect(shape(a)).not.toEqual(shape(b));
  });

  it("carries sub-tick time forward instead of dropping it", () => {
    // Ten calls below the quantum must still add up to one tick of progress.
    let s = createAgentSim(1);
    for (let i = 0; i < 10; i++) s = stepAgents(s, TICK_MS / 10, DEFAULT_AGENT_CONFIG);
    expect(s.now).toBe(TICK_MS);
  });
});

describe("population control", () => {
  it("converges instead of running away", () => {
    // Fan-out halves per level and backs off near the ceiling, so a long run stays bounded.
    const s = run(createAgentSim(5), 60_000, 20);
    const active = s.agents.filter((a) => a.retiredAt === null).length;
    expect(active).toBeLessThanOrEqual(DEFAULT_AGENT_CONFIG.maxActive + 2);
  });

  it("never nests past the configured depth", () => {
    const s = run(createAgentSim(9), 30_000, 20);
    expect(Math.max(...s.agents.map((a) => a.depth))).toBeLessThanOrEqual(
      DEFAULT_AGENT_CONFIG.maxDepth,
    );
  });

  it("spawns nothing at all when nesting is disabled", () => {
    let s = createAgentSim(4);
    for (let i = 0; i < 1000; i++) s = stepAgents(s, 20, { ...DEFAULT_AGENT_CONFIG, maxDepth: 0 });
    expect(s.spawnedTotal).toBe(0);
    expect(s.agents).toHaveLength(1);
  });
});

describe("laneOrder", () => {
  it("places a subagent directly beneath the agent that spawned it", () => {
    const s = run(createAgentSim(5), 20_000, 20);
    const order = laneOrder(s.agents);
    for (const agent of order) {
      if (agent.parentId === null) continue;
      const parentIndex = order.findIndex((a) => a.id === agent.parentId);
      expect(parentIndex).toBeGreaterThanOrEqual(0);
      expect(order.indexOf(agent)).toBeGreaterThan(parentIndex);
    }
  });
});

describe("idleFraction", () => {
  it("counts overlapping lanes as one busy stretch, not two", () => {
    const base = createAgentSim(1);
    const state: AgentSimState = {
      ...base,
      now: 1000,
      turns: [
        { ...stubTurn(1, 0), startAt: 0, endAt: 500 },
        { ...stubTurn(2, 1), startAt: 100, endAt: 400 },
      ],
    };
    // Union of [0,500] and [100,400] is 500ms busy out of 1000 — half idle, not a quarter.
    expect(idleFraction(state, 1000)).toBeCloseTo(0.5, 5);
  });

  it("reports a fully idle window as entirely idle", () => {
    const state = { ...createAgentSim(1), now: 1000, turns: [] };
    expect(idleFraction(state, 1000)).toBe(1);
  });
});

function stubTurn(id: number, agentId: number) {
  return {
    id,
    agentId,
    startAt: 0,
    endAt: null as number | null,
    firstTokenAt: 0,
    ttftMs: 10,
    itlMs: 10,
    tokens: 5,
    emitted: 5,
    canSpawn: false,
    spawnCount: 0,
  };
}
