/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — a live agent session: a main agent that takes turns, spawns subagents, and waits.
//!
//! The subject is structure appearing over time. A recorded trace is a finished forest; here the
//! forest *grows* — a lane does not exist until its parent spawns it, and the dead air between
//! turns accumulates in front of you rather than being summarised after the fact.

/** One agent lane. Lanes are born by a spawn and retire when their work runs out. */
export type Agent = {
  id: number;
  label: string;
  parentId: number | null;
  depth: number;
  bornAt: number;
  /** Null while still working. */
  retiredAt: number | null;
  /** Sim ms the agent may start its next turn. */
  nextTurnAt: number;
  turnsLeft: number;
};

/** One request on a lane. `endAt` is null while it is still streaming. */
export type Turn = {
  id: number;
  agentId: number;
  startAt: number;
  endAt: number | null;
  /** Sim ms the first token landed; before that the turn is in prefill. */
  firstTokenAt: number | null;
  ttftMs: number;
  itlMs: number;
  tokens: number;
  emitted: number;
  /** Whether nesting depth allows this turn to spawn at all. */
  canSpawn: boolean;
  /** Resolved at completion by `spawnCountFor`. */
  spawnCount: number;
};

/** A parent turn's completion linked to the child lane it created. */
export type Spawn = {
  parentTurnId: number;
  childAgentId: number;
  at: number;
};

export type AgentSimConfig = {
  /** Think time between a lane's turns, ms. This is the dead air the warp later collapses. */
  thinkMs: number;
  /** Chance a completed turn spawns subagents, 0..1. */
  spawnChance: number;
  /** Nesting cap. Depth 0 is the main agent. */
  maxDepth: number;
  /** Soft ceiling on concurrently-working lanes. Spawning backs off as this is approached. */
  maxActive: number;
  serviceScale: number;
};

export type AgentSimState = {
  now: number;
  agents: Agent[];
  turns: Turn[];
  spawns: Spawn[];
  nextAgentId: number;
  nextTurnId: number;
  spawnedTotal: number;
};

export const DEFAULT_AGENT_CONFIG: AgentSimConfig = {
  thinkMs: 600,
  spawnChance: 0.55,
  maxDepth: 2,
  maxActive: 7,
  serviceScale: 1,
};

/** How much history the view keeps, in sim ms. */
export const WINDOW_MS = 14_000;

const SUB_NAMES = ["explore", "research", "code", "review", "search", "verify", "plan", "test"];

/** Deterministic pseudo-random in [0,1) from two integers. No global RNG state to reset. */
export function rand(a: number, b: number): number {
  const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export function createAgentSim(): AgentSimState {
  return {
    now: 0,
    agents: [
      {
        id: 0,
        label: "main",
        parentId: null,
        depth: 0,
        bornAt: 0,
        retiredAt: null,
        nextTurnAt: 0,
        turnsLeft: Infinity,
      },
    ],
    turns: [],
    spawns: [],
    nextAgentId: 1,
    nextTurnId: 1,
    spawnedTotal: 0,
  };
}

function startTurn(state: AgentSimState, agent: Agent, config: AgentSimConfig): Turn {
  const id = state.nextTurnId++;
  const s = config.serviceScale;
  const canSpawn = agent.depth < config.maxDepth;
  return {
    id,
    agentId: agent.id,
    startAt: state.now,
    endAt: null,
    firstTokenAt: null,
    ttftMs: (240 + rand(id, 1) * 380) * s,
    itlMs: (30 + rand(id, 2) * 30) * s,
    tokens: Math.round(14 + rand(id, 3) * 30),
    emitted: 0,
    canSpawn,
    spawnCount: 0,
  };
}

/**
 * How many subagents a completing turn spawns.
 *
 * Decided at completion, not at creation, so it can see the live population. Two dampers keep a
 * session from running away: fan-out halves with each level of nesting, and spawning backs off
 * smoothly as active lanes approach the ceiling instead of stopping dead at it.
 */
export function spawnCountFor(
  turn: Turn,
  depth: number,
  activeCount: number,
  config: AgentSimConfig,
): number {
  if (!turn.canSpawn) return 0;
  const headroom = Math.max(0, 1 - activeCount / Math.max(1, config.maxActive));
  const chance = config.spawnChance * Math.pow(0.5, depth) * headroom;
  if (rand(turn.id, 4) >= chance) return 0;
  return 1 + (rand(turn.id, 5) < 0.35 ? 1 : 0);
}

/**
 * Fraction of the visible window in which *no* lane was working.
 *
 * This is the quantity the idle-gap warp exists to remove, measured as a union of active
 * intervals rather than a sum — two subagents overlapping is one busy stretch, not two. Watching
 * it climb while subagents think is the argument for warping, made before the warp is mentioned.
 */
export function idleFraction(state: AgentSimState, windowMs: number): number {
  const from = Math.max(0, state.now - windowMs);
  const span = state.now - from;
  if (span <= 0) return 0;

  const intervals = state.turns
    .map((t) => [Math.max(from, t.startAt), Math.min(state.now, t.endAt ?? state.now)] as const)
    .filter(([a, b]) => b > a)
    .sort((p, q) => p[0] - q[0]);

  let busy = 0;
  let cursor = from;
  for (const [a, b] of intervals) {
    const start = Math.max(a, cursor);
    if (b > start) {
      busy += b - start;
      cursor = b;
    }
  }
  return Math.max(0, Math.min(1, 1 - busy / span));
}

/** The turn currently running on a lane, if any. */
export function runningTurn(state: AgentSimState, agentId: number): Turn | undefined {
  return state.turns.find((t) => t.agentId === agentId && t.endAt === null);
}

/**
 * Depth-first lane order, so a subagent sits directly beneath the agent that spawned it.
 *
 * Arrival order would scatter siblings apart as unrelated lanes are born between them, which
 * loses the one relationship this view exists to show.
 */
export function laneOrder(agents: readonly Agent[]): Agent[] {
  const byParent = new Map<number | null, Agent[]>();
  for (const a of agents) {
    const list = byParent.get(a.parentId) ?? [];
    list.push(a);
    byParent.set(a.parentId, list);
  }
  const out: Agent[] = [];
  const walk = (parentId: number | null): void => {
    for (const a of (byParent.get(parentId) ?? []).sort((x, y) => x.bornAt - y.bornAt)) {
      out.push(a);
      walk(a.id);
    }
  };
  walk(null);
  return out;
}

/** Advance the session by `dtMs`. */
export function stepAgents(
  state: AgentSimState,
  dtMs: number,
  config: AgentSimConfig,
): AgentSimState {
  const now = state.now + dtMs;
  const next: AgentSimState = { ...state, now };

  // Drop lanes and turns that have scrolled out of the window entirely.
  next.agents = state.agents.filter(
    (a) => a.retiredAt === null || now - a.retiredAt < WINDOW_MS,
  );
  const liveIds = new Set(next.agents.map((a) => a.id));
  next.turns = state.turns.filter(
    (t) => liveIds.has(t.agentId) && (t.endAt === null || now - t.endAt < WINDOW_MS),
  );
  next.spawns = state.spawns.filter((s) => now - s.at < WINDOW_MS && liveIds.has(s.childAgentId));

  const turns = next.turns.map((t) => ({ ...t }));
  const agents = next.agents.map((a) => ({ ...a }));
  const spawns = [...next.spawns];
  let { nextAgentId, nextTurnId, spawnedTotal } = next;

  for (const turn of turns) {
    if (turn.endAt !== null) continue;
    const elapsed = now - turn.startAt;
    if (turn.firstTokenAt === null) {
      if (elapsed >= turn.ttftMs) {
        turn.firstTokenAt = now;
        turn.emitted = 1;
      }
      continue;
    }
    const due = Math.floor((now - turn.firstTokenAt) / turn.itlMs) + 1;
    turn.emitted = Math.min(turn.tokens, due);
    if (turn.emitted < turn.tokens) continue;

    turn.endAt = now;
    const agent = agents.find((a) => a.id === turn.agentId);
    if (agent === undefined) continue;
    agent.turnsLeft -= 1;
    agent.nextTurnAt = now + config.thinkMs * (0.6 + rand(turn.id, 6) * 1.2);

    const activeNow = agents.filter((a) => a.retiredAt === null).length;
    turn.spawnCount = spawnCountFor(turn, agent.depth, activeNow, config);
    for (let i = 0; i < turn.spawnCount; i++) {
      const id = nextAgentId++;
      const name = SUB_NAMES[id % SUB_NAMES.length]!;
      agents.push({
        id,
        label: `${name}-${id}`,
        parentId: agent.id,
        depth: agent.depth + 1,
        bornAt: now,
        retiredAt: null,
        // Staggered so siblings do not start in lockstep, which reads as one event not two.
        nextTurnAt: now + 120 + rand(id, 7) * 420,
        turnsLeft: 2 + Math.floor(rand(id, 8) * 3),
      });
      spawns.push({ parentTurnId: turn.id, childAgentId: id, at: now });
      spawnedTotal += 1;
    }
  }

  const state2: AgentSimState = { ...next, turns, agents, spawns, nextAgentId, nextTurnId, spawnedTotal };

  for (const agent of agents) {
    if (agent.retiredAt !== null) continue;
    const running = turns.find((t) => t.agentId === agent.id && t.endAt === null);
    if (running !== undefined) continue;
    if (agent.turnsLeft <= 0) {
      agent.retiredAt = now;
      continue;
    }
    if (now >= agent.nextTurnAt) {
      state2.nextTurnId = nextTurnId;
      const turn = startTurn(state2, agent, config);
      nextTurnId = state2.nextTurnId;
      turns.push(turn);
    }
  }

  return { ...state2, nextTurnId, spawnedTotal };
}
