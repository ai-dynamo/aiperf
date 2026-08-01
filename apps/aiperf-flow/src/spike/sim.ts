/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — a live request-lifecycle simulation, advanced by wall time.
//!
//! Not a replay of a finished run: requests are born here, contend for an admission gate, and
//! die here. That is the difference the spike exists to show — a static chart can draw the
//! outcome of contention, but only a running system can show a queue *filling*.

/** Where a request currently is. Ordered; a request only ever moves forward. */
export type Stage = "queued" | "connect" | "prefill" | "decode" | "done";

export const STAGES: readonly Stage[] = ["queued", "connect", "prefill", "decode", "done"];

export type Request = {
  id: number;
  stage: Stage;
  /** Wall ms the request entered its current stage. */
  enteredAt: number;
  /** Wall ms the request was created. */
  bornAt: number;
  /** Connect duration, ms. */
  connectMs: number;
  /** Time to first token, ms. */
  ttftMs: number;
  /** Inter-token latency, ms. */
  itlMs: number;
  /** Output tokens to emit. */
  tokens: number;
  /** Tokens emitted so far. */
  emitted: number;
  /** Wall ms of the most recent token, for the pulse animation. */
  lastTokenAt: number;
  /** Wall ms the request finished; drives the fade-out. */
  doneAt: number;
};

export type SimConfig = {
  /** Requests started per second. */
  rate: number;
  /** Max requests past the gate at once. Excess waits, visibly. */
  concurrency: number;
  /** Multiplies every duration. Lower is a faster server. */
  serviceScale: number;
};

export type SimState = {
  now: number;
  requests: Request[];
  nextId: number;
  /** Fractional request carried between frames so a rate under 1/frame still works. */
  spawnCredit: number;
  /** Concurrency samples, newest last, for the live curve. */
  history: { t: number; inFlight: number; queued: number }[];
  completed: number;
  /** Tokens emitted since the run started. */
  tokensOut: number;
};

export const DEFAULT_CONFIG: SimConfig = { rate: 3, concurrency: 4, serviceScale: 1 };

/** How long the live curve remembers, ms. */
export const HISTORY_MS = 12_000;

export function createSim(now: number): SimState {
  return {
    now,
    requests: [],
    nextId: 1,
    spawnCredit: 0,
    history: [],
    completed: 0,
    tokensOut: 0,
  };
}

/** Deterministic-ish jitter so the rig has texture without needing a seeded RNG. */
function jitter(id: number, salt: number): number {
  const x = Math.sin(id * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function spawn(id: number, now: number, scale: number): Request {
  return {
    id,
    stage: "queued",
    enteredAt: now,
    bornAt: now,
    connectMs: (60 + jitter(id, 1) * 90) * scale,
    ttftMs: (260 + jitter(id, 2) * 420) * scale,
    itlMs: (28 + jitter(id, 3) * 34) * scale,
    tokens: Math.round(18 + jitter(id, 4) * 44),
    emitted: 0,
    lastTokenAt: 0,
    doneAt: 0,
  };
}

/** Requests past the admission gate — everything not still queued and not finished. */
export function inFlight(requests: readonly Request[]): number {
  return requests.filter((r) => r.stage !== "queued" && r.stage !== "done").length;
}

export function queued(requests: readonly Request[]): number {
  return requests.filter((r) => r.stage === "queued").length;
}

/**
 * Advance the world by `dtMs`.
 *
 * Admission is the interesting part: the gate admits only up to `concurrency`, in arrival order,
 * so lowering the limit does not drop work — it makes the queue grow, which is the whole point of
 * being able to watch it.
 */
export function step(state: SimState, dtMs: number, config: SimConfig): SimState {
  const now = state.now + dtMs;
  const scale = config.serviceScale;

  // Retire finished requests a moment after they land, so the eye can catch them leaving.
  const requests = state.requests.filter((r) => r.stage !== "done" || now - r.doneAt < 900);

  let { nextId, spawnCredit, completed, tokensOut } = state;

  spawnCredit += (config.rate * dtMs) / 1000;
  while (spawnCredit >= 1) {
    spawnCredit -= 1;
    requests.push(spawn(nextId, now, scale));
    nextId += 1;
  }

  for (const r of requests) {
    const elapsed = now - r.enteredAt;
    if (r.stage === "connect" && elapsed >= r.connectMs) {
      r.stage = "prefill";
      r.enteredAt = now;
    } else if (r.stage === "prefill" && elapsed >= r.ttftMs) {
      r.stage = "decode";
      r.enteredAt = now;
      r.emitted = 1;
      r.lastTokenAt = now;
    } else if (r.stage === "decode") {
      const due = Math.floor(elapsed / r.itlMs) + 1;
      const target = Math.min(r.tokens, due);
      if (target > r.emitted) {
        tokensOut += target - r.emitted;
        r.emitted = target;
        r.lastTokenAt = now;
      }
      if (r.emitted >= r.tokens) {
        r.stage = "done";
        r.enteredAt = now;
        r.doneAt = now;
        completed += 1;
      }
    }
  }

  // Admit in arrival order, up to the limit. Everyone else keeps waiting, visibly.
  let free = config.concurrency - inFlight(requests);
  for (const r of requests) {
    if (free <= 0) break;
    if (r.stage !== "queued") continue;
    r.stage = "connect";
    r.enteredAt = now;
    free -= 1;
  }

  const history = [
    ...state.history,
    { t: now, inFlight: inFlight(requests), queued: queued(requests) },
  ].filter((h) => now - h.t <= HISTORY_MS);

  return { now, requests, nextId, spawnCredit, history, completed, tokensOut };
}
