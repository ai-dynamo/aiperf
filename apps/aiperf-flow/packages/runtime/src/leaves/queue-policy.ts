// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type QueueArrival = Readonly<{
  id: string;
  arriveAt: number;
  serviceMs: number;
  priority?: number;
}>;

export type QueuePolicy = "fifo" | "priority";

export type QueueEvent = Readonly<{
  id: string;
  requestId: string;
  kind: "enqueue" | "start-service" | "depart" | "reject";
  at: number;
}>;

export type QueueSimulation = Readonly<{
  events: readonly QueueEvent[];
}>;

type WaitingArrival = Readonly<{
  id: string;
  arriveAt: number;
  serviceMs: number;
  priority: number;
}>;

/** Simulates single-server queue service under FIFO or priority policy. */
export function simulateQueue(
  arrivals: readonly QueueArrival[],
  policy: QueuePolicy,
  capacity?: number,
): QueueSimulation {
  const sorted = [...arrivals].sort(
    (left, right) => left.arriveAt - right.arriveAt || left.id.localeCompare(right.id),
  );

  const events: QueueEvent[] = [];
  const waiting: WaitingArrival[] = [];
  let serverFreeAt = 0;
  let eventIndex = 0;

  const nextEventId = (): string => {
    const id = `qev-${eventIndex}`;
    eventIndex += 1;
    return id;
  };

  const enqueue = (arrival: WaitingArrival): void => {
    events.push({
      id: nextEventId(),
      requestId: arrival.id,
      kind: "enqueue",
      at: arrival.arriveAt,
    });
    waiting.push(arrival);
  };

  const reject = (arrival: WaitingArrival): void => {
    events.push({
      id: nextEventId(),
      requestId: arrival.id,
      kind: "reject",
      at: arrival.arriveAt,
    });
  };

  const compareWaiting = (left: WaitingArrival, right: WaitingArrival): number => {
    if (policy === "priority") {
      if (right.priority !== left.priority) {
        return right.priority - left.priority;
      }
    }
    return left.arriveAt - right.arriveAt || left.id.localeCompare(right.id);
  };

  const selectNext = (): WaitingArrival | undefined => {
    if (waiting.length === 0) {
      return undefined;
    }
    waiting.sort(compareWaiting);
    return waiting.shift();
  };

  const startService = (arrival: WaitingArrival, at: number): void => {
    events.push({
      id: nextEventId(),
      requestId: arrival.id,
      kind: "start-service",
      at,
    });
    events.push({
      id: nextEventId(),
      requestId: arrival.id,
      kind: "depart",
      at: at + arrival.serviceMs,
    });
    serverFreeAt = at + arrival.serviceMs;
  };

  for (const arrival of sorted) {
    const normalized: WaitingArrival = {
      id: arrival.id,
      arriveAt: arrival.arriveAt,
      serviceMs: arrival.serviceMs,
      priority: arrival.priority ?? 0,
    };

    if (capacity !== undefined && waiting.length >= capacity) {
      reject(normalized);
      continue;
    }

    enqueue(normalized);

    if (serverFreeAt <= normalized.arriveAt) {
      const next = selectNext();
      if (next !== undefined) {
        startService(next, normalized.arriveAt);
      }
    }
  }

  while (waiting.length > 0) {
    const next = selectNext();
    if (next === undefined) {
      break;
    }
    const startAt = Math.max(serverFreeAt, next.arriveAt);
    startService(next, startAt);
  }

  return { events };
}
