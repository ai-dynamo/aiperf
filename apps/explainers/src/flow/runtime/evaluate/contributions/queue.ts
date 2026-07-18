// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Bounds, DrawCommand, HitRegion } from "../../display-list.js";
import {
  simulateQueue,
  type QueueArrival,
  type QueueEvent,
  type QueuePolicy,
} from "../../leaves/queue-policy.js";
import type { SemanticEntityProjection } from "../types.js";

export type QueueContributionInput = Readonly<{
  id: string;
  arrivals: readonly QueueArrival[];
  policy: QueuePolicy;
  capacity?: number;
  atMs: number;
  bounds: Bounds;
  chipWidth?: number;
  padding?: number;
  gap?: number;
  order?: number;
  laneFill?: string;
  waitingFill?: string;
  servingFill?: string;
}>;

export type QueueOccupancy = Readonly<{
  waiting: readonly string[];
  serving: string | null;
  departed: readonly string[];
  rejected: readonly string[];
}>;

export type QueueContribution = Readonly<{
  occupancy: QueueOccupancy;
  commands: readonly DrawCommand[];
  semanticEntities: readonly SemanticEntityProjection[];
  hitRegions: readonly HitRegion[];
}>;

type RequestState = "waiting" | "serving" | "departed" | "rejected";

type RequestSnapshot = Readonly<{
  arrival: QueueArrival;
  state: RequestState;
  transitionAt: number;
  serviceOrderAt: number;
}>;

const EVENT_PRECEDENCE: Readonly<Record<QueueEvent["kind"], number>> = {
  enqueue: 0,
  "start-service": 1,
  depart: 2,
  reject: 3,
};

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }
  return value;
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer.`);
  }
}

function assertInput(input: QueueContributionInput): void {
  assertNonNegativeInteger(
    input.atMs,
    "Queue evaluation time",
  );
  if (
    ![
      input.bounds.x,
      input.bounds.y,
      input.bounds.width,
      input.bounds.height,
    ].every(Number.isFinite) ||
    input.bounds.width < 0 ||
    input.bounds.height < 0
  ) {
    throw new RangeError("Queue bounds must contain finite non-negative dimensions.");
  }
  if (input.capacity !== undefined) {
    assertNonNegativeInteger(input.capacity, "Queue capacity");
  }
  for (const arrival of input.arrivals) {
    assertNonNegativeInteger(arrival.arriveAt, `Queue arrival ${arrival.id} time`);
    assertNonNegativeInteger(
      arrival.serviceMs,
      `Queue arrival ${arrival.id} service time`,
    );
  }
  for (const [name, value] of [
    ["Queue chip width", input.chipWidth ?? 32],
    ["Queue padding", input.padding ?? 4],
    ["Queue gap", input.gap ?? 4],
  ] as const) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be finite and non-negative.`);
    }
  }
  if (!Number.isSafeInteger(input.order ?? 0)) {
    throw new RangeError("Queue order must be a safe integer.");
  }
}

function stateForEvent(kind: QueueEvent["kind"]): RequestState {
  switch (kind) {
    case "enqueue":
      return "waiting";
    case "start-service":
      return "serving";
    case "depart":
      return "departed";
    case "reject":
      return "rejected";
  }
}

function snapshotsAt(
  arrivals: readonly QueueArrival[],
  events: readonly QueueEvent[],
  atMs: number,
): RequestSnapshot[] {
  const eventsByRequest = new Map<string, QueueEvent[]>();
  for (const event of events) {
    const requestEvents = eventsByRequest.get(event.requestId) ?? [];
    requestEvents.push(event);
    eventsByRequest.set(event.requestId, requestEvents);
  }

  return [...arrivals]
    .sort(
      (left, right) =>
        left.arriveAt - right.arriveAt || left.id.localeCompare(right.id),
    )
    .flatMap((arrival): RequestSnapshot[] => {
      const requestEvents = eventsByRequest.get(arrival.id) ?? [];
      const elapsed = requestEvents
        .filter(({ at }) => at <= atMs)
        .sort(
          (left, right) =>
            right.at - left.at ||
            EVENT_PRECEDENCE[right.kind] - EVENT_PRECEDENCE[left.kind],
        );
      const current = elapsed[0];
      if (current === undefined) {
        return [];
      }
      const service = requestEvents.find(({ kind }) => kind === "start-service");
      return [
        {
          arrival,
          state: stateForEvent(current.kind),
          transitionAt: current.at,
          serviceOrderAt: service?.at ?? Number.POSITIVE_INFINITY,
        },
      ];
    });
}

function rectanglePath({ x, y, width, height }: Bounds): string {
  return `M ${x} ${y} H ${x + width} V ${y + height} H ${x} Z`;
}

function requestBounds(
  input: QueueContributionInput,
  waitingIndex: number | null,
): Bounds {
  const padding = input.padding ?? 4;
  const gap = input.gap ?? 4;
  const width = Math.min(
    input.chipWidth ?? 32,
    Math.max(0, input.bounds.width - 2 * padding),
  );
  const height = Math.max(0, input.bounds.height - 2 * padding);
  const x =
    waitingIndex === null
      ? input.bounds.x + input.bounds.width - padding - width
      : input.bounds.x + padding + waitingIndex * (width + gap);
  return {
    x,
    y: input.bounds.y + padding,
    width,
    height,
  };
}

/**
 * Projects a queue simulation at one authored integer time into immutable,
 * backend-neutral display and semantic products.
 */
export function contributeQueue(
  input: QueueContributionInput,
): QueueContribution {
  assertInput(input);
  const simulation = simulateQueue(input.arrivals, input.policy, input.capacity);
  const snapshots = snapshotsAt(input.arrivals, simulation.events, input.atMs);
  const waiting = snapshots
    .filter(({ state }) => state === "waiting")
    .sort(
      (left, right) =>
        left.serviceOrderAt - right.serviceOrderAt ||
        left.arrival.arriveAt - right.arrival.arriveAt ||
        left.arrival.id.localeCompare(right.arrival.id),
    );
  const serving = snapshots.find(({ state }) => state === "serving");
  const byTransition = (left: RequestSnapshot, right: RequestSnapshot): number =>
    left.transitionAt - right.transitionAt ||
    left.arrival.id.localeCompare(right.arrival.id);
  const departed = snapshots
    .filter(({ state }) => state === "departed")
    .sort(byTransition);
  const rejected = snapshots
    .filter(({ state }) => state === "rejected")
    .sort(byTransition);
  const occupancy: QueueOccupancy = {
    waiting: waiting.map(({ arrival }) => arrival.id),
    serving: serving?.arrival.id ?? null,
    departed: departed.map(({ arrival }) => arrival.id),
    rejected: rejected.map(({ arrival }) => arrival.id),
  };

  const order = input.order ?? 0;
  const commands: DrawCommand[] = [
    {
      kind: "path",
      id: `${input.id}:lane`,
      order,
      paintBounds: { ...input.bounds },
      damageBounds: { ...input.bounds },
      path: rectanglePath(input.bounds),
      fill: input.laneFill ?? "#111827",
    },
  ];
  const hitRegions: HitRegion[] = [];
  const visible = [
    ...waiting.map((snapshot, index) => ({
      snapshot,
      bounds: requestBounds(input, index),
      fill: input.waitingFill ?? "#64748b",
    })),
    ...(serving === undefined
      ? []
      : [
          {
            snapshot: serving,
            bounds: requestBounds(input, null),
            fill: input.servingFill ?? "#22c55e",
          },
        ]),
  ];
  for (const [index, item] of visible.entries()) {
    const semanticId = `${input.id}:request:${item.snapshot.arrival.id}`;
    const commandOrder = order + index + 1;
    commands.push({
      kind: "path",
      id: semanticId,
      order: commandOrder,
      paintBounds: item.bounds,
      damageBounds: item.bounds,
      path: rectanglePath(item.bounds),
      fill: item.fill,
    });
    hitRegions.push({
      id: `hit:${semanticId}`,
      semanticId,
      order: commandOrder,
      bounds: item.bounds,
    });
  }

  const semanticEntities: SemanticEntityProjection[] = snapshots.map(
    ({ arrival, state }) => ({
      id: `${input.id}:request:${arrival.id}`,
      label: arrival.id,
      role: "listitem",
      kind: state,
      description: `Queue request ${arrival.id} is ${state}`,
    }),
  );

  return deepFreeze({
    occupancy,
    commands,
    semanticEntities,
    hitRegions,
  });
}
