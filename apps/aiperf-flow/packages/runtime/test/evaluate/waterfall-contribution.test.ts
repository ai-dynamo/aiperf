// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  contributeWaterfall,
  type WaterfallContributionInput,
} from "../../src/evaluate/contributions/waterfall.js";

const layout = {
  laneOrder: ["arrival", "admission", "connect", "first-token"],
  originX: 0,
  originY: 0,
  laneHeight: 16,
  laneGap: 4,
  pxPerMs: 1,
} as const;

const events = [
  { id: "ev-arrival", lane: "arrival", start: 0, end: 0 },
  { id: "ev-admission", lane: "admission", start: 2, end: 2 },
  { id: "ev-connect", lane: "connect", start: 2, end: 18 },
  { id: "ev-first-token", lane: "first-token", start: 120, end: 120 },
] as const;

const baseInput = {
  id: "lifecycle",
  events,
  layout,
  order: 3,
} satisfies Omit<WaterfallContributionInput, "atMs">;

function evaluate(atMs: number) {
  return contributeWaterfall({ ...baseInput, atMs });
}

describe("contributeWaterfall", () => {
  test("emits nested-lane commands, labels, relations, and hit regions from leaf layout", () => {
    const contribution = contributeWaterfall({
      ...baseInput,
      atMs: 120,
      reducedMotion: true,
    });

    expect(contribution.commands).toEqual([
      {
        kind: "group",
        id: "lifecycle:lane:arrival",
        order: 3,
        paintBounds: { x: 0, y: 0, width: 1, height: 16 },
        damageBounds: { x: 0, y: 0, width: 1, height: 16 },
        children: [
          {
            kind: "text",
            id: "lifecycle:lane:arrival:label",
            order: 0,
            paintBounds: { x: -40, y: 0, width: 36, height: 16 },
            damageBounds: { x: -40, y: 0, width: 36, height: 16 },
            text: "arrival",
            origin: { x: -40, y: 12 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
          {
            kind: "path",
            id: "lifecycle:ev-arrival",
            order: 1,
            paintBounds: { x: 0, y: 0, width: 1, height: 16 },
            damageBounds: { x: 0, y: 0, width: 1, height: 16 },
            path: "M 0 0 H 1 V 16 H 0 Z",
            fill: "#7dcfff",
          },
        ],
      },
      {
        kind: "group",
        id: "lifecycle:lane:admission",
        order: 4,
        paintBounds: { x: 2, y: 20, width: 1, height: 16 },
        damageBounds: { x: 2, y: 20, width: 1, height: 16 },
        children: [
          {
            kind: "text",
            id: "lifecycle:lane:admission:label",
            order: 0,
            paintBounds: { x: -40, y: 20, width: 36, height: 16 },
            damageBounds: { x: -40, y: 20, width: 36, height: 16 },
            text: "admission",
            origin: { x: -40, y: 32 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
          {
            kind: "path",
            id: "lifecycle:ev-admission",
            order: 1,
            paintBounds: { x: 2, y: 20, width: 1, height: 16 },
            damageBounds: { x: 2, y: 20, width: 1, height: 16 },
            path: "M 2 20 H 3 V 36 H 2 Z",
            fill: "#7dcfff",
          },
        ],
      },
      {
        kind: "group",
        id: "lifecycle:lane:connect",
        order: 5,
        paintBounds: { x: 2, y: 40, width: 16, height: 16 },
        damageBounds: { x: 2, y: 40, width: 16, height: 16 },
        children: [
          {
            kind: "text",
            id: "lifecycle:lane:connect:label",
            order: 0,
            paintBounds: { x: -40, y: 40, width: 36, height: 16 },
            damageBounds: { x: -40, y: 40, width: 36, height: 16 },
            text: "connect",
            origin: { x: -40, y: 52 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
          {
            kind: "path",
            id: "lifecycle:ev-connect",
            order: 1,
            paintBounds: { x: 2, y: 40, width: 16, height: 16 },
            damageBounds: { x: 2, y: 40, width: 16, height: 16 },
            path: "M 2 40 H 18 V 56 H 2 Z",
            fill: "#38bdf8",
          },
        ],
      },
      {
        kind: "group",
        id: "lifecycle:lane:first-token",
        order: 6,
        paintBounds: { x: 120, y: 60, width: 1, height: 16 },
        damageBounds: { x: 120, y: 60, width: 1, height: 16 },
        children: [
          {
            kind: "text",
            id: "lifecycle:lane:first-token:label",
            order: 0,
            paintBounds: { x: -40, y: 60, width: 36, height: 16 },
            damageBounds: { x: -40, y: 60, width: 36, height: 16 },
            text: "first-token",
            origin: { x: -40, y: 72 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
          {
            kind: "path",
            id: "lifecycle:ev-first-token",
            order: 1,
            paintBounds: { x: 120, y: 60, width: 1, height: 16 },
            damageBounds: { x: 120, y: 60, width: 1, height: 16 },
            path: "M 120 60 H 121 V 76 H 120 Z",
            fill: "#7dcfff",
          },
        ],
      },
    ]);

    expect(contribution.semanticEntities).toEqual([
      {
        id: "lifecycle:lane:arrival",
        label: "arrival",
        role: "row",
        kind: "lane",
      },
      {
        id: "ev-arrival",
        label: "ev-arrival",
        role: "listitem",
        kind: "point",
        description: "Point event on lane arrival",
      },
      {
        id: "lifecycle:lane:admission",
        label: "admission",
        role: "row",
        kind: "lane",
      },
      {
        id: "ev-admission",
        label: "ev-admission",
        role: "listitem",
        kind: "point",
        description: "Point event on lane admission",
      },
      {
        id: "lifecycle:lane:connect",
        label: "connect",
        role: "row",
        kind: "lane",
      },
      {
        id: "ev-connect",
        label: "ev-connect",
        role: "listitem",
        kind: "interval",
        description: "Interval event on lane connect",
      },
      {
        id: "lifecycle:lane:first-token",
        label: "first-token",
        role: "row",
        kind: "lane",
      },
      {
        id: "ev-first-token",
        label: "ev-first-token",
        role: "listitem",
        kind: "point",
        description: "Point event on lane first-token",
      },
    ]);

    expect(contribution.semanticRelations).toEqual([
      {
        id: "lifecycle:rel:ev-arrival",
        fromId: "lifecycle:lane:arrival",
        toId: "ev-arrival",
        label: "contains",
        role: "contains",
      },
      {
        id: "lifecycle:rel:ev-admission",
        fromId: "lifecycle:lane:admission",
        toId: "ev-admission",
        label: "contains",
        role: "contains",
      },
      {
        id: "lifecycle:rel:ev-connect",
        fromId: "lifecycle:lane:connect",
        toId: "ev-connect",
        label: "contains",
        role: "contains",
      },
      {
        id: "lifecycle:rel:ev-first-token",
        fromId: "lifecycle:lane:first-token",
        toId: "ev-first-token",
        label: "contains",
        role: "contains",
      },
    ]);

    expect(contribution.hitRegions).toEqual([
      {
        id: "lifecycle:ev-arrival:hit",
        semanticId: "ev-arrival",
        order: 3,
        bounds: { x: 0, y: 0, width: 1, height: 16 },
      },
      {
        id: "lifecycle:ev-admission:hit",
        semanticId: "ev-admission",
        order: 4,
        bounds: { x: 2, y: 20, width: 1, height: 16 },
      },
      {
        id: "lifecycle:ev-connect:hit",
        semanticId: "ev-connect",
        order: 5,
        bounds: { x: 2, y: 40, width: 16, height: 16 },
      },
      {
        id: "lifecycle:ev-first-token:hit",
        semanticId: "ev-first-token",
        order: 6,
        bounds: { x: 120, y: 60, width: 1, height: 16 },
      },
    ]);
  });

  test("keeps nested intervals, points, and open spans under stable semantic ids", () => {
    const contribution = contributeWaterfall({
      id: "lifecycle",
      atMs: 10,
      events: [
        { id: "ev-arrival", lane: "arrival", start: 0, end: 0 },
        {
          id: "ev-inflight",
          lane: "connect",
          start: 2,
          end: 2,
          open: true,
          label: "In flight",
        },
      ],
      layout: {
        laneOrder: ["arrival", "connect"],
        originX: 0,
        originY: 0,
        laneHeight: 16,
        laneGap: 4,
        pxPerMs: 1,
      },
    });

    expect(contribution.semanticEntities.map(({ id, kind, label }) => ({
      id,
      kind,
      label,
    }))).toEqual([
      { id: "lifecycle:lane:arrival", kind: "lane", label: "arrival" },
      { id: "ev-arrival", kind: "point", label: "ev-arrival" },
      { id: "lifecycle:lane:connect", kind: "lane", label: "connect" },
      { id: "ev-inflight", kind: "open-interval", label: "In flight" },
    ]);
    expect(
      contribution.commands
        .flatMap((command) => (command.kind === "group" ? command.children : []))
        .find((command) => command.id === "lifecycle:ev-inflight"),
    ).toMatchObject({
      paintBounds: { x: 2, y: 20, width: 8, height: 16 },
    });
    expect(contribution.semanticRelations.map(({ toId }) => toId)).toEqual([
      "ev-arrival",
      "ev-inflight",
    ]);
  });

  test("direct seek equals continuous playback at the same integer time", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });

    let continuous = evaluate(0);
    for (let atMs = 1; atMs <= 18; atMs += 1) {
      continuous = evaluate(atMs);
    }

    expect(evaluate(18)).toEqual(continuous);
    expect(dateNow).not.toHaveBeenCalled();
    expect(evaluate(2).hitRegions.map(({ semanticId }) => semanticId)).toEqual([
      "ev-arrival",
      "ev-admission",
      "ev-connect",
    ]);
    expect(evaluate(120).hitRegions).toHaveLength(4);
  });

  test("reduced motion drops only decorative playhead chrome", () => {
    const withMotion = contributeWaterfall({
      ...baseInput,
      atMs: 10,
      reducedMotion: false,
    });
    const reduced = contributeWaterfall({
      ...baseInput,
      atMs: 10,
      reducedMotion: true,
    });

    expect(
      withMotion.commands.some((command) => command.id === "lifecycle:playhead"),
    ).toBe(true);
    expect(
      reduced.commands.some((command) => command.id === "lifecycle:playhead"),
    ).toBe(false);
    expect(reduced.semanticEntities).toEqual(withMotion.semanticEntities);
    expect(reduced.semanticRelations).toEqual(withMotion.semanticRelations);
    expect(reduced.hitRegions).toEqual(
      withMotion.hitRegions.filter(({ id }) => !id.includes("playhead")),
    );
  });

  test("returns deeply immutable finite-serializable products", () => {
    const contribution = evaluate(18);
    const serialized = JSON.parse(JSON.stringify(contribution));

    expect(serialized).toEqual(contribution);
    expect(Object.isFrozen(contribution)).toBe(true);
    expect(Object.isFrozen(contribution.commands)).toBe(true);
    expect(Object.isFrozen(contribution.commands[0])).toBe(true);
    expect(Object.isFrozen(contribution.semanticEntities[0])).toBe(true);
    expect(Object.isFrozen(contribution.semanticRelations[0])).toBe(true);
    expect(Object.isFrozen(contribution.hitRegions[0]?.bounds)).toBe(true);
  });

  test("rejects non-integer authored time and non-finite geometry", () => {
    expect(() => evaluate(1.5)).toThrow(
      "Waterfall evaluation time must be a non-negative safe integer",
    );
    expect(() =>
      contributeWaterfall({
        ...baseInput,
        atMs: 0,
        layout: { ...layout, pxPerMs: Number.POSITIVE_INFINITY },
      }),
    ).toThrow("Waterfall layout values must be finite and non-negative");
  });
});
