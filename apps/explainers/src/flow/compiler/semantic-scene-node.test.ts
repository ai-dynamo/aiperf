/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { sceneNodeSchema } from "../schema/ir.js";
import { lowerSemanticSceneNode } from "./semantic-scene-node.js";

const SOURCE_MAP = {
  source: "semantic.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

describe("semantic Scene node lowering", () => {
  it("retains panel payload without generating visual children", () => {
    const panel = lowerSemanticSceneNode(
      {
        id: "panel",
        capability: "core.panel",
        geometry: { x: 10, y: 20, width: 160, height: 64 },
        title: "Profile",
        detail: "source",
      },
      {
        id: "panel",
        capability: "core.panel",
        children: [],
        label: "Profile",
        fallback: "Profile",
        sourceMap: SOURCE_MAP,
      },
    );

    expect(panel).toMatchObject({
      id: "panel",
      kind: "group",
      capabilityId: "core.panel",
      props: { title: "Profile", detail: "source" },
      children: [],
    });
    expect(sceneNodeSchema.parse(panel)).toEqual(panel);
  });

  it("retains semantic stepper labels and linkage", () => {
    const stepper = lowerSemanticSceneNode(
      {
        id: "steps",
        capability: "core.stepper",
        geometry: { x: 0, y: 0, width: 160, height: 90 },
        steps: ["layout", "slots", "timeline"],
        linked: true,
        gap: 16,
      },
      {
        id: "steps",
        capability: "core.stepper",
        children: [],
        label: "stepper",
        fallback: "stepper",
        sourceMap: SOURCE_MAP,
      },
    );

    expect(stepper).toMatchObject({
      capabilityId: "core.stepper",
      props: {
        steps: ["layout", "slots", "timeline"],
        linked: true,
        gap: 16,
      },
      children: [],
    });
    expect(sceneNodeSchema.parse(stepper)).toEqual(stepper);
  });

  it("retains an edge-bound motion signal without synthetic endpoints", () => {
    const signal = lowerSemanticSceneNode(
      {
        id: "motion",
        capability: "motion.signal",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { motion: "signal", markerEnd: "none" },
        edgeRef: "request-credit",
      },
      {
        id: "motion",
        capability: "motion.signal",
        children: [],
        label: "motion signal",
        fallback: "motion signal",
        sourceMap: SOURCE_MAP,
      },
    );

    expect(signal).toMatchObject({
      id: "motion",
      kind: "connector",
      capabilityId: "motion.signal",
      edgeRef: "request-credit",
    });
    expect(signal).not.toHaveProperty("from");
    expect(signal).not.toHaveProperty("to");
    expect(signal).not.toHaveProperty("props.edgeRef");
    expect(sceneNodeSchema.parse(signal)).toEqual(signal);
  });

  it("rejects missing and mixed connector modes in the strict schema", () => {
    const base = {
      id: "connector",
      kind: "connector",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      style: {},
      accessibility: { label: "connector" },
      fallback: "connector",
      sourceMap: SOURCE_MAP,
    } as const;

    expect(
      sceneNodeSchema.safeParse({
        ...base,
        capabilityId: "core.connector",
      }).success,
    ).toBe(false);
    expect(
      sceneNodeSchema.safeParse({
        ...base,
        capabilityId: "core.connector",
        from: { x: 0, y: 0 },
        to: { x: 1, y: 1 },
        edgeRef: "edge",
      }).success,
    ).toBe(false);
    expect(
      sceneNodeSchema.safeParse({
        ...base,
        capabilityId: "motion.signal",
        edgeRef: "edge",
        path: "M0 0 L1 1",
      }).success,
    ).toBe(false);
    expect(
      sceneNodeSchema.safeParse({
        ...base,
        capabilityId: "motion.signal",
        from: { x: 0, y: 0 },
      }).success,
    ).toBe(false);
  });
});

