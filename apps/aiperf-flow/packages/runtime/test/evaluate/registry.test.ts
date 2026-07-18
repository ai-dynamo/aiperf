// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ComponentNodeIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import type {
  CapabilityContribution,
  CapabilityEvaluator,
} from "../../src/evaluate/registry.js";
import {
  CapabilityEvaluatorRegistry,
  DuplicateCapabilityEvaluatorError,
  UnknownCapabilityEvaluatorError,
} from "../../src/evaluate/registry.js";

const sourceMap = {
  source: "registry.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function component(capabilityId = "core.glyph-run"): ComponentNodeIr {
  return {
    kind: "component",
    id: "prompt",
    capabilityId,
    geometry: { x: 0, y: 0, width: 100, height: 20 },
    style: {},
    accessibility: { label: "Prompt" },
    fallback: "Prompt unavailable",
    sourceMap,
    props: {},
    children: [],
  };
}

function evaluator(capabilityId: string): CapabilityEvaluator {
  return {
    capabilityId,
    evaluate(node, context): CapabilityContribution {
      return {
        layout: node.layoutPlan,
        display: { commands: [], hitRegions: [] },
        semantic: {
          entities: [],
          relations: [],
          readingOrder: [`${node.id}@${context.atMs}`],
        },
      };
    },
  };
}

describe("CapabilityEvaluatorRegistry", () => {
  test("provides typed backend-neutral evaluators through a frozen lookup", () => {
    const registry = new CapabilityEvaluatorRegistry();
    registry.register(evaluator("core.glyph-run"));

    const lookup = registry.freeze();
    const contribution = lookup
      .require("core.glyph-run")
      .evaluate(component(), { atMs: 17 });

    expect(contribution.semantic.readingOrder).toEqual(["prompt@17"]);
    expect(lookup.capabilityIds()).toEqual(["core.glyph-run"]);
    expect(Object.isFrozen(lookup)).toBe(true);
  });

  test("rejects duplicate batches transactionally", () => {
    const registry = new CapabilityEvaluatorRegistry([
      evaluator("core.glyph-run"),
    ]);

    expect(() =>
      registry.registerAll([
        evaluator("viz.queue"),
        evaluator("core.glyph-run"),
      ]),
    ).toThrow(DuplicateCapabilityEvaluatorError);

    expect(registry.freeze().capabilityIds()).toEqual(["core.glyph-run"]);
  });

  test("rejects duplicates within one batch without partial registration", () => {
    const registry = new CapabilityEvaluatorRegistry();

    expect(() =>
      registry.registerAll([
        evaluator("viz.queue"),
        evaluator("viz.queue"),
      ]),
    ).toThrow('Capability evaluator "viz.queue" is already registered.');

    expect(registry.freeze().capabilityIds()).toEqual([]);
  });

  test("freezes a snapshot and rejects later registry mutation", () => {
    const registry = new CapabilityEvaluatorRegistry([
      evaluator("core.glyph-run"),
    ]);
    const lookup = registry.freeze();

    expect(() => registry.register(evaluator("viz.queue"))).toThrow(
      "Capability evaluator registry is frozen.",
    );
    expect(lookup.capabilityIds()).toEqual(["core.glyph-run"]);
  });

  test("reports unknown capability ids clearly", () => {
    const lookup = new CapabilityEvaluatorRegistry().freeze();

    expect(() => lookup.require("viz.missing")).toThrow(
      new UnknownCapabilityEvaluatorError("viz.missing"),
    );
  });
});
