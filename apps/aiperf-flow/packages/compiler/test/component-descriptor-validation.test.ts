/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentDescriptor, SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { validateProps } from "../src/components.js";

const descriptor: ComponentDescriptor = {
  id: "viz.queue",
  symbolExport: "Queue",
  version: "1.0.0",
  classification: "hybrid",
  props: {
    capacity: { type: "number", required: true },
    label: { type: "string", required: false },
  },
  slots: {},
  events: [],
  capabilityId: "viz.queue",
  leafId: "viz.queue.policy",
  deterministic: true,
};

const range: SourceRange = {
  source: "<test>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

describe("descriptor-backed component prop validation", () => {
  test("accepts props validated directly against a ComponentDescriptor", () => {
    const result = validateProps(
      { capacity: 8, label: "Ready queue" },
      descriptor,
      range,
    );

    expect(result.ok).toBe(true);
  });

  test("preserves existing diagnostics for descriptor-backed props", () => {
    const result = validateProps(
      { capacity: "eight", unexpected: true },
      descriptor,
      range,
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "STRICT_UNKNOWN_PROP" }),
        expect.objectContaining({ code: "PROP_TYPE_MISMATCH" }),
      ]),
    );
  });
});
